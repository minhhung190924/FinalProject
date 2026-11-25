"""
student_web.py
Web dành cho THÍ SINH (còn teacher_web.py là dành cho giáo viên/admin):
  /          Trang chủ: chọn "Đăng ký khuôn mặt" hoặc "Vào thi"
  /register  Đăng ký khuôn mặt bằng camera trình duyệt (1 tư thế nhìn thẳng)
  /exam      Xác thực khuôn mặt -> làm bài -> nộp bài, camera giám sát suốt buổi thi

Cách hoạt động: trình duyệt mở camera, cứ ~4 lần/giây chụp 1 khung hình gửi về đây.
Server dùng lại đúng code AI cũ (face_utils.py + MediaPipe) để phân tích, ghi vào
exam_system.db qua db.py - nên 3_view_data.py và teacher_web.py xem được ngay.

Chạy:   python student_web.py            -> http://127.0.0.1:5001   (1 máy)
        python student_web.py --host 0.0.0.0 --https   (nhiều máy trong phòng, xem README bên dưới)
Lưu ý: trình duyệt CHỈ cho mở camera trên localhost hoặc HTTPS.

BẢN SỬA (hiệu năng + phát hiện gian lận):
  * FaceMesh được giữ theo từng phiên (trước đây mỗi request tạo lại một FaceMesh mới vì
    Flask threaded sinh thread mới cho mỗi request -> rất chậm và rò bộ nhớ).
  * InsightFace luôn chạy (trước đây chỉ chạy khi MediaPipe thấy mặt).
  * Xác nhận hành vi bằng cửa sổ bỏ phiếu theo thời gian, không còn bị reset khi frame thưa.
  * Hiệu chuẩn tư thế lấy sẵn từ lúc chờ xác thực, bắt đầu thi là có baseline ngay.
  * Cảnh báo (mức 1) cũng được ghi vào database với severity="warning".
  * Server bỏ qua frame đến khi đang xử lý frame trước, trả lại kết quả gần nhất (không dồn hàng đợi).
"""

import argparse
import base64
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request

import db
import face_utils
from exam_config import EXAM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
VIOLATIONS_DIR = os.path.join(BASE_DIR, "violation_evidence")
ANSWERS_DIR = os.path.join(BASE_DIR, "exam_answers")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # mỗi request tối đa 2MB

# ---------------- Thông số AI ----------------
PROC_W, PROC_H = 640, 480
MAX_FACES = 2   # MediaPipe chỉ cần bám mặt chính + 1 mặt phụ; InsightFace đếm thêm người

LOG_COOLDOWN_SECONDS = 5
# Khi đã xác thực và đang thi: InsightFace (nặng nhất) chỉ chạy mỗi chừng này giây để kiểm tra danh tính.
# Các frame còn lại chỉ chạy MediaPipe (nhẹ) nên phát hiện quay đầu nhanh và mượt hơn nhiều.
# MediaPipe thấy != 1 khuôn mặt thì InsightFace vẫn chạy ngay.
INSIGHT_INTERVAL = 1.0
LOCK_FRAMES = 3            # số frame khớp liên tiếp để chốt danh tính (trước đây 5)
STRAIGHT_FRAMES = 3         # giảm thời gian chờ: 3 frame liên tiếp nhìn thẳng + yên
STRAIGHT_MAX_YAW = 12        # độ: lệch ngang tối đa so với hướng thẳng camera để coi là nhìn thẳng
STRAIGHT_MAX_PITCH = 25      # độ: ngẩng/cúi tối đa (camera đặt thấp/cao nên để rộng hơn)
STRAIGHT_MAX_RANGE = 6       # độ: đầu dao động tối đa trong các frame đó (đang ngồi yên)
CALIB_TIMEOUT = 4.0        # giây: quá thời gian này thì chốt mốc bằng các frame hiện có (nếu không quay lệch quá nhiều)
CALIB_MAX_RANGE = 10        # độ: đầu dao động quá mức này thì chưa coi là đang ngồi yên
CALIB_MAX_ABS_YAW = 20     # độ: quay lệch quá mức này (so với thẳng camera) thì chưa coi là nhìn thẳng
CALIB_FRAMES = 4           # hiệu chuẩn dự phòng khi chưa có dữ liệu từ lúc chờ
PRE_CALIB_MIN = 5          # tối thiểu bấy nhiêu frame lúc chờ để dùng làm baseline ngay
MISMATCH_SECONDS = 2.0
IDENTITY_FAIL_SIM = 0.25   # cosine thấp hơn mức này so với người đã xác thực => bằng chứng "người khác"
STALE_SECONDS = 45             # chưa bắt đầu thi: quá chừng này không có khung hình thì dọn phiên
STARTED_STALE_SECONDS = 300    # ĐANG thi: chịu được mất kết nối/máy ngủ tới chừng này giây (5 phút)
SUBMIT_GRACE_SECONDS = 30      # nộp trễ quá hạn tối đa chừng này giây vẫn coi là đúng giờ (độ trễ mạng)
EXPIRED_KEEP_SECONDS = 12 * 3600   # giữ thông tin phiên đã hết hạn để thí sinh vẫn nộp được bài


# ---- Ngưỡng góc đầu (độ lệch so với baseline) ----
# dyaw > 0: quay trái; dyaw < 0: quay phải.

# Quay đầu (độ): dưới 25 bình thường | 25-40 cảnh báo | từ 40 trở lên vi phạm. Muốn xa hơn thì tăng các số này.
YAW_WARNING_LEFT = 25
YAW_CHEAT_LEFT = 40

YAW_WARNING_RIGHT = 25
YAW_CHEAT_RIGHT = 40
DIAG_YAW_SCALE = 18   # thang riêng cho hướng chéo, giữ nguyên độ nhạy cũ

YAW_CHEAT = min(YAW_CHEAT_LEFT, YAW_CHEAT_RIGHT)

# Khi quay quá mạnh khiến MediaPipe mất landmark
LOST_TRACK_MIN_OFFSET = 15
LOST_TRACK_FRESH_SECONDS = 3.0

# Ngẩng đầu / cúi đầu
PITCH_UP_WARNING = 12
PITCH_UP_CHEAT = 20

PITCH_DOWN_WARNING = -14
PITCH_DOWN_CHEAT = -23

# Hướng nhìn của mắt
GAZE_H_WARNING_LOW = 0.20
GAZE_H_WARNING_HIGH = 0.80
GAZE_V_DOWN_WARNING = 0.84

# ---- Hướng mặt chéo ----
DIAG_ANGLE_MIN, DIAG_ANGLE_MAX = 22.5, 67.5
DIAG_R_CHEAT = 0.75
DIAG_R_WARNING = 0.45

# ---- Nghiêng đầu ----
ROLL_WARNING_ALONE = 22
ROLL_COMBO_CHEAT = 15
ROLL_COMBO_MIN_R = 0.35


# Xác nhận hành vi bằng cửa sổ bỏ phiếu: trong VOTE_WINDOW_SECONDS gần nhất, hành vi phải
# xuất hiện >= VOTE_MIN_FRAMES frame, chiếm >= VOTE_MIN_RATIO số frame, và kéo dài >= CONFIRM_SECONDS.
VOTE_WINDOW_SECONDS = 3.0
VOTE_MIN_FRAMES = 2
VOTE_MIN_RATIO = 0.5
CONFIRM_SECONDS = {
    "NO_FACE": 1.2, "MULTI_FACE": 0.8,
    "LOOK_AWAY_LEFT": 0.3, "LOOK_AWAY_RIGHT": 0.3,
    "LOOK_AWAY_UP": 0.8, "LOOK_AWAY_DOWN": 0.8,
    "LOOK_AWAY_UP_LEFT": 0.8, "LOOK_AWAY_UP_RIGHT": 0.8,
    "LOOK_AWAY_DOWN_LEFT": 0.8, "LOOK_AWAY_DOWN_RIGHT": 0.8,
    "LOOK_AWAY_TILT": 1.0,
    "WARNING_DIAG": 0.6, "WARNING_TILT_ROLL": 1.0,
    "WARNING_TILT_LEFT": 0.5, "WARNING_TILT_RIGHT": 0.5,
    "WARNING_TILT_UP": 0.5, "WARNING_TILT_DOWN": 0.5,
    "EYE_LOOK_AWAY": 1.5,
}
DEFAULT_CONFIRM_SECONDS = 1.0

# ---- CHẾ ĐỘ TEST: phát hiện ngay lập tức ----
# True  = thấy hành vi nghi ngờ ở frame nào là ghi vi phạm/cảnh báo ngay ở frame đó (không chờ, không bỏ phiếu).
# False = dùng lại thời gian xác nhận ở trên. Khi chạy thật hãy đặt False.
INSTANT_DETECTION = False

# Log góc đo và quyết định AI (mặc định tắt; chỉ bật bằng biến môi trường DEBUG_DETECTION=1).
DEBUG_DETECTION = os.getenv("DEBUG_DETECTION", "0").strip().lower() in ("1", "true", "yes", "on")
ENABLE_SELFTEST = os.getenv("ENABLE_SELFTEST", "0").strip().lower() in ("1", "true", "yes", "on")
CODE_VERSION = "2026-11-23-f"
if INSTANT_DETECTION:
    MISMATCH_SECONDS = 0.0   # đổi người cũng báo ngay

# Vi phạm do trình duyệt báo về (JavaScript theo dõi tab / toàn màn hình)
BROWSER_EVENTS = {
    "TAB_SWITCH": "Thí sinh chuyển sang tab/cửa sổ khác.",
    "WINDOW_BLUR": "Cửa sổ thi mất tiêu điểm (bấm ra ngoài trang thi).",
    "FULLSCREEN_EXIT": "Thí sinh thoát chế độ toàn màn hình.",
    "COPY_PASTE": "Thí sinh thao tác sao chép/dán.",
}

# ---------------- Đăng ký khuôn mặt: chỉ cần nhìn thẳng ----------------
REGISTER_STEPS = [
    {"id": "LOOK_STRAIGHT", "prompt": "Nhìn thẳng vào camera"},
]
STRAIGHT_MIN_FRAMES = 5       # cần tối thiểu bấy nhiêu khung hình "nhìn thẳng" để lấy đặc trưng trung bình

reg_sessions = {}
reg_lock = threading.Lock()
exam_sessions = {}
expired_sessions = {}          # token -> thông tin phiên ĐANG THI bị dọn do mất kết nối (để vẫn nộp bài được)
exam_lock = threading.Lock()


# =====================================================================
# Tiện ích xử lý ảnh + MediaPipe
# =====================================================================
def _close_mesh(mesh):
    if mesh is None:
        return
    try:
        mesh.close()
    except Exception:
        pass


def new_mesh(max_faces):
    mesh = face_utils.get_face_mesh(max_num_faces=max_faces, static_image_mode=False)
    if mesh is None:
        raise RuntimeError("Thiếu thư viện mediapipe. Hãy chạy: pip install mediapipe==0.10.21")
    return mesh


def decode_frame(data_url):
    """dataURL/base64 -> ảnh BGR 640x480, cắt giữa về tỉ lệ 4:3 (không làm méo mặt).
    Mọi camera đều được đưa về cùng tỉ lệ để đặc trưng khuôn mặt so sánh được giữa các máy."""
    try:
        b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None:
        return None
    h, w = img.shape[:2]
    if w * 3 > h * 4:            # quá rộng -> cắt bớt 2 bên
        new_w = h * 4 // 3
        x0 = (w - new_w) // 2
        img = img[:, x0:x0 + new_w]
    elif w * 3 < h * 4:          # quá cao -> cắt bớt trên/dưới
        new_h = w * 3 // 4
        y0 = (h - new_h) // 2
        img = img[y0:y0 + new_h, :]
    return cv2.resize(img, (640, 480))


def run_mesh(img, mesh):
    """Trả về danh sách khuôn mặt (lớn nhất trước): {'area', 'coords' (pixel), 'norm' (0-1)}.
    `mesh` là FaceMesh của riêng phiên này (luôn được gọi dưới khóa của phiên)."""
    proc = img if img.shape[1] == PROC_W and img.shape[0] == PROC_H else cv2.resize(img, (PROC_W, PROC_H))
    res = mesh.process(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
    faces = []
    if res.multi_face_landmarks:
        for lm in res.multi_face_landmarks:
            coords = face_utils.landmarks_to_pixel_array(lm, PROC_W, PROC_H)
            area = (coords[:, 0].max() - coords[:, 0].min()) * (coords[:, 1].max() - coords[:, 1].min())
            faces.append({"area": area, "coords": coords, "norm": face_utils.landmarks_to_norm_array(lm)})
        faces.sort(key=lambda f: -f["area"])
    return faces


def jpeg_bytes(img, quality=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def clean_name(raw):
    """Tên được dùng làm tên thư mục ảnh -> chỉ cho chữ, số, _ và - (chặn ký tự đường dẫn)."""
    name = re.sub(r"\s+", "_", (raw or "").strip())
    return name if re.fullmatch(r"[\w\-]{1,50}", name) else None


# =====================================================================
# Trang giao diện
# =====================================================================
@app.route("/")
def home():
    return render_template("student/home.html")


@app.route("/register")
def register_page():
    return render_template("student/register.html")


@app.route("/exam")
def exam_page():
    return render_template("student/exam.html")


# =====================================================================
# API đăng ký khuôn mặt
# =====================================================================
@app.route("/api/register/start", methods=["POST"])
def register_start():
    name = clean_name((request.get_json(silent=True) or {}).get("name"))
    if not name:
        return jsonify(ok=False, message="Tên chỉ gồm chữ, số, dấu gạch dưới hoặc gạch nối (tối đa 50 ký tự)."), 400
    if db.candidate_exists(name):
        return jsonify(ok=False, message=f"Tên '{name}' đã được đăng ký. Hãy dùng tên khác (ví dụ thêm mã số sinh viên)."), 409

    token = secrets.token_urlsafe(16)
    now = time.time()
    with reg_lock:
        for t in [t for t, s in reg_sessions.items() if now - s["t"] > 900]:
            _close_mesh(reg_sessions[t].get("mesh"))
            del reg_sessions[t]
        reg_sessions[token] = {
            "name": name, "step": 0, "t": now, "lock": threading.Lock(),
            "buf": deque(maxlen=10),     # (yaw, pitch, features) của các khung hình "nhìn thẳng" gần nhất
            "base": None, "features": None, "photos": [], "mesh": None,
        }
    return jsonify(ok=True, token=token, name=name, steps=REGISTER_STEPS)


@app.route("/api/register/frame", methods=["POST"])
def register_frame():
    data = request.get_json(silent=True) or {}
    st = reg_sessions.get(data.get("token"))
    if st is None:
        return jsonify(ok=False, message="Phiên đăng ký đã hết hạn. Hãy tải lại trang."), 404
    img = decode_frame(data.get("image", ""))
    if img is None:
        return jsonify(ok=False, message="Ảnh không hợp lệ."), 400
    with st["lock"]:
        result = process_register_frame(st, img, bool(data.get("capture")))
        if result.get("done"):
            _close_mesh(st.pop("mesh", None))
            with reg_lock:
                reg_sessions.pop(data["token"], None)
    return jsonify(result)


def process_register_frame(st, img, capture):
    st["t"] = time.time()
    if st["step"] >= len(REGISTER_STEPS):
        return {"ok": False, "message": "Đăng ký đã hoàn tất."}

    if st.get("mesh") is None:
        st["mesh"] = new_mesh(1)

    sid = REGISTER_STEPS[st["step"]]["id"]
    faces = run_mesh(img, st["mesh"])
    has_face = bool(faces)
    yaw = pitch = dyaw = dpitch = 0.0
    pose_valid = False
    embedding_count = None

    if has_face:
        face = faces[0]

        # Dùng solvePnP để lấy yaw/pitch theo ĐỘ.
        # Không dùng compute_turn_ratio() ở đây vì nó trả về landmark ratio.
        pose = face_utils.compute_head_pose_checked(
            face["coords"], PROC_W, PROC_H
        )

        if pose is not None:
            yaw, pitch, _ = pose
            yaw = float(yaw)
            pitch = float(pitch)
            pose_valid = True

            if st["base"] is not None:
                dyaw = yaw - st["base"][0]
                dpitch = pitch - st["base"][1]

    # Ở bước nhìn thẳng, kiểm tra InsightFace độc lập để biết rõ bộ phát hiện nào bị lỗi.
    if sid == "LOOK_STRAIGHT":
        try:
            embeddings = face_utils.extract_face_embeddings(img)
            embedding_count = len(embeddings)
        except Exception as exc:
            return {
                "ok": False, "has_face": has_face, "step": st["step"],
                "step_id": sid, "angle_ok": False, "captured": False,
                "done": False,
                "message": f"Lỗi khởi tạo/nhận diện InsightFace: {type(exc).__name__}: {exc}"
            }

        # Chỉ lấy embedding khi MediaPipe và InsightFace đều xác nhận một khuôn mặt.
        if has_face and embedding_count == 1:
            st["buf"].append((yaw, pitch, embeddings[0]))
        else:
            st["buf"].clear()

        if sid == "LOOK_STRAIGHT":
            ok = has_face and pose_valid and embedding_count == 1

            if not has_face:
                hint = "MediaPipe chưa thấy mặt. Hãy đưa mặt vào giữa khung hình, đủ sáng và nhìn rõ camera."
            elif not pose_valid:
                hint = "Chưa đo được góc khuôn mặt. Hãy nhìn thẳng và ngồi gần camera hơn."
            elif embedding_count == 0:
                hint = "MediaPipe đã thấy mặt nhưng InsightFace chưa nhận được. Hãy ngồi gần camera hơn, bỏ khẩu trang/kính râm và tăng ánh sáng."
            elif embedding_count > 1:
                hint = "Phát hiện nhiều khuôn mặt. Chỉ để một người trong khung hình."
            else:
                hint = "Đã nhận diện được khuôn mặt - giữ thẳng và bấm Chụp."

    result = {
        "ok": True, "has_face": has_face, "step": st["step"], "step_id": sid,
        "angle_ok": bool(ok), "dyaw": round(dyaw), "dpitch": round(dpitch),
        "message": "Đúng tư thế - giữ yên..." if ok else hint,
        "captured": False, "done": False,
    }
    if not (capture and ok):
        return result

    if sid == "LOOK_STRAIGHT":
        if len(st["buf"]) < STRAIGHT_MIN_FRAMES:
            result["message"] = (
                f"Đã nhận diện được mặt ({len(st['buf'])}/{STRAIGHT_MIN_FRAMES}). "
                "Giữ thẳng và bấm Chụp lại."
            )
            return result
        samples = list(st["buf"])
        st["base"] = (
            float(np.mean([sample[0] for sample in samples])),
            float(np.mean([sample[1] for sample in samples]))
        )
        st["features"] = face_utils.average_embeddings([sample[2] for sample in samples])

    st["photos"].append(jpeg_bytes(img))
    st["step"] += 1
    result.update(captured=True, step=st["step"], message="Đã chụp xong tư thế này.")

    if st["step"] >= len(REGISTER_STEPS):
        candidate_id = finish_registration(st)
        if candidate_id is None:
            result.update(ok=False, message="Tên này vừa được người khác đăng ký. Hãy đăng ký lại với tên khác.")
        else:
            result.update(done=True, candidate_id=candidate_id, name=st["name"], message="Đăng ký thành công!")
    return result


def finish_registration(st):
    try:
        candidate_id = db.add_candidate(st["name"], st["features"])
    except sqlite3.IntegrityError:
        return None
    folder = os.path.join(DATASET_DIR, st["name"])
    os.makedirs(folder, exist_ok=True)
    for i, jpg in enumerate(st["photos"], 1):
        with open(os.path.join(folder, f"face_{i}.jpg"), "wb") as f:
            f.write(jpg)
    return candidate_id


# =====================================================================
# API thi + giám sát
# =====================================================================
class ExamState:
    """Trạng thái giám sát của 1 thí sinh đang ở trang thi."""

    def __init__(self, candidates):
        self.candidates = candidates
        self.t = time.time()
        self.lock = threading.Lock()
        self.mesh = None                      # FaceMesh riêng của phiên, tạo lười ở frame đầu tiên
        self.closed = False
        self.last_result = None
        self.candidate = None                 # dict {"id","full_name","features"} sau khi chốt danh tính
        self.pending_id, self.pending_n, self.pending_miss = None, 0, 0
        self.pending_distances = deque(maxlen=LOCK_FRAMES)
        self.started = False                  # đã bấm "Bắt đầu làm bài" chưa
        self.ended = False
        self.session_id = None
        self.evidence_dir = None
        self.pre_calib = deque(maxlen=12)     # tư thế (yaw, pitch, roll) lúc chờ, đã xác thực, chưa bắt đầu thi
        self.roll_hist = deque(maxlen=3)
        self.last_pose = None                 # (thời điểm, dyaw, dpitch) của lần đo hợp lệ gần nhất
        self.lost_dir = None                  # hướng đang quay đi khi MediaPipe mất landmark
        self.calib, self.baseline = [], None
        # Cửa sổ ngắn để giảm nhiễu nhưng không làm trễ khi thí sinh quay đầu.
        self.yaw_hist, self.pitch_hist = deque(maxlen=3), deque(maxlen=3)
        self.mismatch_since = None
        self.identity_last_match = None
        self.events = deque()                 # (thời điểm, loại hành vi hoặc None) cho bỏ phiếu xác nhận
        self.last_logged = {}
        self.last_insight = 0.0
        self.begin_time = None
        self.answers = None                   # bản lưu tự động gần nhất của đáp án
        self.log_error_reported = False
        self.pre_baseline = None              # mốc nhìn thẳng đo TRƯỚC khi thi; chỉ có khi đang thẳng + yên

        # Trạng thái duy trì vi phạm quay đầu trái/phải
        self.active_turn_type = None
        self.active_turn_message = None

    def get_mesh(self):
        if self.mesh is None:
            self.mesh = new_mesh(MAX_FACES)
        return self.mesh

    def close(self):
        self.closed = True
        _close_mesh(self.mesh)
        self.mesh = None


def expire_exam_sessions():
    now = time.time()
    with exam_lock:
        stale = [t for t, s in exam_sessions.items()
                 if now - s.t > (STARTED_STALE_SECONDS if s.started else STALE_SECONDS)]
        for t in stale:
            s = exam_sessions.pop(t)
            if s.session_id is not None and not s.ended:
                db.end_session(s.session_id, status="aborted")
                # Không vứt hẳn: thí sinh vẫn nộp được bài đã làm (xem exam_finish).
                expired_sessions[t] = {
                    "session_id": s.session_id, "candidate": s.candidate["full_name"],
                    "begin_time": s.begin_time or now, "answers": s.answers, "t": now, "finished": False,
                }
            s.close()
        for t in [t for t, r in expired_sessions.items() if now - r["t"] > EXPIRED_KEEP_SECONDS]:
            del expired_sessions[t]


def get_exam(token):
    with exam_lock:
        return exam_sessions.get(token)


def log_violation(st, v_type, message, img=None, severity="violation", force=False):
    """Ghi vi phạm/cảnh báo vào database, có giãn cách chống spam.
    Không bao giờ để lỗi ghi (ví dụ phiên đã bị giáo viên xóa) làm hỏng việc giám sát."""

    if st.session_id is None:
        return

    now = time.time()

    # Chỉ bỏ qua cooldown nếu không phải lần bắt đầu vi phạm mới.
    if not force:
        if now - st.last_logged.get(v_type, 0) < LOG_COOLDOWN_SECONDS:
            return

    st.last_logged[v_type] = now

    try:
        path = None
        if img is not None and st.evidence_dir:
            path = os.path.join(st.evidence_dir, f"{v_type}_{int(now * 1000)}.jpg")
            with open(path, "wb") as f:
                f.write(jpeg_bytes(img))
        db.log_violation(st.session_id, v_type, message, path, severity)
        print(f"[LOG] phiên {st.session_id}: {'CẢNH BÁO' if severity == 'warning' else 'VI PHẠM'} {v_type}")
    except Exception:
        # Chỉ báo một lần để khỏi tràn log mỗi khung hình.
        if not st.log_error_reported:
            st.log_error_reported = True
            app.logger.exception("Không ghi được vi phạm cho phiên %s (phiên có thể đã bị xóa)", st.session_id)


def _vote_group(v_type):
    if v_type is None:
        return None
    if v_type.startswith("LOOK_AWAY_"):
        return "HEAD_VIOLATION"
    if v_type.startswith("WARNING_") or v_type == "EYE_LOOK_AWAY":
        return "HEAD_WARNING"
    return v_type


def confirm_behavior(st, v_type, now):
    """Bỏ phiếu theo cửa sổ thời gian: không bị reset khi frame đến thưa hoặc hụt một vài frame."""
    if INSTANT_DETECTION:
        return v_type is not None
    # Bỏ phiếu theo NHÓM: mặt quay đi ở vùng giáp ranh hay đổi nhãn qua lại (LEFT <-> UP_LEFT <-> TILT...),
    # nếu đếm theo nhãn riêng thì không nhãn nào đủ phiếu và không bao giờ được xác nhận.
    group = _vote_group(v_type)
    # Máy chậm (frame đến cách nhau vài giây): cửa sổ 3 giây chỉ chứa tối đa 1 frame nên không bao giờ đủ phiếu.
    # Vì vậy cửa sổ tự giãn ra theo tốc độ frame thực tế (luôn chứa được ít nhất ~3 frame).
    last_t = getattr(st, "last_vote_t", None)
    if last_t is not None and 0 < now - last_t < 30:
        st.frame_dt = 0.7 * getattr(st, "frame_dt", now - last_t) + 0.3 * (now - last_t)
    st.last_vote_t = now
    window = max(VOTE_WINDOW_SECONDS, 3.0 * getattr(st, "frame_dt", 0.0))
    st.events.append((now, group))
    while st.events and now - st.events[0][0] > window:
        st.events.popleft()
    if v_type is None:
        return False
    same = [t for t, v in st.events if v == group]
    if len(same) < VOTE_MIN_FRAMES:
        return False
    if len(same) / len(st.events) < VOTE_MIN_RATIO:
        return False
    return now - same[0] >= CONFIRM_SECONDS.get(v_type, DEFAULT_CONFIRM_SECONDS)


def head_roll_deg(coords):
    """Góc nghiêng đầu (độ) từ đường nối hai khóe mắt ngoài (landmark 33 và 263), ảnh không lật gương.
    Dương = đầu nghiêng sang TRÁI của thí sinh (mắt bên trái hạ thấp), cùng quy ước dyaw > 0 là trái."""
    p1, p2 = coords[33], coords[263]
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))


def face_direction(dyaw, dpitch):
    """Gộp độ lệch ngang/dọc thành 1 hướng mặt: (r, theta). r = 1 tương đương quay đơn tới ngưỡng vi phạm."""
    ny = dyaw / DIAG_YAW_SCALE
    np_ = dpitch / (PITCH_UP_CHEAT if dpitch >= 0 else -PITCH_DOWN_CHEAT)
    r = float(np.hypot(ny, np_))
    theta = float(np.degrees(np.arctan2(abs(np_), abs(ny))))   # 0 = ngang thuần, 90 = dọc thuần
    return r, theta


def lost_tracking_behavior(st, now):
    """MediaPipe mất landmark nhưng vẫn thấy một khuôn mặt (thường do quay đầu quá mạnh).
    Chỉ coi là vi phạm khi ngay trước lúc mất dấu thí sinh đã lệch rõ; ngược lại là nhiễu tạm thời."""
    if st.lost_dir is None:
        lp = st.last_pose
        if lp is None or now - lp[0] > max(LOST_TRACK_FRESH_SECONDS, 2.5 * getattr(st, "frame_dt", 0.0)):
            return None
        _, dy, dp = lp
        if max(abs(dy), abs(dp)) < LOST_TRACK_MIN_OFFSET:
            return None
        if abs(dy) >= abs(dp):
            st.lost_dir = ("LEFT", "trái") if dy > 0 else ("RIGHT", "phải")
        else:
            st.lost_dir = ("UP", "ngẩng lên") if dp > 0 else ("DOWN", "cúi xuống")
    code, vi = st.lost_dir
    return (2, f"LOOK_AWAY_{code}",
            f"Nghi vấn gian lận: thí sinh quay mặt quá mức ({vi}), hệ thống không còn đo được góc khuôn mặt.")


def classify_pose(dyaw, dpitch, gh, gv, droll=0.0):
    """Phân loại tư thế. Quay đầu được xét bằng góc đầu, không bắt buộc gaze cùng lệch.
    Quy ước: dyaw > 0 là quay sang TRÁI của thí sinh, dpitch > 0 là ngẩng, dpitch < 0 là cúi,
    droll > 0 là nghiêng đầu sang trái."""
    r, theta = face_direction(dyaw, dpitch)
    diagonal = DIAG_ANGLE_MIN <= theta <= DIAG_ANGLE_MAX
    side = "LEFT" if dyaw > 0 else "RIGHT"
    side_vi = "trái" if dyaw > 0 else "phải"
    vert = "UP" if dpitch >= 0 else "DOWN"

    # --- Mặt hướng chéo: xét trước, vì nó nói rõ hướng hơn quay đơn ---
    if diagonal and r >= DIAG_R_CHEAT:
        if dpitch >= 0:
            return (2, f"LOOK_AWAY_UP_{side}",
                    f"Nghi vấn gian lận: thí sinh nhìn chéo lên về phía {side_vi} (nhón/ngẩng lên dòm) và giữ tư thế này.")
        return (2, f"LOOK_AWAY_DOWN_{side}",
                f"Nghi vấn gian lận: thí sinh nhìn chéo xuống về phía {side_vi} (liếc bài bên cạnh) và giữ tư thế này.")

    if dyaw >= YAW_CHEAT_LEFT:
        return 2, "LOOK_AWAY_LEFT", "Nghi vấn gian lận: thí sinh quay đầu sang trái và giữ tư thế này."
    if dyaw <= -YAW_CHEAT_RIGHT:
        return 2, "LOOK_AWAY_RIGHT", "Nghi vấn gian lận: thí sinh quay đầu sang phải và giữ tư thế này."
    if dpitch >= PITCH_UP_CHEAT:
        return 2, "LOOK_AWAY_UP", "Nghi vấn gian lận: thí sinh ngẩng đầu và giữ tư thế này."
    if dpitch <= PITCH_DOWN_CHEAT:
        return 2, "LOOK_AWAY_DOWN", "Nghi vấn gian lận: thí sinh cúi đầu và giữ tư thế này."

    # Nghiêng đầu kèm quay/ngẩng/cúi: dáng "ghé đầu nhìn lén".
    if abs(droll) >= ROLL_COMBO_CHEAT and r >= ROLL_COMBO_MIN_R:
        tilt_vi = "trái" if droll > 0 else "phải"
        return (2, "LOOK_AWAY_TILT",
                f"Nghi vấn gian lận: thí sinh nghiêng đầu sang {tilt_vi} kèm quay/ngẩng/cúi (ghé nhìn) và giữ tư thế này.")

    # Cảnh báo chéo nhẹ: mặt hướng chéo dù chưa tới mức vi phạm.
    if diagonal and r >= DIAG_R_WARNING:
        return 1, "WARNING_DIAG", "Cảnh báo: bạn đang nhìn lệch chéo khỏi màn hình. Hãy nhìn lại màn hình."
    if abs(droll) >= ROLL_WARNING_ALONE:
        return 1, "WARNING_TILT_ROLL", "Cảnh báo: bạn đang nghiêng đầu nhiều. Hãy ngồi thẳng và nhìn lại màn hình."

    if dyaw >= YAW_WARNING_LEFT:
        return 1, "WARNING_TILT_LEFT", "Cảnh báo: bạn đang quay đầu sang trái. Hãy nhìn lại màn hình."
    if dyaw <= -YAW_WARNING_RIGHT:
        return 1, "WARNING_TILT_RIGHT", "Cảnh báo: bạn đang quay đầu sang phải. Hãy nhìn lại màn hình."
    if dpitch >= PITCH_UP_WARNING:
        return 1, "WARNING_TILT_UP", "Cảnh báo: bạn đang ngẩng đầu. Hãy nhìn lại màn hình."
    if dpitch <= PITCH_DOWN_WARNING:
        return 1, "WARNING_TILT_DOWN", "Cảnh báo: bạn đang cúi đầu. Hãy nhìn lại màn hình."

    # Gaze chỉ là tín hiệu phụ; không dùng nó làm điều kiện bắt buộc cho quay đầu.
    if gh <= GAZE_H_WARNING_LOW or gh >= GAZE_H_WARNING_HIGH or gv >= GAZE_V_DOWN_WARNING:
        return 1, "EYE_LOOK_AWAY", "Cảnh báo: hướng nhìn của mắt đang lệch khỏi màn hình."
    return None


def _mesh_face_bbox(face):
    pts = np.asarray(face.get("coords"), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return None
    x1, y1 = np.min(pts[:, :2], axis=0)
    x2, y2 = np.max(pts[:, :2], axis=0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _bbox_match_score(mesh_bbox, insight_bbox):
    """Return (is_match, score) using overlap and center distance in the same 640x480 frame."""
    a = np.asarray(mesh_bbox, dtype=np.float32)
    b = np.asarray(insight_bbox, dtype=np.float32)
    aw, ah = max(1.0, a[2] - a[0]), max(1.0, a[3] - a[1])
    bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    min_area = min(aw * ah, bw * bh)
    overlap = inter / max(1.0, min_area)
    ac = np.array([(a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0])
    bc = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
    scale = max(1.0, float(np.hypot(aw, ah)))
    center_distance = float(np.linalg.norm(ac - bc) / scale)
    matched = overlap >= 0.08 or center_distance <= 0.60
    score = overlap - center_distance * 0.25
    return matched, score


def associate_face_observations(mesh_faces, observations):
    """Map InsightFace observations to MediaPipe faces without assuming equal sort order."""
    assigned = [None] * len(mesh_faces)
    used = set()
    pairs = []
    for mi, mesh_face in enumerate(mesh_faces):
        mb = _mesh_face_bbox(mesh_face)
        if mb is None:
            continue
        for oi, obs in enumerate(observations):
            ib = obs.get("bbox")
            if ib is None:
                continue
            ok, score = _bbox_match_score(mb, ib)
            if ok:
                pairs.append((score, mi, oi))
    for score, mi, oi in sorted(pairs, reverse=True):
        if assigned[mi] is None and oi not in used:
            assigned[mi] = observations[oi]
            used.add(oi)
    # Single-face fallback: tolerate modest detector-box shape differences.
    if len(mesh_faces) == 1 and len(observations) == 1 and assigned[0] is None:
        assigned[0] = observations[0]
        used.add(0)
    unmatched = [obs for i, obs in enumerate(observations) if i not in used]
    return assigned, unmatched


def analyze_exam_frame(st, img):
    t0 = time.time()
    now = t0
    st.t = now

    t_mesh0 = time.time()
    mesh_faces = run_mesh(img, st.get_mesh())
    mesh_ms = int((time.time() - t_mesh0) * 1000)
    insight_ms = 0
    mesh_count = len(mesh_faces)

    # InsightFace luôn chạy (kể cả khi MediaPipe không thấy mặt) để xác thực danh tính
    # và đếm người độc lập với MediaPipe.
    observations = []
    need_insight = (st.candidate is None or mesh_count != 1
                    or now - st.last_insight >= INSIGHT_INTERVAL)
    if need_insight:
        try:
            t_ins0 = time.time()
            observations = face_utils.extract_face_observations(img)
            insight_ms = int((time.time() - t_ins0) * 1000)
            st.last_insight = now
        except Exception as exc:
            app.logger.warning("InsightFace frame error: %s", exc)
            observations = []

    aligned, unmatched_observations = associate_face_observations(mesh_faces, observations)
    main = mesh_faces[0] if mesh_faces else None
    if main is not None:
        main_observation = aligned[0]
    elif len(observations) == 1:
        main_observation = observations[0]
    else:
        main_observation = None

    face_count = max(mesh_count, len(observations))

    # ---------------- Danh tính ----------------
    matched = dist = sim_locked = None
    if main_observation is not None:
        emb = main_observation["embedding"]
        matched, dist = face_utils.match_face(emb, st.candidates, verbose=False)
        if st.candidate is not None:
            _, s = face_utils.match_specific(emb, st.candidate["features"])
            if np.isfinite(s):
                sim_locked = float(s)

    if st.candidate is None:
        # Chốt danh tính: LOCK_FRAMES frame khớp cùng một người, chấp nhận hụt 1 frame xen kẽ.
        if face_count == 1 and matched is not None:
            if matched["id"] == st.pending_id:
                st.pending_n += 1
            else:
                st.pending_id, st.pending_n = matched["id"], 1
                st.pending_distances.clear()
            st.pending_distances.append(float(dist))
            st.pending_miss = 0
        else:
            st.pending_miss += 1
            if face_count != 1 or st.pending_miss >= 2:
                st.pending_id, st.pending_n, st.pending_miss = None, 0, 0
                st.pending_distances.clear()
        if st.pending_id is not None and st.pending_n >= LOCK_FRAMES:
            st.candidate = next((c for c in st.candidates if c["id"] == st.pending_id), None)
            if st.candidate is not None:
                st.identity_last_match = now
                st.pre_calib.clear()
    else:
        # Sau khi đã xác thực: chỉ coi là "người khác" khi có bằng chứng rõ (khớp một người
        # đăng ký khác, hoặc độ giống với thí sinh quá thấp). Vùng lưng chừng (mặt hơi nghiêng,
        # ánh sáng xấu) và frame không có embedding thì giữ nguyên, tránh báo nhầm.
        if sim_locked is not None:
            if sim_locked >= face_utils.MATCH_THRESHOLD:
                st.mismatch_since = None
                st.identity_last_match = now
            else:
                other_person = matched is not None and matched["id"] != st.candidate["id"]
                if other_person or sim_locked < IDENTITY_FAIL_SIM:
                    if st.mismatch_since is None:
                        st.mismatch_since = now

    
        # ---------------- Tư thế đầu ----------------

    dyaw = dpitch = droll = 0.0
    gh = gv = 0.5

    yaw_valid = False
    pitch_valid = False
    roll_valid = False
    gaze_valid = False
    pose_valid = False

    yaw_s = None
    pitch_s = None
    roll_s = None

    if main is not None:

        # ==========================================
        # 1. YAW + PITCH: Dùng solvePnP, đơn vị ĐỘ
        # ==========================================
        pose = face_utils.compute_head_pose_checked(
            main["coords"], PROC_W, PROC_H
        )

        if pose is not None:
            yaw, pitch, _ = pose

            yaw = float(yaw)
            pitch = float(pitch)

            # Quay trái/phải đo từ MŨI (đối xứng hai bên, không bị lệch như solvePnP khi mất điểm bên xa).
            nose_yaw = face_utils.compute_nose_yaw(main["coords"], pitch)
            if nose_yaw is not None:
                yaw = float(nose_yaw)

            st.yaw_hist.append(yaw)
            st.pitch_hist.append(pitch)

            # Làm mượt nhẹ (EMA): bám sát khi quay đầu, vẫn lọc nhiễu; không bị trễ như trung vị 3 khung.
            prev = getattr(st, "yaw_ema", None)
            st.yaw_ema = yaw if prev is None else 0.7 * yaw + 0.3 * prev
            prevp = getattr(st, "pitch_ema", None)
            st.pitch_ema = pitch if prevp is None else 0.7 * pitch + 0.3 * prevp
            yaw_s = float(st.yaw_ema)
            pitch_s = float(st.pitch_ema)

            yaw_valid = True
            pitch_valid = True

        # ==========================================
        # 2. ROLL: Đo nghiêng đầu độc lập
        # ==========================================
        try:
            roll_value = head_roll_deg(main["coords"])

            if np.isfinite(roll_value):
                st.roll_hist.append(float(roll_value))
                roll_s = float(np.median(st.roll_hist))
                roll_valid = True

        except Exception as exc:
            app.logger.debug("Head roll error: %s", exc)

        # Pose tổng hợp hợp lệ khi yaw và pitch đều đo được.
        pose_valid = yaw_valid and pitch_valid

        if pose_valid:
            st.lost_dir = None

    # ==========================================
    # 3. HIỆU CHUẨN BASELINE
    # ==========================================

    calib_ok = (
        face_count == 1
        and pose_valid
        and roll_valid
        and st.candidate is not None
        and st.mismatch_since is None
    )

    # Thu thập tư thế bình thường khi thí sinh đã xác thực
    # nhưng chưa bắt đầu làm bài.
    st.pre_baseline = None
    if not st.started and st.candidate is not None:
        if calib_ok:
            st.pre_calib.append((yaw_s, pitch_s, roll_s))
        else:
            st.pre_calib.clear()        # mất mặt / nhiều mặt / đo lỗi: bỏ dữ liệu cũ, phải thẳng lại từ đầu
        pc = list(st.pre_calib)[-STRAIGHT_FRAMES:]
        if len(pc) >= STRAIGHT_FRAMES:
            ys = [c[0] for c in pc]; ps = [c[1] for c in pc]
            if (max(ys) - min(ys) <= STRAIGHT_MAX_RANGE and max(ps) - min(ps) <= STRAIGHT_MAX_RANGE
                    and abs(float(np.median(ys))) <= STRAIGHT_MAX_YAW
                    and abs(float(np.median(ps))) <= STRAIGHT_MAX_PITCH):
                st.pre_baseline = (float(np.median(ys)), float(np.median(ps)),
                                   float(np.median([c[2] for c in pc])))

    # ==========================================
    # 4. TÍNH ĐỘ LỆCH SO VỚI BASELINE
    # ==========================================

    if st.baseline is not None:

        if yaw_valid:
            dyaw = yaw_s - st.baseline[0]

        if pitch_valid:
            dpitch = pitch_s - st.baseline[1]

        if roll_valid:
            droll = roll_s - st.baseline[2]

        # Chỉ cập nhật last_pose khi cả yaw và pitch hợp lệ.
        # Giữ nguyên cấu trúc dữ liệu để lost_tracking_behavior sử dụng.
        if pose_valid:
            st.last_pose = (now, dyaw, dpitch)

        # Chỉ tính hướng mắt khi có đủ landmark mắt.
        if main is not None:
            coords = np.asarray(main["coords"])

            if coords.ndim == 2 and coords.shape[0] >= 478:
                try:
                    gaze_result = face_utils.compute_eye_gaze_ratio(
                        main["coords"]
                    )

                    if gaze_result is not None:
                        gh, gv = gaze_result
                        gh = float(gh)
                        gv = float(gv)

                        if np.isfinite(gh) and np.isfinite(gv):
                            gaze_valid = True
                        else:
                            gh, gv = 0.5, 0.5

                except Exception as exc:
                    app.logger.debug("Eye gaze error: %s", exc)



    # ---------------- Quyết định hành vi ----------------
    raw = None
    if st.started:
        if st.candidate and st.mismatch_since is not None and now - st.mismatch_since >= MISMATCH_SECONDS:
            raw = (2, "IDENTITY_MISMATCH", "Khuôn mặt không khớp với thí sinh đã xác thực!")
        elif face_count > 1:
            names = []
            for obs in unmatched_observations:
                m2, _ = face_utils.match_face(obs["embedding"], st.candidates, verbose=False)
                names.append(m2["full_name"] if m2 else "không rõ")
            raw = (2, "MULTI_FACE",
                   f"Phát hiện {face_count} khuôn mặt trong camera ({', '.join(names) if names else 'nhiều khuôn mặt'}).")
        elif face_count == 0:
            raw = (2, "NO_FACE", "Không thấy khuôn mặt trong camera (quay đi hoặc che camera).")
        elif st.baseline is not None and pose_valid:
            raw = classify_pose(dyaw, dpitch, gh, gv, droll)

        elif st.baseline is not None and not pose_valid:
            raw = lost_tracking_behavior(st, now)

    # ==========================================================
# DUY TRÌ TRẠNG THÁI VI PHẠM QUAY ĐẦU TRÁI / PHẢI
# ==========================================================

    TURN_TYPES = {
        "LOOK_AWAY_LEFT",
        "LOOK_AWAY_RIGHT",
    }

    new_turn_started = False

    # 1. Xác nhận hành vi
    if raw and raw[1] == "IDENTITY_MISMATCH":
        confirmed = True
        st.events.clear()

    elif raw and raw[1] in TURN_TYPES:
        confirmed = confirm_behavior(st, raw[1], now)
        # Mất landmark do quay hẳn sang bên: báo ngay, không chờ bỏ phiếu (frame lúc này thưa/trống).
        if st.lost_dir is not None and not pose_valid:
            confirmed = True

        # Chỉ bắt đầu một lần vi phạm khi chưa có vi phạm quay đầu đang hoạt động.
        if confirmed and st.active_turn_type is None:
            st.active_turn_type = raw[1]
            st.active_turn_message = raw[2]
            new_turn_started = True

    else:
        confirmed = confirm_behavior(
            st,
            raw[1] if raw else None,
            now
        ) if st.started else False


    # 2. Kiểm tra thí sinh đã quay đầu về bình thường chưa.
    if st.active_turn_type is not None and pose_valid and st.baseline is not None:

        returned_to_normal = False

        if st.active_turn_type == "LOOK_AWAY_LEFT":
            returned_to_normal = dyaw < YAW_WARNING_LEFT

        elif st.active_turn_type == "LOOK_AWAY_RIGHT":
            returned_to_normal = dyaw > -YAW_WARNING_RIGHT

        if returned_to_normal:
            st.active_turn_type = None
            st.active_turn_message = None

            # Xóa dữ liệu bỏ phiếu cũ để tránh kích hoạt lại ngay lập tức.
            st.events.clear()


    # ==========================================================
    # XÁC ĐỊNH TRẠNG THÁI HIỂN THỊ
    # ==========================================================

    calibrating = st.started and st.baseline is None

    if st.active_turn_type is not None:

        # Đang vi phạm quay đầu: luôn giữ trạng thái vi phạm.
        status = 2
        v_type = st.active_turn_type
        message = st.active_turn_message

    elif confirmed and raw is not None:

        status, v_type, message = raw

    elif raw is not None and st.started:
        # Cảnh báo VÀ vi phạm đều hiện NGAY trên màn hình (không chờ xác nhận, tránh hiện "bình thường" lúc
        # đang quay đi); chỉ việc ghi vào database mới chờ xác nhận để khỏi ghi nhầm khi nhiễu một khung hình.
        status, v_type, message = raw

    else:

        status, v_type = 0, None

        if not st.started:
            message = idle_message(
                st,
                face_count,
                main_observation,
                matched,
                dist
            )

        elif calibrating:
            message = "Đang xác định tư thế nhìn thẳng - hãy ngồi yên, nhìn thẳng vào màn hình vài giây."

        elif face_count == 1 and not pose_valid:
            message = "Đang ổn định dữ liệu camera, vui lòng nhìn vào màn hình."

        else:
            message = "Đang làm bài bình thường."


    # ==========================================================
    # GHI VI PHẠM VÀ LƯU ẢNH BẰNG CHỨNG
    # ==========================================================

    if new_turn_started:

        # Chỉ ghi database và chụp ảnh đúng một lần khi bắt đầu vi phạm.
        log_violation(
            st,
            st.active_turn_type,
            st.active_turn_message,
            img,
            severity="violation",
            force=True
        )

    elif st.active_turn_type is None:

        # Những loại vi phạm khác vẫn sử dụng cơ chế ghi nhận cũ.
        if status == 2 and confirmed:
            log_violation(
                st,
                v_type,
                message,
                img
            )

        elif status == 1 and confirmed:
            log_violation(
                st,
                v_type,
                message,
                img,
                severity="warning"
            )

    if DEBUG_DETECTION and st.started and now - getattr(st, "last_debug", 0.0) >= 1.0:
        st.last_debug = now
        line = (f"[DETECT] phiên {st.session_id} | trái(+)/phải(-) {dyaw:+.1f}° | ngẩng(+)/cúi(-) {dpitch:+.1f}° | "
                f"nghiêng {droll:+.1f}° | mesh {mesh_ms}ms insight {insight_ms}ms tổng {int((time.time() - t0) * 1000)}ms | mặt {face_count} | đo được góc: {pose_valid} | raw={raw[1] if raw else None} | "
                f"xác nhận={confirmed} | hiển thị={status}:{v_type}")
        print(line)
        try:   # ghi thêm ra file để gửi lại cho người hỗ trợ khi terminal khó copy
            with open(os.path.join(BASE_DIR, "detect_debug.log"), "a", encoding="utf-8") as _f:
                _f.write(datetime.now().strftime("%H:%M:%S ") + line + "\n")
        except OSError:
            pass

    return {
        "ok": True, "status": status, "type": v_type, "message": message,
        "face_count": face_count, "mesh_face_count": mesh_count,
        "insight_face_count": len(observations),
        "identified": st.candidate is not None,
        "name": st.candidate["full_name"] if st.candidate else None,
        "started": st.started, "calibrating": calibrating,
        "verify_progress": 1.0 if st.candidate else round(min(1.0, st.pending_n / LOCK_FRAMES), 2),
        "verify_count": st.pending_n, "lock_frames": LOCK_FRAMES,
        "pose_valid": pose_valid, "face_visible": face_count > 0,
        "processing_ms": int((time.time() - t0) * 1000),
        "distance": None if dist is None else round(1.0 - float(dist), 2),
        "similarity": None if dist is None else round(float(dist), 2),

        "yaw": round(dyaw, 2),
        "pitch": round(dpitch, 2),
        "roll": round(droll, 2),
        "gaze_h": round(gh, 3),
        "gaze_v": round(gv, 3),

        "yaw_valid": yaw_valid,
        "pitch_valid": pitch_valid,
        "roll_valid": roll_valid,
        "gaze_valid": gaze_valid,
        "baseline_ready": st.baseline is not None,
        "straight_ok": st.pre_baseline is not None,
    }


def idle_message(st, face_count, main_observation, matched, dist):
    if st.candidate:
        if face_count == 1 and st.pre_baseline is not None:
            return "Đã xác thực và đang nhìn thẳng. Bạn có thể bắt đầu làm bài."
        if face_count == 0:
            return "Đã xác thực nhưng không thấy khuôn mặt. Hãy ngồi vào giữa khung hình, nhìn thẳng vào màn hình."
        return "Đã xác thực. Hãy ngồi thẳng, nhìn thẳng vào màn hình và giữ yên để bắt đầu."
    if face_count == 0:
        return "Không thấy khuôn mặt. Hãy ngồi vào giữa khung hình."
    if face_count > 1:
        return "Chỉ một người được ở trước camera."
    if main_observation is None:
        return "Đã thấy khuôn mặt nhưng chưa nhận diện được. Hãy nhìn thẳng, đủ sáng, bỏ khẩu trang/kính râm."
    if st.pending_n > 0:
        return f"Đang xác thực khuôn mặt... ({st.pending_n}/{LOCK_FRAMES})"
    if matched is None and dist is not None:
        return f"Khuôn mặt chưa khớp với ai đã đăng ký (độ giống {float(dist):.2f}). Hãy nhìn thẳng vào camera hoặc đăng ký khuôn mặt trước."
    return "Đang xác thực khuôn mặt..."


def busy_result(st):
    """Trả kết quả gần nhất khi server còn đang xử lý frame trước (không dồn hàng đợi)."""
    if st.last_result:
        return dict(st.last_result, busy=True)
    return {"ok": True, "status": 0, "type": None, "message": "Đang khởi động bộ nhận diện...",
            "face_count": 0, "identified": False, "name": None, "started": st.started,
            "calibrating": False, "verify_progress": 0.0, "verify_count": 0,
            "lock_frames": LOCK_FRAMES, "busy": True}


@app.route("/debug/selftest")
def debug_selftest():
    """Bài test ghi dữ liệu dành riêng cho phát triển; mặc định bị vô hiệu hóa."""
    if not ENABLE_SELFTEST or request.remote_addr not in ("127.0.0.1", "::1"):
        abort(404)
    cands = db.get_all_candidates()
    if not cands:
        return "Chưa có thí sinh nào đăng ký. Hãy đăng ký 1 thí sinh rồi mở lại trang này.", 400
    sid = db.start_session(cands[0]["id"])
    folder = os.path.join(VIOLATIONS_DIR, f"session_{sid}")
    os.makedirs(folder, exist_ok=True)
    for severity, v_type in (("warning", "SELFTEST_WARNING"), ("violation", "SELFTEST_VIOLATION")):
        img = np.full((240, 320, 3), 70, np.uint8)
        cv2.putText(img, v_type, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        path = os.path.join(folder, f"{v_type}_{int(time.time() * 1000)}.jpg")
        with open(path, "wb") as f:
            f.write(jpeg_bytes(img))
        db.log_violation(sid, v_type, "Bản ghi thử nghiệm đường ghi dữ liệu (không phải vi phạm thật).", path, severity)
    db.end_session(sid, status="completed")
    return (f"OK: đã tạo phiên thử #{sid} với 1 cảnh báo + 1 vi phạm.\n"
            f"Mở dashboard (http://127.0.0.1:5000) và xem phiên #{sid}. Có hiện ra thì đường ghi dữ liệu bình thường; "
            f"không hiện thì dashboard đang đọc cơ sở dữ liệu khác (kiểm tra 2 chương trình có chạy cùng thư mục không)."), 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/api/exam/start", methods=["POST"])
def exam_start():
    expire_exam_sessions()
    candidates = db.get_all_candidates()
    if not candidates:
        return jsonify(ok=False, message="Chưa có thí sinh nào đăng ký khuôn mặt. Hãy đăng ký trước."), 400
    token = secrets.token_urlsafe(16)
    with exam_lock:
        exam_sessions[token] = ExamState(candidates)
    return jsonify(ok=True, token=token)


@app.route("/api/exam/frame", methods=["POST"])
def exam_frame():
    expire_exam_sessions()
    data = request.get_json(silent=True) or {}
    st = get_exam(data.get("token"))
    if st is None:
        return jsonify(ok=False, message="Phiên thi không tồn tại hoặc đã kết thúc."), 410
    img = decode_frame(data.get("image", ""))
    if img is None:
        return jsonify(ok=False, message="Ảnh không hợp lệ."), 400
    # Đang xử lý frame trước -> bỏ frame này, trả kết quả gần nhất thay vì xếp hàng chờ.
    if not st.lock.acquire(blocking=False):
        return jsonify(busy_result(st))
    try:
        if st.ended or st.closed:
            return jsonify(ok=False, message="Phiên thi đã kết thúc."), 410
        result = analyze_exam_frame(st, img)
        st.last_result = result
        return jsonify(result)
    finally:
        st.lock.release()


@app.route("/api/exam/begin", methods=["POST"])
def exam_begin():
    st = get_exam((request.get_json(silent=True) or {}).get("token"))
    if st is None or st.candidate is None:
        return jsonify(ok=False, message="Chưa xác thực khuôn mặt."), 400
    with st.lock:
        if not st.started:
            # Bắt buộc đã nhìn thẳng + ngồi yên ở bước chờ: mốc nhìn thẳng đo TRƯỚC khi thi, không đo lại khi đang làm bài.
            if st.pre_baseline is None:
                return jsonify(ok=False, retry=True,
                               message="Chưa xác định được tư thế nhìn thẳng. Hãy ngồi thẳng, nhìn thẳng vào màn hình, giữ yên rồi bấm lại."), 409
            st.started = True
            st.baseline = st.pre_baseline
            st.begin_time = time.time()
            st.mismatch_since = None
            st.yaw_hist.clear()
            st.pitch_hist.clear()
            st.roll_hist.clear()
            st.events.clear()
            st.session_id = db.start_session(st.candidate["id"])
            st.evidence_dir = os.path.join(VIOLATIONS_DIR, f"session_{st.session_id}")
            os.makedirs(st.evidence_dir, exist_ok=True)
    return jsonify(
        ok=True, name=st.candidate["full_name"], session_id=st.session_id,
        title=EXAM["title"], duration_minutes=EXAM["duration_minutes"],
        questions=[{"q": q["q"], "options": q["options"]} for q in EXAM["questions"]],
    )


@app.route("/api/exam/event", methods=["POST"])
def exam_event():
    data = request.get_json(silent=True) or {}
    st = get_exam(data.get("token"))
    v_type = data.get("type")
    if st is None or not st.started or v_type not in BROWSER_EVENTS:
        return jsonify(ok=False), 400
    with st.lock:
        log_violation(st, v_type, BROWSER_EVENTS[v_type])
    return jsonify(ok=True)


def clean_answers(raw):
    """Chỉ nhận đáp án hợp lệ: khóa là số thứ tự câu, giá trị là chỉ số đáp án hoặc None."""
    out = {}
    raw = raw if isinstance(raw, dict) else {}
    for i, q in enumerate(EXAM["questions"]):
        v = raw.get(str(i), raw.get(i))
        out[str(i)] = v if isinstance(v, int) and not isinstance(v, bool) and 0 <= v < len(q["options"]) else None
    return out


def exam_deadline(begin_time):
    return begin_time + EXAM["duration_minutes"] * 60 + SUBMIT_GRACE_SECONDS


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _draft_path(session_id):
    return os.path.join(ANSWERS_DIR, f"session_{session_id}.draft.json")


@app.route("/api/exam/save", methods=["POST"])
def exam_save():
    """Trình duyệt tự lưu đáp án định kỳ: mất kết nối/máy ngủ giữa chừng vẫn không mất bài."""
    data = request.get_json(silent=True) or {}
    token = data.get("token")
    answers = clean_answers(data.get("answers"))
    now = time.time()
    rec = None
    st = get_exam(token)
    if st is not None and st.started and st.session_id is not None and not st.ended:
        sid, name, begin = st.session_id, st.candidate["full_name"], st.begin_time or now
    else:
        rec = expired_sessions.get(token)
        if rec is None or rec["finished"]:
            return jsonify(ok=False, message="Phiên thi không còn hiệu lực."), 410
        sid, name, begin = rec["session_id"], rec["candidate"], rec["begin_time"]
    if now > exam_deadline(begin):
        # Hết giờ: không nhận thay đổi nữa (đáp án đã lưu trước hạn được giữ nguyên).
        return jsonify(ok=True, frozen=True)
    if st is not None:
        st.answers = answers
    elif rec is not None:
        rec["answers"] = answers
    try:
        _write_json_atomic(_draft_path(sid), {"session_id": sid, "candidate": name, "answers": answers,
                                              "saved_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    except OSError:
        app.logger.exception("Không lưu được bản nháp đáp án của phiên %s", sid)
        return jsonify(ok=False, message="Không lưu được bản nháp."), 500
    return jsonify(ok=True)


@app.route("/api/exam/finish", methods=["POST"])
def exam_finish():
    data = request.get_json(silent=True) or {}
    token = data.get("token")
    st = get_exam(token)
    rec = None
    if st is None:
        # Phiên giám sát đã bị dọn do mất kết nối: vẫn cho nộp bài đã làm.
        rec = expired_sessions.get(token)
        if rec is None:
            return jsonify(ok=False, message="Phiên thi không tồn tại hoặc đã hết hạn."), 410
        if rec["finished"]:
            return jsonify(ok=True, already_finished=True)
        sid, name, begin, saved = rec["session_id"], rec["candidate"], rec["begin_time"], rec["answers"]
        lock = threading.Lock()
    else:
        lock = st.lock
    with lock:
        if st is not None:
            # Cho phép retry nếu server đã lưu thành công nhưng phản hồi bị mất trên mạng.
            if st.ended:
                return jsonify(ok=True, already_finished=True)
            if not st.started or st.session_id is None:
                return jsonify(ok=False, message="Phiên thi chưa được bắt đầu hoặc không hợp lệ."), 400
            sid, name, begin, saved = st.session_id, st.candidate["full_name"], st.begin_time or time.time(), st.answers
        now = time.time()
        answers = clean_answers(data["answers"]) if "answers" in data else (saved or clean_answers({}))
        overtime = now - exam_deadline(begin)          # > 0: nộp sau hạn (đã trừ thời gian châm chước)
        late = overtime > 0
        answer_path = os.path.join(ANSWERS_DIR, f"session_{sid}.json")
        try:
            _write_json_atomic(answer_path, {
                "session_id": sid,
                "candidate": name,
                "answers": answers,
                "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "elapsed_seconds": int(now - begin),
                "allowed_seconds": EXAM["duration_minutes"] * 60,
                "late": late,
                "overtime_seconds": int(max(0.0, overtime)),
                "recovered_after_disconnect": rec is not None,
            })
            db.end_session(sid, status="completed")
            if st is not None:
                st.ended = True
                st.close()
            else:
                rec["finished"] = True
        except Exception:
            app.logger.exception("Không thể lưu/nộp bài cho session %s", sid)
            return jsonify(ok=False, message="Không thể lưu bài thi vào máy chủ. Dữ liệu chưa được xác nhận nộp thành công."), 500
    # Các việc phụ dưới đây không được làm hỏng việc nộp bài.
    if late:
        try:
            db.log_violation(sid, "LATE_SUBMIT",
                             f"Nộp bài trễ {int(overtime)} giây so với thời gian quy định ({EXAM['duration_minutes']} phút).",
                             None, "violation")
        except Exception:
            app.logger.exception("Không ghi được cảnh báo nộp trễ cho phiên %s", sid)
    try:
        os.remove(_draft_path(sid))
    except OSError:
        pass
    # Giữ token trong thời gian ngắn để request retry nhận được already_finished.
    return jsonify(ok=True, late=late)


@app.route("/api/exam/abort", methods=["POST"])
def exam_abort():
    """Trình duyệt gọi khi đóng/tải lại trang giữa chừng (navigator.sendBeacon)."""
    token = (request.get_json(silent=True) or {}).get("token")
    with exam_lock:
        st = exam_sessions.pop(token, None)
    if st is not None:
        with st.lock:
            if st.session_id is not None and not st.ended:
                st.ended = True
                db.end_session(st.session_id, status="aborted")
            st.close()
    return jsonify(ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web thí sinh: đăng ký khuôn mặt + thi có giám sát")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Mặc định 127.0.0.1 (chỉ máy này). Dùng 0.0.0.0 để các máy khác trong mạng truy cập.")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--https", action="store_true",
                        help="Bật HTTPS tự ký (cần: pip install cryptography). Bắt buộc nếu máy khác truy cập qua mạng LAN.")
    args = parser.parse_args()

    if face_utils.get_face_mesh() is None:
        print("[ERROR] Thiếu thư viện mediapipe. Hãy chạy: pip install -r requirements.txt")
        raise SystemExit(1)
    try:
        face_utils.init_face_recognition()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)

    db.init_db()
    for d in (DATASET_DIR, VIOLATIONS_DIR, ANSWERS_DIR):
        os.makedirs(d, exist_ok=True)

    print(f"[SYSTEM] Mã phiên bản giám sát: {CODE_VERSION}")
    print(f"[SYSTEM] Cơ sở dữ liệu đang dùng: {db.DB_PATH}")
    scheme = "https" if args.https else "http"
    print(f"[SYSTEM] Web thí sinh đang chạy tại: {scheme}://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False,
            ssl_context="adhoc" if args.https else None)