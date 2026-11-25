"""
teacher_web.py
Dashboard xem dữ liệu qua trình duyệt web (thay vì gõ lệnh terminal).
Chạy: python teacher_web.py
Sau đó mở trình duyệt vào: http://127.0.0.1:5000

Đây là bản xem-dữ-liệu cục bộ, chạy ngay trên máy bạn, chưa phải server thật
để giáo viên truy cập từ xa — nhưng logic (Flask + đọc từ db.py) chính là nền
tảng bạn sẽ dùng để xây dashboard giáo viên thật sau này.
"""

import json
import os
import secrets
import shutil
import time
from datetime import datetime, timedelta

from flask import Flask, render_template, send_file, abort, redirect, url_for, request, session
from werkzeug.security import generate_password_hash, check_password_hash

import db
from exam_config import EXAM

app = Flask(__name__)


def format_datetime(value):
    """Hiển thị ngày giờ gọn: YYYY-MM-DD HH:MM:SS, bỏ phần múi giờ."""
    if not value:
        return "-"
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        # Dữ liệu cũ không đúng ISO: bỏ phần timezone nếu có và thay T bằng dấu cách.
        text = text.replace("T", " ")
        if len(text) >= 6 and text[-6] in ("+", "-") and text[-3] == ":":
            text = text[:-6]
        return text


app.add_template_filter(format_datetime, "datetime_display")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
VIOLATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "violation_evidence")
AUDIT_LOG_FILE = os.path.join(BASE_DIR, "viewer_audit_log.jsonl")
ANSWERS_DIR = os.path.join(BASE_DIR, "exam_answers")

def load_exam_review(session_id):
    """Đọc bài đã nộp; nếu phiên bị bỏ giữa chừng thì đọc bản nháp tự lưu."""
    final_path = os.path.join(ANSWERS_DIR, f"session_{session_id}.json")
    draft_path = os.path.join(ANSWERS_DIR, f"session_{session_id}.draft.json")
    path = final_path if os.path.isfile(final_path) else draft_path if os.path.isfile(draft_path) else None
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    raw_answers = payload.get("answers", {})
    if not isinstance(raw_answers, dict):
        raw_answers = {}
    questions = []
    correct_count = 0
    answered_count = 0
    for i, source in enumerate(EXAM["questions"]):
        raw = raw_answers.get(str(i), raw_answers.get(i))
        selected = raw if isinstance(raw, int) and not isinstance(raw, bool) and 0 <= raw < len(source["options"]) else None
        is_correct = selected == source["correct"]
        if selected is not None:
            answered_count += 1
        if is_correct:
            correct_count += 1
        questions.append({
            "number": i + 1,
            "question": source["q"],
            "options": source["options"],
            "selected": selected,
            "correct": source["correct"],
            "is_correct": is_correct,
        })
    total = len(questions)
    return {
        "questions": questions,
        "correct_count": correct_count,
        "answered_count": answered_count,
        "total": total,
        "score_percent": round(correct_count * 100 / total) if total else 0,
        "submitted": path == final_path,
        "submitted_at": payload.get("submitted_at") or payload.get("saved_at"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "late": bool(payload.get("late", False)),
        "recovered_after_disconnect": bool(payload.get("recovered_after_disconnect", False)),
    }

# =====================================================================
# Tài khoản đăng nhập giáo viên
# Lần đầu chạy phiên bản này: tạo tên đăng nhập và mật khẩu trên trang /setup.
# Thông tin được lưu trong viewer_auth.json; mật khẩu chỉ lưu dưới dạng mã băm.
# =====================================================================
AUTH_FILE = os.path.join(BASE_DIR, "viewer_auth.json")
LOGIN_MAX_FAILS = 5          # sai quá số lần này thì khóa tạm
LOGIN_LOCK_SECONDS = 60
_login_fails = {}            # ip -> (số lần sai, thời điểm hết khóa)


def save_auth(data):
    """Lưu cấu hình xác thực vào viewer_auth.json."""
    temp_file = AUTH_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_file, AUTH_FILE)


def load_auth():
    data = {}
    if os.path.exists(AUTH_FILE):
        try:
            with open(AUTH_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}

    if not data.get("secret_key"):
        data["secret_key"] = secrets.token_hex(32)
        save_auth(data)

    return data


AUTH = load_auth()
app.secret_key = AUTH["secret_key"]
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  PERMANENT_SESSION_LIFETIME=timedelta(hours=8))


def account_is_configured():
    return bool(AUTH.get("username") and AUTH.get("password_hash"))


def check_credentials(username, password):
    saved_username = AUTH.get("username", "")
    username_ok = secrets.compare_digest(
        (username or "").strip().encode("utf-8"),
        saved_username.encode("utf-8")
    )
    password_hash = AUTH.get("password_hash", "")
    password_ok = check_password_hash(password_hash, password or "") if password_hash else False
    return username_ok and password_ok


def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


@app.context_processor
def inject_csrf():
    return {"csrf": csrf_token()}


@app.before_request
def require_login():
    if request.endpoint in ("login", "setup", "static"):
        return None
    if not session.get("auth"):
        if not account_is_configured():
            return redirect(url_for("setup"))
        return redirect(url_for("login", next=request.full_path.rstrip("?")))
    if request.method == "POST":
        sent = request.form.get("csrf", "")
        if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
            abort(400)
    return None


def write_audit_log(action, target_type, target_id, details):
    """Ghi nhật ký thao tác theo dạng JSON Lines; mỗi thao tác là một dòng riêng."""
    record = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "actor": session.get("teacher_username") or "unknown",
        "action": action,
        "target_type": target_type,
        "target_id": str(target_id),
        "details": details,
    }
    # Mở ở chế độ append để không ghi đè các nhật ký trước đó.
    with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_audit_logs(limit=500):
    """Đọc nhật ký mới nhất; bỏ qua dòng lỗi nếu file bị chỉnh sửa thủ công."""
    if not os.path.isfile(AUDIT_LOG_FILE):
        return []
    records = []
    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
            except (ValueError, TypeError):
                continue
    return list(reversed(records[-limit:]))


def remove_ip_from_existing_audit_logs():
    """Xóa trường IP khỏi các bản ghi nhật ký cũ, giữ nguyên các thông tin khác."""
    if not os.path.isfile(AUDIT_LOG_FILE):
        return

    cleaned_lines = []
    changed = False
    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    if "ip" in item:
                        item.pop("ip", None)
                        changed = True
                    cleaned_lines.append(json.dumps(item, ensure_ascii=False) + "\n")
                else:
                    cleaned_lines.append(line)
            except (ValueError, TypeError):
                # Giữ nguyên dòng không đọc được để tránh làm mất dữ liệu nhật ký.
                cleaned_lines.append(line)

    if changed:
        temp_file = AUDIT_LOG_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.writelines(cleaned_lines)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, AUDIT_LOG_FILE)


# Làm sạch trường IP trong nhật ký cũ ngay khi khởi động dashboard.
remove_ip_from_existing_audit_logs()


# Phiên đang thi thật thì không cho xóa (xóa giữa chừng làm hỏng việc ghi vi phạm của thí sinh đó).
# Phiên bị treo ở trạng thái 'active' quá chừng này giờ (máy chủ thi từng sập...) thì coi như đã bỏ, được phép xóa.
LIVE_SESSION_MAX_HOURS = 12


def session_is_live(s):
    if (s.get("status") or "active") in ("completed", "aborted"):
        return False
    try:
        started = datetime.fromisoformat(s["started_at"])
    except (KeyError, TypeError, ValueError):
        return False
    return (datetime.now() - started) < timedelta(hours=LIVE_SESSION_MAX_HOURS)


def safe_next(target):
    return target if target and target.startswith("/") and not target.startswith("//") else url_for("home")


@app.route("/setup", methods=["GET", "POST"])
def setup():
    """Cho phép tạo tài khoản giáo viên trong lần chạy đầu tiên."""
    global AUTH

    if account_is_configured():
        return redirect(url_for("login"))

    error = None
    if request.method == "POST":
        sent = request.form.get("csrf", "")
        if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
            abort(400)

        username = (request.form.get("username") or "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(username) < 3 or len(username) > 50:
            error = "Tên đăng nhập phải có từ 3 đến 50 ký tự."
        elif len(password) < 8:
            error = "Mật khẩu phải có ít nhất 8 ký tự."
        elif password != confirm_password:
            error = "Mật khẩu xác nhận không khớp."
        else:
            AUTH["username"] = username
            AUTH["password_hash"] = generate_password_hash(password)
            save_auth(AUTH)
            session.clear()
            csrf_token()
            return redirect(url_for("login", created="1"))

    return render_template("teacher/setup.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if not account_is_configured():
        return redirect(url_for("setup"))

    error = None
    ip = request.remote_addr or "?"
    now = time.time()
    fails, locked_until = _login_fails.get(ip, (0, 0))
    if request.method == "POST":
        sent = request.form.get("csrf", "")
        if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
            abort(400)
        if now < locked_until:
            error = f"Sai quá nhiều lần. Thử lại sau {int(locked_until - now) + 1} giây."
        elif check_credentials(
            request.form.get("username", ""),
            request.form.get("password", "")
        ):
            _login_fails.pop(ip, None)
            session.clear()
            session["auth"] = True
            session["teacher_username"] = AUTH["username"]
            session.permanent = True
            csrf_token()
            return redirect(safe_next(request.args.get("next")))
        else:
            fails += 1
            _login_fails[ip] = (fails, now + LOGIN_LOCK_SECONDS if fails >= LOGIN_MAX_FAILS else 0)
            if fails >= LOGIN_MAX_FAILS:
                _login_fails[ip] = (0, now + LOGIN_LOCK_SECONDS)
                error = f"Sai quá nhiều lần. Thử lại sau {LOGIN_LOCK_SECONDS} giây."
            else:
                error = "Tên đăng nhập hoặc mật khẩu không đúng."

    return render_template(
        "teacher/login.html",
        error=error,
        created=request.args.get("created") == "1"
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def _session_matches(s, f):
    if f["q"] and f["q"].lower() not in (s.get("full_name") or "").lower():
        return False
    status = s.get("status") or ""
    if f["status"]:
        if f["status"] == "active":
            if status in ("completed", "aborted"):
                return False
        elif status != f["status"]:
            return False
    day = (s.get("started_at") or "")[:10]
    if f["date_from"] and day < f["date_from"]:
        return False
    if f["date_to"] and day > f["date_to"]:
        return False
    if f["only_violations"] and s["n_violations"] == 0:
        return False
    return True


@app.route("/")
def home():
    f = {
        "q": (request.args.get("q") or "").strip()[:50],
        "status": request.args.get("status") or "",
        "date_from": (request.args.get("date_from") or "")[:10],
        "date_to": (request.args.get("date_to") or "")[:10],
        "only_violations": request.args.get("only_violations") == "1",
    }
    candidates = db.get_all_candidates()
    total_candidates = len(candidates)
    all_sessions = db.get_all_sessions()
    live_candidate_ids = {s["candidate_id"] for s in all_sessions if session_is_live(s)}
    for c in candidates:
        c["has_live"] = c["id"] in live_candidate_ids
    if f["q"]:
        candidates = [c for c in candidates if f["q"].lower() in c["full_name"].lower()]
    for c in candidates:
        folder = os.path.join(DATASET_DIR, c["full_name"])
        c["n_photos"] = len([f for f in os.listdir(folder) if f.endswith(".jpg")]) if os.path.isdir(folder) else 0
        c["registered_at"] = db.get_candidate_by_name(c["full_name"])["registered_at"]

    sessions = all_sessions
    for s in sessions:
        s["live"] = session_is_live(s)
        s["n_violations"] = int(s.get("n_violations") or 0)
        s["n_warnings"] = int(s.get("n_warnings") or 0)

    total_sessions = len(sessions)
    sessions = [s for s in sessions if _session_matches(s, f)]

    message = request.args.get("message")
    error = request.args.get("error")
    return render_template("teacher/home.html", candidates=candidates, sessions=sessions, message=message, error=error,
                                  f=f, total_candidates=total_candidates, total_sessions=total_sessions)


@app.route("/audit")
def audit_log_page():
    return render_template("teacher/audit.html", logs=read_audit_logs())


@app.route("/candidate/<int:candidate_id>/delete", methods=["POST"])
def delete_candidate_route(candidate_id):
    candidate = None
    for c in db.get_all_candidates():
        if c["id"] == candidate_id:
            candidate = c
            break
    if candidate is None:
        abort(404)

    if any(session_is_live(s) for s in db.get_all_sessions() if s["candidate_id"] == candidate_id):
        return redirect(url_for("home", error=f"Không thể xóa '{candidate['full_name']}': thí sinh này đang có phiên thi. Hãy đợi thi xong rồi xóa."))

    session_ids = db.get_session_ids_by_candidate(candidate_id)
    db.delete_candidate(candidate_id)

    folder = os.path.join(DATASET_DIR, candidate["full_name"])
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    for sid in session_ids:
        evidence_folder = os.path.join(VIOLATIONS_DIR, f"session_{sid}")
        if os.path.isdir(evidence_folder):
            shutil.rmtree(evidence_folder)

    write_audit_log(
        action="XÓA THÍ SINH",
        target_type="Thí sinh",
        target_id=candidate_id,
        details=f"Đã xóa thí sinh '{candidate['full_name']}' (ID {candidate_id}) và {len(session_ids)} phiên thi liên quan."
    )

    return redirect(url_for("home", message=f"Đã xóa thí sinh '{candidate['full_name']}' (ID {candidate_id})."))


@app.route("/session/<int:session_id>/delete", methods=["POST"])
def delete_session_route(session_id):
    summary = db.get_session_summary(session_id)
    if not summary:
        abort(404)

    if session_is_live(summary):
        return redirect(url_for("home", error=f"Không thể xóa phiên #{session_id}: thí sinh đang thi. Hãy đợi phiên kết thúc rồi xóa."))

    db.delete_session(session_id)

    evidence_folder = os.path.join(VIOLATIONS_DIR, f"session_{session_id}")
    if os.path.isdir(evidence_folder):
        shutil.rmtree(evidence_folder)

    write_audit_log(
        action="XÓA PHIÊN THI",
        target_type="Phiên thi",
        target_id=session_id,
        details=f"Đã xóa phiên thi #{session_id} của thí sinh '{summary.get('full_name', 'Không rõ')}'."
    )

    return redirect(url_for("home", message=f"Đã xóa session #{session_id}."))


@app.route("/candidate/<int:candidate_id>")
def candidate_detail(candidate_id):
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        abort(404)
    candidate = dict(row)
    folder = os.path.join(DATASET_DIR, candidate["full_name"])
    photos = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")]) if os.path.isdir(folder) else []
    return render_template("teacher/candidate.html", candidate=candidate, photos=photos)


@app.route("/session/<int:session_id>")
def session_detail(session_id):
    summary = db.get_session_summary(session_id)
    if not summary:
        abort(404)
    violations = [dict(v) for v in db.get_violations_by_session(session_id)]
    evidence_list = []
    for v in violations:
        if v.get("evidence_path"):
            v["ev_idx"] = len(evidence_list)
            evidence_list.append({"id": v["id"], "time": (v.get("created_at") or "")[:19],
                                  "severity": v.get("severity") or "violation",
                                  "type": v.get("violation_type") or "", "message": v.get("message") or ""})
    exam_review = load_exam_review(session_id)
    return render_template("teacher/session.html", session=summary, violations=violations,
                           evidence=evidence_list, exam_review=exam_review)


@app.route("/photo/<candidate_name>/<int:idx>")
def photo(candidate_name, idx):
    # Candidate folders are generated by clean_name() at registration; reject traversal anyway.
    if not candidate_name or not all(ch.isalnum() or ch in "_-" for ch in candidate_name):
        abort(404)
    folder = os.path.realpath(os.path.join(DATASET_DIR, candidate_name))
    if os.path.commonpath([os.path.realpath(DATASET_DIR), folder]) != os.path.realpath(DATASET_DIR) or not os.path.isdir(folder):
        abort(404)
    photos = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
    if idx >= len(photos):
        abort(404)
    return send_file(os.path.join(folder, photos[idx]))


@app.route("/evidence/<int:violation_id>")
def evidence(violation_id):
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("SELECT evidence_path FROM violations WHERE id = ?", (violation_id,))
    row = cur.fetchone()
    conn.close()
    if not row or not row["evidence_path"]:
        abort(404)
    path = os.path.realpath(row["evidence_path"])
    root = os.path.realpath(VIOLATIONS_DIR)
    if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
        abort(404)
    return send_file(path)


if __name__ == "__main__":
    db.init_db()
    print("[SYSTEM] Dashboard đang chạy tại: http://127.0.0.1:5000")
    app.run(debug=False, port=5000)