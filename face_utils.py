"""Shared face utilities for registration and online exam monitoring.
Identity recognition: InsightFace buffalo_l (SCRFD detector + ArcFace embedding).
Head pose/gaze: MediaPipe Face Mesh + OpenCV solvePnP. No anti-spoofing or phone detector.
"""
import os
import threading
import numpy as np
import cv2

solutions = None
try:
    import mediapipe as mp
    solutions = getattr(mp, "solutions", None)
except Exception:
    mp = None
if solutions is None:
    try:
        from mediapipe import solutions as solutions
    except Exception:
        try:
            from mediapipe.python import solutions as solutions
        except Exception:
            solutions = None
mp_face_mesh = solutions.face_mesh if solutions and hasattr(solutions, "face_mesh") else None

KEY_LANDMARKS = [33, 133, 362, 263, 4, 1, 61, 291, 199, 152, 10, 105, 334, 54, 284,
                 67, 109, 103, 297, 338, 296, 389, 161, 251, 21, 441, 172, 397]
MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.40"))
MATCH_MIN_MARGIN = float(os.getenv("FACE_MATCH_MIN_MARGIN", "0.05"))
MIN_DET_SCORE = float(os.getenv("FACE_MIN_DET_SCORE", "0.60"))
MIN_FACE_SIZE = int(os.getenv("FACE_MIN_SIZE", "80"))
MIN_BLUR_SCORE = float(os.getenv("FACE_MIN_BLUR_SCORE", "25"))
MIN_BRIGHTNESS = float(os.getenv("FACE_MIN_BRIGHTNESS", "35"))
MAX_BRIGHTNESS = float(os.getenv("FACE_MAX_BRIGHTNESS", "225"))

_RECOGNITION_APP = None
_RECOGNITION_LOCK = threading.Lock()
_RECOGNITION_INIT_LOCK = threading.Lock()


def init_face_recognition():
    global _RECOGNITION_APP
    if _RECOGNITION_APP is not None:
        return _RECOGNITION_APP
    with _RECOGNITION_INIT_LOCK:
        if _RECOGNITION_APP is not None:
            return _RECOGNITION_APP
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise RuntimeError("Thiếu InsightFace. Cài: python -m pip install insightface onnxruntime") from exc
        model_name = os.getenv("FACE_RECOGNITION_MODEL", "buffalo_l")
        det_size = int(os.getenv("FACE_DET_SIZE", "320"))
        try:
            # Chỉ nạp detection + recognition. Mặc định buffalo_l còn chạy thêm 2 model landmark
            # và model tuổi/giới tính trên mỗi frame mà dự án không dùng -> chậm gấp nhiều lần.
            import onnxruntime as ort
            try:
                ort.preload_dlls(directory="")   # nạp file CUDA/cuDNN từ các gói pip nvidia
            except Exception as exc:
                print(f"[SYSTEM] Không nạp được thư viện CUDA: {exc}")
            use_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
            use_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
            print(f"[SYSTEM] InsightFace chạy bằng: {providers[0]}")
            app = FaceAnalysis(name=model_name, allowed_modules=["detection", "recognition"],
                               providers=providers)
            app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))

        except Exception as exc:
            raise RuntimeError(f"Không khởi tạo được InsightFace '{model_name}'. Kiểm tra model pack/cache.") from exc
        try:
            # Chạy thử 1 lần để frame thật đầu tiên không bị khựng lúc nạp model.
            app.get(np.zeros((480, 640, 3), dtype=np.uint8))
        except Exception:
            pass
        _RECOGNITION_APP = app
    return _RECOGNITION_APP


def _valid_vector(value):
    try:
        arr = np.asarray(value, dtype=np.float32).flatten()
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return None
    return arr / norm


def _quality_for_face(image_bgr, bbox, det_score):
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "Không lấy được vùng khuôn mặt.", {"face_width": 0, "face_height": 0, "blur": 0.0, "brightness": 0.0}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    width, height = x2 - x1, y2 - y1
    metrics = {"face_width": width, "face_height": height, "blur": round(blur, 1), "brightness": round(brightness, 1), "det_score": round(float(det_score), 3)}
    if float(det_score) < MIN_DET_SCORE:
        return False, "Camera chưa nhận diện rõ khuôn mặt. Hãy nhìn thẳng và đủ sáng.", metrics
    if min(width, height) < MIN_FACE_SIZE:
        return False, "Khuôn mặt quá nhỏ. Hãy ngồi gần camera hơn.", metrics
    if blur < MIN_BLUR_SCORE:
        return False, "Ảnh khuôn mặt bị mờ. Hãy giữ đầu/camera ổn định.", metrics
    if not (MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS):
        return False, "Ánh sáng chưa phù hợp. Hãy tăng sáng hoặc tránh ngược sáng.", metrics
    return True, "OK", metrics


def extract_face_observations(image_bgr):
    """Return detected faces sorted by area: embedding, bbox, detector score and quality."""
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return []
    app = init_face_recognition()
    with _RECOGNITION_LOCK:
        detected = app.get(image_bgr)
    detected = sorted(detected, key=lambda f: max(0.0, float(f.bbox[2]-f.bbox[0])) * max(0.0, float(f.bbox[3]-f.bbox[1])), reverse=True)
    result = []
    for face in detected:
        emb = _valid_vector(getattr(face, "normed_embedding", None))
        if emb is None:
            continue
        bbox = np.asarray(face.bbox, dtype=np.float32).tolist()
        score = float(getattr(face, "det_score", 0.0))
        quality_ok, quality_message, metrics = _quality_for_face(image_bgr, bbox, score)
        result.append({"embedding": emb, "bbox": bbox, "det_score": score,
                       "quality_ok": quality_ok, "quality_message": quality_message, "quality": metrics})
    return result


def extract_face_embeddings(image_bgr):
    """Backward-compatible API: return normalized embedding vectors only."""
    return [x["embedding"] for x in extract_face_observations(image_bgr)]


def average_embeddings(embeddings):
    valid = [_valid_vector(item) for item in embeddings]
    valid = [v for v in valid if v is not None]
    if not valid or len({v.size for v in valid}) != 1:
        raise ValueError("Không có đủ embedding khuôn mặt hợp lệ để đăng ký.")
    return _valid_vector(np.mean(np.stack(valid), axis=0).astype(np.float32))


def get_face_mesh(max_num_faces=1, static_image_mode=False):
    if mp_face_mesh is None:
        return None
    return mp_face_mesh.FaceMesh(static_image_mode=static_image_mode, max_num_faces=max_num_faces,
                                 refine_landmarks=True, min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)


def landmarks_to_norm_array(face_landmarks):
    return np.array([[lm.x, lm.y] for lm in face_landmarks.landmark], dtype=np.float32)


def landmarks_to_pixel_array(face_landmarks, w_img, h_img):
    return np.array([[lm.x*w_img, lm.y*h_img] for lm in face_landmarks.landmark], dtype=np.float32)


def extract_stable_features(pts_norm):
    pts_norm = np.asarray(pts_norm, dtype=np.float32)
    if pts_norm.ndim != 2 or pts_norm.shape[0] <= max(KEY_LANDMARKS) or pts_norm.shape[1] < 2:
        raise ValueError("Landmark khuôn mặt không hợp lệ hoặc thiếu điểm.")
    key_pts = pts_norm[KEY_LANDMARKS, :2]
    nose_pt = pts_norm[4, :2]
    base_dist = float(np.linalg.norm(pts_norm[33, :2]-pts_norm[263, :2]))
    if not np.isfinite(base_dist) or base_dist < 1e-5:
        raise ValueError("Không đủ chất lượng landmark để xác thực.")
    return np.linalg.norm(key_pts-nose_pt, axis=1)/base_dist


# Model 3D theo hệ trục camera (x phải, y xuống, z ra xa camera), cùng thứ tự với
# _POSE_LANDMARK_IDS: mũi, cằm, khóe mắt (ảnh trái / ảnh phải), khóe miệng (ảnh trái / ảnh phải).
# Bản cũ dùng hệ y-lên / z-về-camera nên mặt nhìn thẳng cho pitch ~ -180 và quay đầu 25 độ cho
# yaw ~ 155; điều kiện abs(v) <= 90 bên dưới loại hết -> góc đầu gần như không bao giờ hợp lệ.
_MODEL_POINTS_3D = np.array([(0.,0.,0.), (0.,330.,65.), (-225.,-170.,135.),
                             (225.,-170.,135.), (-150.,150.,125.), (150.,150.,125.)], dtype=np.float64)
_POSE_LANDMARK_IDS = [4, 152, 33, 263, 61, 291]


def compute_head_pose_checked(mesh_coords_px, w_img, h_img):
    """Return (yaw, pitch, roll) in degrees, or None if landmarks/solvePnP are invalid.

    Landmark order and sign convention intentionally match the existing implementation.
    Do not silently convert a failed solvePnP result to (0, 0, 0): that would look like
    a perfectly neutral head pose and could pollute baseline calibration.
    """
    try:
        pts = np.asarray(mesh_coords_px, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] <= max(_POSE_LANDMARK_IDS) or pts.shape[1] < 2:
            return None
        image_points = pts[_POSE_LANDMARK_IDS, :2]
        if not np.all(np.isfinite(image_points)) or w_img <= 0 or h_img <= 0:
            return None
        # Degenerate landmarks (nearly zero eye distance) are not a usable pose.
        eye_span = float(np.linalg.norm(image_points[2] - image_points[3]))
        if not np.isfinite(eye_span) or eye_span < 8.0:
            return None
        focal_length = float(w_img)
        center = (w_img / 2.0, h_img / 2.0)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype=np.float64)
        success, rotation_vec, _ = cv2.solvePnP(
            _MODEL_POINTS_3D, image_points, camera_matrix, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success or rotation_vec is None or not np.all(np.isfinite(rotation_vec)):
            return None
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        angles, *_ = cv2.RQDecomp3x3(rotation_mat)
        # Quy ước đầu ra (khớp nhãn trong classify_pose và bước đăng ký, trên ảnh KHÔNG lật gương):
        #   yaw   > 0 : thí sinh quay sang TRÁI của chính họ   (mũi lệch về phía phải ảnh)
        #   pitch > 0 : thí sinh NGẨNG đầu                     (cúi đầu là pitch < 0)
        # solvePnP với model này cho dấu ngược lại cả hai, nên đảo dấu ở đây.
        pitch, yaw, roll = (-float(angles[0]), -float(angles[1]), float(angles[2]))
        result = (yaw, pitch, roll)
        if not all(np.isfinite(v) and abs(v) <= 90.0 for v in result):
            return None
        return result
    except (cv2.error, TypeError, ValueError, IndexError, OverflowError):
        return None


def compute_head_pose(mesh_coords_px, w_img, h_img):
    """Backward-compatible wrapper; invalid pose keeps the historic neutral fallback."""
    result = compute_head_pose_checked(mesh_coords_px, w_img, h_img)
    return result if result is not None else (0.0, 0.0, 0.0)


# ---- Góc quay trái/phải đo trực tiếp từ MŨI ----
# Mũi nhô ra phía trước hai khóe mắt, nên khi quay đầu thì mũi lệch ngang so với điểm giữa hai mắt.
# Độ lệch này (chia cho khoảng cách hai khóe mắt) tỉ lệ với tan(góc quay):  r = K * tan(yaw)
# K = NOSE_DEPTH_RATIO ~ 0.36 ở khoảng cách ngồi ~50cm (theo mô hình mặt 3D chung + phối cảnh).
# Chỉnh K nếu góc hiện trên màn hình thường lớn hơn/nhỏ hơn thực tế (K lớn -> góc ước lượng nhỏ đi).
NOSE_DEPTH_RATIO = float(os.getenv("NOSE_DEPTH_RATIO", "0.36"))
# Ngẩng/cúi làm độ nhô hiệu dụng của mũi thay đổi (ngẩng: ước lượng quay hơi lớn, cúi: hơi nhỏ), bù lại theo góc ngẩng/cúi.
NOSE_PITCH_COMP = float(os.getenv("NOSE_PITCH_COMP", "0.011"))


def compute_nose_yaw(mesh_coords_px, pitch_deg=0.0):
    """Góc quay trái/phải (độ) đo từ mũi, hoặc None nếu landmark không hợp lệ.
    Quy ước: > 0 là thí sinh quay sang TRÁI của họ (mũi lệch về phía PHẢI ảnh, ảnh không lật gương),
    < 0 là quay sang PHẢI. Đo theo đường nối hai khóe mắt nên không bị ảnh hưởng khi nghiêng đầu."""
    try:
        pts = np.asarray(mesh_coords_px, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] <= 263 or pts.shape[1] < 2:
            return None
        a, b, nose = pts[33, :2], pts[263, :2], pts[4, :2]    # khóe mắt (trái ảnh), khóe mắt (phải ảnh), mũi
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(nose))):
            return None
        eye_vec = b - a
        eye_dist = float(np.linalg.norm(eye_vec))
        if eye_dist < 8.0:
            return None
        r = float(np.dot(nose - (a + b) / 2.0, eye_vec / eye_dist)) / eye_dist
        yaw = float(np.degrees(np.arctan(r / NOSE_DEPTH_RATIO)))
        comp = min(1.25, max(0.75, 1.0 + NOSE_PITCH_COMP * float(pitch_deg or 0.0)))
        yaw /= comp
        return yaw if np.isfinite(yaw) and abs(yaw) <= 90.0 else None
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


RIGHT_EYE_CORNERS=(33,133); LEFT_EYE_CORNERS=(362,263)
RIGHT_EYE_LID=(159,145); LEFT_EYE_LID=(386,374)
RIGHT_IRIS_IDX=[468,469,470,471,472]; LEFT_IRIS_IDX=[473,474,475,476,477]


def compute_eye_gaze_ratio(mesh_coords_px):
    pts = np.asarray(mesh_coords_px)
    if pts.shape[0] < 478:
        return 0.5, 0.5
    def eye_ratio(corners, lids, iris_idx):
        c1,c2=pts[corners[0]],pts[corners[1]]
        upper,lower=pts[lids[0]],pts[lids[1]]
        iris=pts[iris_idx].mean(axis=0)
        xmin,xmax=min(c1[0],c2[0]),max(c1[0],c2[0])
        ymin,ymax=min(upper[1],lower[1]),max(upper[1],lower[1])
        return ((iris[0]-xmin)/(xmax-xmin+1e-6),(iris[1]-ymin)/(ymax-ymin+1e-6))
    h1,v1=eye_ratio(RIGHT_EYE_CORNERS,RIGHT_EYE_LID,RIGHT_IRIS_IDX)
    h2,v2=eye_ratio(LEFT_EYE_CORNERS,LEFT_EYE_LID,LEFT_IRIS_IDX)
    return float(np.clip((h1+h2)/2,0,1)),float(np.clip((v1+v2)/2,0,1))


def compute_turn_ratio(mesh_coords_px):
    nose=mesh_coords_px[4]; left_eye=mesh_coords_px[33]; right_eye=mesh_coords_px[263]; chin=mesh_coords_px[152]
    scale=np.linalg.norm(left_eye-right_eye)+1e-6
    left_dist=np.linalg.norm(nose-left_eye); right_dist=np.linalg.norm(nose-right_eye)
    yaw=((left_dist-right_dist)/scale)*100
    eye_y=(left_eye[1]+right_eye[1])/2.0
    pitch=((chin[1]-nose[1])-(nose[1]-eye_y))/scale*100
    return float(yaw),float(pitch)


def _best_similarity(query, main_ref, extra_refs=None):
    """Độ giống cao nhất giữa vector query và các vector tham chiếu cùng kích thước. None nếu không có vector hợp lệ."""
    best=None
    for raw in [main_ref]+list(extra_refs or []):
        ref=_valid_vector(raw)
        if ref is None or ref.size != query.size:
            continue
        sim=float(np.dot(query,ref))
        if np.isfinite(sim) and (best is None or sim>best):
            best=sim
    return best


def match_face(features, candidates, verbose=False, use_extra=False):
    """Tìm thí sinh khớp nhất.
    use_extra=False (mặc định, dùng khi XÁC THỰC vào thi): chỉ so với vector nhìn thẳng - chặt nhất.
    use_extra=True: so thêm với các vector tư thế khác (quay trái/phải) và lấy độ giống cao nhất."""
    query=_valid_vector(features)
    if query is None or not candidates:
        return None,None
    scored=[]
    for candidate in candidates:
        extras=candidate.get("extra_features") if use_extra else None
        sim=_best_similarity(query,candidate.get("features"),extras)
        if sim is not None:
            scored.append((sim,candidate))
            if verbose: print(f"[MATCH] {candidate.get('full_name','unknown')} cosine={sim:.3f}")
    if not scored: return None,None
    scored.sort(key=lambda x:x[0],reverse=True)
    best,best_candidate=scored[0]
    if best < MATCH_THRESHOLD: return None,best
    if len(scored)>1 and best-scored[1][0] < MATCH_MIN_MARGIN:
        if verbose: print(f"[MATCH] Ambiguous best={best:.3f}, second={scored[1][0]:.3f}")
        return None,best
    return best_candidate,best


def match_specific(features, reference_features, extra_features=None):
    """So với đúng một thí sinh. Có extra_features (vector các tư thế khác) thì lấy độ giống cao nhất trong tất cả."""
    q=_valid_vector(features)
    if q is None: return False,float("-inf")
    sim=_best_similarity(q,reference_features,extra_features)
    if sim is None: return False,float("-inf")
    return bool(sim>=MATCH_THRESHOLD),sim