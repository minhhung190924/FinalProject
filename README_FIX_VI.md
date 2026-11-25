# Bản cập nhật AI Proctoring

## Phạm vi cập nhật

- Nhận diện danh tính bằng InsightFace `buffalo_l` (SCRFD detector + embedding nhận diện ArcFace).
- MediaPipe Face Mesh tiếp tục dùng cho số lượng khuôn mặt, landmark, head pose và gaze.
- Thêm kiểm tra chất lượng ảnh khuôn mặt (độ rõ, kích thước, độ sáng, detector confidence) trước khi lấy embedding để tránh xác thực bằng ảnh quá mờ/tối/nhỏ.
- Đăng ký khuôn mặt chỉ có 1 bước `LOOK_STRAIGHT`: thí sinh nhìn thẳng vào camera. Hệ thống lấy trung bình nhiều embedding từ các khung hình đạt chất lượng; không yêu cầu quay trái/phải khi đăng ký.
- Xác thực danh tính trong 3 khung hình khớp liên tiếp (`LOCK_FRAMES = 3`); khi ảnh mờ/chất lượng thấp thì không vội kết luận đổi người.
- Lưu cả cảnh báo và vi phạm vào database, phân biệt `warning` và `violation`; dashboard hiển thị hai loại riêng.
- Database tự bổ sung cột mới theo kiểu migration cộng thêm, không xóa dữ liệu cũ.
- `migrate_face_embeddings.py` tạo backup trước khi ghi và hỗ trợ `--dry-run`.

Không thêm anti-spoofing và không thêm nhận diện điện thoại.

## Chế độ debug và self-test

- Mặc định `DEBUG_DETECTION=0`: không in log góc đầu ra terminal và không ghi `detect_debug.log`.
- Chỉ bật log khi cần kiểm tra AI bằng cách đặt biến môi trường `DEBUG_DETECTION=1` trước khi chạy `student_web.py`. Tắt lại bằng `DEBUG_DETECTION=0` hoặc xóa biến môi trường.
- Endpoint `/debug/selftest` mặc định bị khóa (trả 404), tránh tạo phiên/cảnh báo/vi phạm giả trong dữ liệu thật.
- Chỉ khi đang phát triển và muốn kiểm tra đường ghi dữ liệu dashboard, có thể bật `ENABLE_SELFTEST=1`. Endpoint vẫn chỉ cho phép truy cập từ `127.0.0.1` hoặc `::1`. Bài test này tạo dữ liệu giả trong database và thư mục `violation_evidence`; sau khi kiểm tra, nên tắt `ENABLE_SELFTEST=0` và xóa phiên test nếu không cần giữ.

Trên Windows CMD, bật tạm trong cửa sổ terminal hiện tại:

```bat
set DEBUG_DETECTION=1
set ENABLE_SELFTEST=1
python student_web.py
```

Chạy bình thường, không debug:

```bat
set DEBUG_DETECTION=0
set ENABLE_SELFTEST=0
python student_web.py
```

## Cài thư viện

Khuyến nghị dùng Python 3.10 hoặc 3.11 trong virtual environment:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-proctoring.txt
```

Lần đầu chạy InsightFace có thể cần tải model pack `buffalo_l` về cache. Máy hiện tại của bạn cần cài các thư viện trong file requirements; môi trường kiểm tra của gói này chưa có Flask, MediaPipe và InsightFace nên chưa thể chạy thử camera end-to-end.

## Cách áp dụng

1. Sao lưu thư mục dự án hiện tại, đặc biệt `exam_system.db`, `dataset`, `templates`, `static`, `violation_evidence`, `exam_answers`.
2. Giải nén gói cập nhật và chép các file `.py` trong gói vào đúng thư mục gốc dự án (đặc biệt phải có `exam_config.py` nằm cùng cấp với `student_web.py` và `teacher_web.py`). Giữ nguyên `templates/` và `static/` hiện có.
3. Kiểm tra embedding cũ:

```bash
python migrate_face_embeddings.py --dry-run
```

4. Nếu kết quả kiểm tra hợp lệ, chạy chuyển đổi:

```bash
python migrate_face_embeddings.py
```

Lệnh trên tạo một bản sao `exam_system_before_embedding_migration_*.db` trước khi cập nhật. Mặc định chỉ dùng `face_1.jpg` (ảnh nhìn thẳng); không lấy trung bình ảnh quay trái/phải vì góc quay lớn có thể làm embedding kém ổn định. Trường hợp không có ảnh `face_1.jpg` hoặc ảnh không đạt chất lượng thì cần đăng ký lại thí sinh đó.

5. Chạy ứng dụng thí sinh và dashboard ở hai terminal:

```bash
python student_web.py
python teacher_web.py
```

- Student: `http://127.0.0.1:5001`
- Dashboard: `http://127.0.0.1:5000`

## Tinh chỉnh ngưỡng nhận diện

Mặc định:

- `FACE_MATCH_THRESHOLD=0.40`
- `FACE_MATCH_MIN_MARGIN=0.05`
- `FACE_MIN_DET_SCORE=0.60`
- `FACE_MIN_SIZE=80`
- `FACE_MIN_BLUR_SCORE=25`

Có thể đặt biến môi trường trước khi chạy. Không nên tự ý tăng/giảm ngưỡng chỉ để nhận diện dễ hơn; hãy kiểm thử nhiều lần với đúng người và người khác trên camera thực tế, rồi chọn ngưỡng cân bằng giữa nhận nhầm và từ chối đúng người.

## Lưu ý đánh giá

Các ngưỡng góc đầu và gaze hiện tại vẫn là ngưỡng khởi đầu từ dự án. Cần kiểm tra thực tế hướng quay trái/phải và ngẩng/cúi trên webcam của bạn; dấu yaw/pitch có thể thay đổi theo góc camera và cách đặt máy. Hệ thống chỉ đưa ra dấu hiệu/cảnh báo hành vi, không thể khẳng định chắc chắn một người gian lận chỉ từ một lần nhìn lệch.
