"""Cấu hình đề thi dùng chung cho ứng dụng thí sinh và dashboard giáo viên.

correct là chỉ số đáp án đúng, bắt đầu từ 0. Ứng dụng thí sinh không gửi trường
correct xuống trình duyệt; dashboard dùng trường này để chấm và xem lại bài.
"""

EXAM = {
    "title": "Bài thi thử",
    "duration_minutes": 10,
    "questions": [
        {
            "q": "Thủ đô của Việt Nam là thành phố nào?",
            "options": ["Hà Nội", "Đà Nẵng", "Huế", "TP. Hồ Chí Minh"],
            "correct": 0,
        },
        {
            "q": "Kết quả của 2 mũ 5 (2^5) là bao nhiêu?",
            "options": ["10", "16", "25", "32"],
            "correct": 3,
        },
        {
            "q": "Trong Python, kiểu dữ liệu nào KHÔNG thể thay đổi sau khi tạo (immutable)?",
            "options": ["list", "dict", "tuple", "set"],
            "correct": 2,
        },
        {
            "q": "Câu lệnh SQL nào dùng để lấy dữ liệu từ bảng?",
            "options": ["INSERT", "SELECT", "DELETE", "UPDATE"],
            "correct": 1,
        },
        {
            "q": "HTTP status code 404 có nghĩa là gì?",
            "options": ["Lỗi server", "Không có quyền truy cập", "Không tìm thấy trang", "Thành công"],
            "correct": 2,
        },
    ],
}
