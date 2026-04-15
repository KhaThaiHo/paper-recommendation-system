# Token Merging (ToMe) - Phân Loại Tạp Chí Học Thuật

## Cài Đặt

### 1. Tạo môi trường ảo
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# hoặc: venv\Scripts\activate
```

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

Thư viện cần thiết:
- `torch >= 2.0.0`
- `transformers >= 4.30.0`
- `pandas >= 1.5.0`
- `numpy >= 1.23.0`
- `scikit-learn >= 1.2.0`

## Chạy Chương Trình

### Yêu cầu
- Dữ liệu CSV có cột: `Title`, `Abstract`, `Keywords`, `Label`
- GPU CUDA (khuyến nghị) hoặc CPU

### Chạy ToMe_1.py
```bash
python ToMe_1.py
```

**Cây người dùng phải:**
1. Chuẩn bị file dữ liệu (hoặc sửa đường dẫn trong dòng cuối cùng của script)
2. Chạy script sẽ tự động:
   - Xử lý dữ liệu
   - Huấn luyện 2 mô hình (Baseline + ToMe ON)
   - So sánh hiệu suất (độ chính xác, tốc độ, bộ nhớ)

### Output
- Kết quả huấn luyện từng epoch
- Bảng so sánh chi tiết (Top-1, Top-3, Top-5, Top-10)
- Tổng thời gian thực hiện
