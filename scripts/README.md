# Các script của dự án

## Tải dữ liệu và tạo báo cáo trên server

Chạy từ thư mục gốc của repo với Python 3.9 trở lên.

```bash
python scripts/download_data.py --dry-run
python scripts/download_data.py --subjects sub-01
python scripts/download_data.py
python -m pip install reportlab
python scripts/build_data_report.py --manifest data/raw/ds004621/download_manifest.json
```

- Lệnh đầu đọc danh sách đối tượng trên OpenNeuro và in dung lượng. Chưa tải bản ghi.
- Lệnh thứ hai tải thử một người. Lần tải đầy đủ sẽ dùng lại các file đã tải nếu ETag, dung lượng và SHA-256 khớp.
- Dữ liệu nằm trong `data/raw/ds004621/`. Mỗi người có năm file Oddball và cả bộ có ba JSON dùng chung. Nếu thiếu file dự kiến, script báo đường dẫn thiếu.
- `download_manifest.json` ghi nguồn, file, dung lượng, SHA-256 và thống kê mã sự kiện, số kênh EEG trước tiền xử lý. Manifest tải thử là `download_manifest_sub-01.json`. Báo cáo có thể dùng manifest này và sẽ ghi rõ phạm vi một người.
- PDF tiếng Anh được tạo tại `docs/report/data_acquisition.pdf`. Có thể chạy script tạo PDF khi chưa có manifest. Bản đó để phần kiểm kê ở trạng thái chờ.
- Dữ liệu lấy từ bản sao công khai S3 của OpenNeuro. Manifest lưu ETag từng file để đối chiếu nguồn đã dùng.

## Các script tiếp theo

| Script | Vai trò |
|---|---|
| `preprocess.py` | Đọc, lọc, cắt lượt, chỉnh nền và kiểm tra chất lượng. |
| `extract_features.py` | Tạo đặc trưng 50 ms cho ML. |
| `train_ml.py` | Chạy baseline, Shrinkage LDA và SVM RBF. |
| `train_dl.py` | Chạy Compact EEG CNN và EEG-Inception. |
| `visualize.py` | Vẽ các hình và đo ảnh hưởng khi che từng khoảng thời gian. |
| `run_pipeline.py` | Điều phối các bước với cùng config và cách chia người. |

Trình tự thực hiện được ghi trong [kế hoạch](../docs/eeg-oddball-ke-hoach-trien-khai.md).
