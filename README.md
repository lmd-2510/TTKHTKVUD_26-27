# EEG Oddball — TTKHTKVUD_26-27

Mini-project phân loại từng lượt EEG Oddball thành `standard` hoặc `target` trên bộ Nencki–Symfonia `ds004621`.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `configs/` | Cấu hình thí nghiệm và danh sách người của từng tập. |
| `data/raw/` | Bản ghi BrainVision gốc. |
| `data/processed/` | Các lượt EEG đã xử lý, nhãn và mã người. |
| `scripts/` | Các bước của pipeline. |
| `results/quality/` | Kiểm tra kênh và số lượt theo người. |
| `results/metrics/` | Ma trận nhầm lẫn và macro-F1. |
| `results/checkpoints/` | Trọng số của các mạng DL. |
| `results/figures/` | Hình Pz, phân bố đặc trưng, kết quả và phép che tín hiệu. |
| `docs/` | Kế hoạch, báo cáo và thuyết trình. |

Xem [kế hoạch triển khai](docs/eeg-oddball-ke-hoach-trien-khai.md) và [hướng dẫn tải dữ liệu và tạo báo cáo trên server](scripts/README.md). Thông số hiện tại là cấu hình khởi đầu. Chốt danh sách người và việc dùng ICA sau khi kiểm tra chất lượng dữ liệu thật.
