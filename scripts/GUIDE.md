# Vai trò các script

| File | Chức năng |
|---|---|
| `download_data.py` | Tải riêng dữ liệu Oddball và lưu thông tin nguồn, dung lượng, hash trong manifest. |
| `build_data_report.py` | Tạo PDF kiểm kê dữ liệu đã tải, giải thích các file và mã sự kiện. |
| `preprocess.py` | Lọc EEG, fit ICA, chọn kênh, cắt lượt, chỉnh nền và loại lượt lỗi. Hỗ trợ tạo mẫu từ EEG đã làm sạch. |
| `apply_saved_ica.py` | Dùng ICA đã fit để loại thành phần nghi nhiễu mắt, dựng lại EEG và xuất hình trước/sau. |
| `create_splits.py` | Kiểm tra dữ liệu cuối và chia train/test theo người, lưu vào splits.json. |
| `extract_features.py` | Tạo 320 đặc trưng mỗi mẫu bằng trung bình 20 cửa sổ 50 ms trên 16 kênh. |
| `train_ml.py` | Huấn luyện baseline, Shrinkage LDA và SVM RBF; lưu macro-F1 và confusion matrix. |
| `train_dl.py` | Huấn luyện Compact EEG CNN và EEG-Inception; lưu checkpoint, lịch sử epoch và kết quả test. |
| `visualize.py` | Vẽ kết quả mô hình, đường học, ERP Pz, phân bố đặc trưng và độ nhạy khi che tín hiệu. |
| `analyze_responses.py` | Ghép mốc kích thích với lần bấm nút, thống kê thời gian phản ứng trên nhóm train để khảo sát thông tin muộn. |
| `cleanup_intermediate.py` | Kiểm tra và dọn các file trung gian đã được thay thế hoặc xác nhận trùng. |

Thứ tự và lệnh chạy được trình bày trong [README chính](../README.md).
