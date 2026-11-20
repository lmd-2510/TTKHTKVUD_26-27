# Phân loại EEG theo từng lượt trong tác vụ Oddball

Dự án dùng EEG của bộ Nencki–Symfonia (OpenNeuro `ds004621`) để dự đoán một lượt thuộc **standard** hay **target**. Nhóm chỉ sử dụng tác vụ Oddball. Mỗi kích thích standard/target tạo một mẫu EEG có nhãn, còn distractor được bỏ qua khi tạo mẫu hai lớp.

Quy trình đã chạy dùng 42 người, ICA riêng từng người và 16 kênh đầu vào. Mẫu cuối có kích thước **16 kênh × 129 điểm thời gian**, lấy từ 0 đến 1.000 ms sau kích thích, sau khi chỉnh nền bằng khoảng trước kích thích. Bộ hiện tại có **24.226 mẫu**, chia **34 người train / 8 người test**.

ML dùng 320 đặc trưng trung bình theo cửa sổ 50 ms để chạy baseline, Shrinkage LDA và SVM RBF. DL dùng EEG trực tiếp cho Compact EEG CNN và EEG-Inception. Đánh giá gồm macro-F1 và confusion matrix; balanced accuracy được tính bổ sung trong báo cáo. DL dùng 4 người validation trong nhóm 34 người train, để chọn checkpoint từ 100 epochs.

## Các thư mục và file chính

```text
TTKHTKVUD_26-27/
├── README.md
├── requirements.txt
├── configs/
│   ├── experiment.yaml                  # Cấu hình toàn bộ kênh cho thử nghiệm
│   ├── experiment_16ch.yaml              # Cấu hình 16 kênh đang dùng
│   └── splits.json                      # Người train/test và đường dẫn mẫu trên server
├── scripts/
│   ├── GUIDE.md                         # Vai trò của từng script
│   ├── download_data.py
│   ├── build_data_report.py
│   ├── preprocess.py
│   ├── apply_saved_ica.py
│   ├── create_splits.py
│   ├── extract_features.py
│   ├── train_ml.py
│   ├── train_dl.py
│   ├── visualize.py
│   └── cleanup_intermediate.py
├── data/
│   ├── raw/ds004621/                     # EEG Oddball gốc, trên server
│   └── processed/
│       ├── ica_cleaned_all/              # EEG liên tục sau bước áp dụng ICA
│       └── channels16/
│           ├── after_ica/               # 42 file NPZ cuối để tạo đầu vào model
│           └── features50ms.npz         # 320 đặc trưng cho ML
├── results/
│   ├── channels16/preprocessing_reports/
│   │   └── after_ica/                   # Kiểm kê mẫu cuối, lỗi và trạng thái ICA
│   ├── preprocessing_evidence/          # Bằng chứng thử nghiệm ban đầu sub-01
│   ├── metrics/ml/                      # Kết quả baseline, LDA, SVM
│   ├── metrics/dl/                      # Kết quả DL và lịch sử epoch
│   ├── checkpoints/dl/                  # Trọng số, chuẩn hóa và thông tin chia người
│   └── figures/                         # Hình kết quả và số liệu visualization
└── docs/
    ├── report/
    │   ├── report.md                    # Bản thảo tiếng Anh
    │   ├── report_vi.md                 # Bản tiếng Việt để đối chiếu
    │   ├── report_complete.pdf          # PDF tiếng Anh
    │   ├── report_complete_vi.pdf       # PDF tiếng Việt
    │   ├── data_acquisition.pdf         # Kiểm kê tải dữ liệu, nguồn số liệu báo cáo
    │   └── figures/                     # Ảnh được chèn trong báo cáo
    └── presentation/                    # Vị trí lưu bài thuyết trình
```

`preprocessing_reports` là tên mới của `quality/preprocessing`. Nó chứa nhật ký chất lượng và mô hình ICA, còn tín hiệu để học nằm trong `data/processed/`. Các JSON lịch sử giữ đường dẫn cũ để bảo toàn bản ghi; code đọc báo cáo hỗ trợ ánh xạ sang tên mới. Khi pull trên server, Git chuyển các file được theo dõi sang đường dẫn mới. File chưa được Git theo dõi ở đường dẫn cũ có thể vẫn còn tại đó.

Vai trò từng script được tóm tắt trong [scripts/GUIDE.md](scripts/GUIDE.md). Thư mục `docs/report/figures/` giữ ảnh dùng trong báo cáo; `results/figures/` chứa đầu ra đầy đủ của script vẽ.

## Chuẩn bị trên server

Chạy từ thư mục gốc repo. Dữ liệu, trích đặc trưng và huấn luyện thực hiện trên server.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

Nếu venv đã có, chỉ cần kích hoạt. DL và bước occlusion mặc định dùng CUDA. Kiểm tra bằng:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Thứ tự chạy từ đầu

Các bước dưới phục vụ tái lập quy trình. Nếu đã có bộ mẫu cuối và split hợp lệ, bắt đầu ở bước 6.

### 1. Tải riêng Oddball và tạo kiểm kê

```bash
python scripts/download_data.py --dry-run
python scripts/download_data.py --subjects sub-01
python scripts/download_data.py
python scripts/build_data_report.py --manifest data/raw/ds004621/download_manifest.json
```

Dry-run in danh sách và dung lượng. Manifest tải đầy đủ ghi nguồn, hash và số sự kiện theo người. PDF ở bước này là kiểm kê thu thập dữ liệu.

### 2. Fit ICA và tạo nhật ký ban đầu

```bash
python scripts/preprocess.py --config configs/experiment_16ch.yaml
```

Script lọc, tham chiếu, fit ICA trên các kênh khả dụng rồi tạo bản mẫu ban đầu. Khi chưa có quyết định loại thành phần, các thành phần được lưu để xem xét. Nhật ký và mô hình ở `results/channels16/preprocessing_reports/`.

### 3. Áp dụng ICA và lưu EEG liên tục

Quyết định thủ công của sub-01 đã lưu trong `results/channels16/preprocessing_reports/ica_review_sub01.json`. Để tái lập quyết định đã duyệt trong nhật ký, chạy trước:

```bash
python scripts/preprocess.py --config configs/experiment_16ch.yaml --subjects sub-01 --ica-review results/channels16/preprocessing_reports/ica_review_sub01.json
python scripts/apply_saved_ica.py --output-dir data/processed/ica_cleaned_all
```

Mỗi người có EEG FIF, hình trước/sau và nhật ký. Quy tắc tự động yêu cầu một ứng viên chung của Fp1/Fp2 có độ lớn tương quan từ 0,9 ở cả hai kênh. Phiên bản đã chốt loại thành phần ở 29 người và giữ EEG sau lọc ở 13 người còn lại. Đây là quyết định của phiên bản thí nghiệm hiện tại.

Script báo lỗi nếu EEG đầu ra đã tồn tại. Khi thử quy tắc mới, dùng `--output-dir` khác và truyền thư mục tương ứng cho bước tạo mẫu.

### 4. Tạo mẫu cuối từ EEG đã dựng lại

```bash
python scripts/preprocess.py --config configs/experiment_16ch.yaml --cleaned-dir data/processed/ica_cleaned_all
```

Đầu ra là `data/processed/channels16/after_ica/sub-XX.npz`. Mỗi file lưu `X`, `y`, tên kênh, thời gian và mã người. Báo cáo cuối nằm ở `results/channels16/preprocessing_reports/after_ica/`.

### 5. Cố định train/test theo người

```bash
python scripts/create_splits.py
```

Script kiểm tra mẫu và lưu `configs/splits.json`. File hiện tại đã chứa split cố định 34/8 người. Nó dùng đường dẫn tuyệt đối trên server; khi đổi nơi lưu dữ liệu, cần cập nhật kiểm kê. Việc thay file split làm thay hash liên kết với đặc trưng và checkpoint, nên cần giữ đúng phiên bản để tái lập kết quả cũ.

### 6. Trích đặc trưng và chạy ML

```bash
python scripts/extract_features.py
python scripts/train_ml.py
```

Script đọc đường dẫn EEG trong split, tạo đặc trưng 50 ms và train ba mô hình. Kết quả lưu vào `results/metrics/ml/`.

### 7. Chạy hai mạng DL

```bash
python scripts/train_dl.py --epochs 100 --patience 0
```

Hai mạng chạy lần lượt, nhận EEG trực tiếp. Mỗi mạng fit trên 30 người, dùng 4 người validation trong nhóm train, chọn checkpoint có macro-F1 validation cao nhất. Kết quả ở `results/metrics/dl/`, checkpoint ở `results/checkpoints/dl/`.

### 8. Vẽ và giải thích mô hình

```bash
python scripts/visualize.py
```

Script tạo confusion matrix, đường học, so sánh mô hình, ERP Pz, phân bố đặc trưng và occlusion theo khoảng 100 ms. Đầu ra PNG, PDF và JSON ở `results/figures/`.

## Kết quả của phiên bản đã chạy

| Mô hình | Macro-F1 test |
|---|---:|
| Baseline | 0,4622 |
| Shrinkage LDA | 0,7103 |
| SVM RBF | 0,7578 |
| Compact EEG CNN | 0,7937 |
| EEG-Inception | 0,8440 |

Tập test chung gồm 4.534 mẫu của 8 người. Các giới hạn, sự khác nhau về validation giữa ML/DL và cách diễn giải occlusion được trình bày trong [báo cáo](docs/report/report.md).

## Lưu trữ và dọn dữ liệu

Giữ code, config, split, kết quả nhỏ và bằng chứng báo cáo trong Git. `.gitignore` bỏ qua EEG gốc, dữ liệu đã xử lý, venv và checkpoint lớn.

`cleanup_intermediate.py` kiểm tra bộ mẫu cuối và liệt kê file trung gian có thể dọn. `--apply` thực hiện xóa theo danh sách đã kiểm tra. Xem [hướng dẫn từng script](scripts/GUIDE.md) trước khi dùng.

Các PDF tổng hiện được biên soạn từ bản thảo và kiểm tra bố cục riêng. `build_data_report.py` chỉ tạo PDF kiểm kê tải dữ liệu.


## Phân tích bổ sung về thời gian phản ứng

Để khảo sát vì sao EEG-Inception nhạy với khoảng 700–800 ms, trước hết kiểm kê độ trễ từ target đến sự kiện bấm nút S1 trên nhóm train.

```bash
python scripts/analyze_responses.py
```

Kết quả ở results/analysis/response_timing/ gồm summary.json, trials.csv và target_response_times.png. Script đọc bảng sự kiện gốc, chưa giới hạn vào những epoch đã giữ. Quy tắc ghép lấy S1 đầu tiên từ 100 ms đến trước 1.000 ms và trước kích thích kế tiếp. Đây là phép kiểm kê theo thời gian; cần kiểm tra tính phù hợp của quy tắc trước khi liên hệ với dự đoán mô hình.


Sau khi kiểm tra kiểm kê thời gian phản ứng, so sánh mô hình trên các lượt validation đã giữ bằng lệnh sau.

```bash
python scripts/analyze_model_responses.py
```

Đầu ra ở results/analysis/response_models/ gồm summary.json, predictions.csv và hình target_recall_by_response_time.png/PDF. Script dùng mốc mẫu sự kiện gốc để ghép đúng epoch. Ngưỡng nhóm nhanh/giữa/chậm lấy từ phân vị 25/75 của target có phản ứng trong 30 người fit. Bốn người validation được lấy từ checkpoint DL. LDA, SVM và baseline được fit lại trên cùng 30 người cho phân tích này; DL dùng lại checkpoint. Chỉ số theo nhóm target là recall, do mỗi nhóm chỉ gồm target. Phân tích có tính mô tả vì validation đã được dùng để chọn checkpoint và thành phần người trong các nhóm có thể khác nhau.
