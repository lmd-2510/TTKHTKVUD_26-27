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

## Preprocessing trên server

Chạy thử một người trong venv của repo. Giới hạn số luồng tính toán để chia sẻ CPU với các tiến trình khác trên server.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/preprocess.py --subjects sub-01
```

- Script đối chiếu mốc `S 5` và `S 6` giữa `events.tsv` và BrainVision, áp dụng tham chiếu trung bình, lọc 0,5–30 Hz và fit ICA riêng cho từng người.
- ICA fit trên bản sao 250 Hz được lọc thông cao 1 Hz. Mô hình mặc định có tối đa 30 thành phần. Khi có EOG, MNE dùng tương quan với EOG để tìm thành phần mắt. Nếu chỉ có kênh trán làm tín hiệu tham khảo, script lưu điểm và thành phần ứng viên để xem lại. Trạng thái `manual_review_required` cho biết chưa có quyết định loại thành phần. File ICA đã fit nằm trong thư mục QC.
- Script resample tín hiệu cùng mốc sự kiện về 128 Hz rồi cắt lượt. Chỉnh nền khoảng −200 đến 0 ms trên lưới mẫu 128 Hz, loại lượt theo biên độ đỉnh-đỉnh trên toàn epoch và giữ 0–1.000 ms cho model. Ngưỡng là 150 µV và 0,1 µV.
- File `data/processed/sub-01.npz` chứa `X` theo thứ tự lượt × kênh × thời gian, `y` với 0 = standard và 1 = target, tên kênh, thời điểm tương đối và mẫu sự kiện gốc. Tín hiệu lưu theo volt. Trước khi gộp người để train, kiểm tra tên và thứ tự kênh trong từng file.
- Xem `results/quality/preprocessing/sub-01.json` hoặc `summary_sub-01.json` để đối chiếu số lượt mỗi lớp, lý do loại lượt, kênh và trạng thái ICA. Một lượt có thể có nhiều lý do loại. Dùng trạng thái ICA trong QC để biết file nào còn cần xem lại.

Khi đã duyệt thành phần ICA, có thể truyền `--ica-review /path/to/ica_review.json`. JSON ánh xạ từng mã người tới `exclude` (danh sách chỉ số thành phần, bắt đầu từ 0) và `reason` (căn cứ thực tế). Danh sách rỗng có ý nghĩa giữ tất cả sau khi đã duyệt. Giữ nguyên cấu hình và phiên bản thư viện khi dùng lại chỉ số thành phần từ lần chạy thử.

Chạy ba người cần xem riêng sau khi kiểm tra `sub-01`.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/preprocess.py --subjects sub-02 sub-14 sub-17
```

Bỏ `--subjects` để chạy đủ 42 người. `summary_all.json` ghi kết quả và lỗi theo người. Kiểm tra tổng hợp trước khi chốt danh sách người dùng để train/test.

## Các script tiếp theo

| Script | Vai trò |
|---|---|
| `extract_features.py` | Tạo đặc trưng 50 ms cho ML. |
| `train_ml.py` | Chạy baseline, Shrinkage LDA và SVM RBF. |
| `train_dl.py` | Chạy Compact EEG CNN và EEG-Inception. |
| `visualize.py` | Vẽ các hình và đo ảnh hưởng khi che từng khoảng thời gian. |
| `run_pipeline.py` | Điều phối các bước với cùng config và cách chia người. |

Trình tự thực hiện được ghi trong [kế hoạch](../docs/eeg-oddball-ke-hoach-trien-khai.md).

## Thử 16 kênh sau ICA

Chạy cấu hình riêng để fit ICA trên toàn bộ kênh khả dụng rồi chọn 16 kênh trước khi kiểm tra ngưỡng. Kết quả ở data/processed/channels16/ và results/channels16/quality/preprocessing/.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/preprocess.py --config configs/experiment_16ch.yaml --subjects sub-01 --ica-review /path/to/ica_review.json
```

Dùng lại quyết định ICA đã duyệt cho từng người, cùng cấu hình fit và phiên bản thư viện. Với sub-01, thành phần 0 đã được duyệt là nhiễu mắt. Bỏ --subjects để chạy cả bộ. Khi thiếu kênh đã chọn, script báo lỗi theo người. Mặc định experiment.yaml vẫn dùng all_eeg.


## Classical ML on the server

Run from the repository root with the project venv active.

```bash
python scripts/extract_features.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/train_ml.py
```

The extractor reads participant NPZ paths from configs/splits.json. Each channel contributes 20 means over consecutive 50 ms windows between 0 and 1000 ms. The last bin includes the endpoint at 1000 ms. At 128 Hz, bins contain different numbers of samples, so assignment uses the saved timestamps. Flattening uses channel-major order. Values are converted from volts to microvolts. Output is data/processed/channels16/features50ms.npz with participant identities and a SHA-256 of the split file.

The three initial models are a majority baseline, LDA with automatic covariance shrinkage and equal class priors, and an RBF SVM with C=1, gamma=scale and balanced class weights. LDA/SVM use a StandardScaler fitted on training participants only. Hyperparameters are fixed for this first run. If tuning is added later, participant-level validation inside the 34 training participants is required.

Each model writes macro-F1 and a 2x2 confusion matrix to results/channels16/metrics/ml/<model>.json. Matrix rows are true labels, columns are predicted labels, ordered standard then target. Per-model progress and metrics are printed during execution. These runs use CPU.

Optional --models selects a subset, for example --models majority_baseline shrinkage_lda. Training and extraction must be run on the server. Feature extraction and command-line parsing have been checked locally with synthetic arrays; real-data training is pending.
