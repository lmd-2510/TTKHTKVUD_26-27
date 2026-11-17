# Kế hoạch triển khai mini-project EEG Oddball

## 1. Mục tiêu và trình tự

Dùng tín hiệu EEG của **một lần hiện ký hiệu** để dự đoán lượt đó là `standard` hay `target`. Mỗi lượt cho một mẫu và một nhãn.

### Ba câu hỏi nghiên cứu trong slide

Các câu dưới đây lấy từ trang 142 của `neuroimaging_EEG_20261006.pdf`. Khi viết báo cáo, dùng RQ1–RQ3 để kiểm tra mỗi câu đã có kết quả trả lời.

| Câu hỏi trong slide | Việc cần làm | Kết quả cần có trong báo cáo |
|---|---|---|
| **RQ1.** What is the best single trial classification result for frequent versus target stimuli? | Train các mô hình đã chọn và chấm **macro-F1** trên cùng những người test. | Bảng macro-F1 và ma trận nhầm lẫn. Nêu mô hình có điểm cao nhất **trong các mô hình đã thử** và phạm vi đánh giá. |
| **RQ2.** Is Deep Learning better for single trial classification compared to standard machine learning techniques? | So Compact EEG CNN và EEG-Inception với Shrinkage LDA và SVM RBF. Dùng cùng cách chia người, cùng các lượt và cùng khoảng EEG 0–1.000 ms. | Nêu chênh lệch macro-F1 quan sát được. Dùng ma trận để chỉ rõ bên nào bỏ sót `target` hoặc đoán nhầm `standard` nhiều hơn. |
| **RQ3.** Are the features that the deep learning model discovers explainable? | Giữ nguyên một mạng đã train, che lần lượt mỗi đoạn 100 ms và đo macro-F1. Đối chiếu khoảng nhạy với ERP trung bình ở Pz. | Biểu đồ thay đổi macro-F1 khi che, kèm giải thích về khoảng thời gian mạng nhạy nhất. Nếu kết quả phù hợp với P300, trình bày đó là dấu hiệu phù hợp với giả thuyết. |

```text
Dữ liệu Oddball → Đọc và kiểm tra bản ghi → Tiền xử lý
→ Tạo các lượt EEG có nhãn → Chia theo người
  ├─ ML → Trích đặc trưng 50 ms → Train LDA và SVM
  └─ DL → Đưa đoạn EEG đã xử lý vào mạng → Train hai CNN
→ Chấm các mô hình và baseline trên cùng tập test
→ Vẽ hình và giải thích → Viết báo cáo và chuẩn bị thuyết trình
```

## 2. Dữ liệu cần sử dụng

Dùng **Nencki–Symfonia, OpenNeuro `ds004621`**, ưu tiên xử lý bản ghi **Oddball của cả 42 người**. Số người dùng trong thí nghiệm được chốt sau khi kiểm tra bản ghi và đếm số lượt hợp lệ. Các task MSIT+, SRT và Rest nằm ngoài bài toán này.

| File | Vai trò |
|---|---|
| `*_eeg.eeg` | Chứa số đo EEG liên tục. Đây là nguồn tín hiệu để cắt mẫu. |
| `*_eeg.vhdr`, `*_eeg.vmrk` | Đi cùng file `.eeg` trong bộ BrainVision. Đọc bản ghi bằng cách truyền file `.vhdr` cho MNE. |
| `*_events.tsv` | Nguồn mốc cắt và nhãn. Mỗi sự kiện chỉ dùng một lần khi tạo mẫu. |
| `*_channels.tsv`, `task-oddball_eeg.json` | Mô tả kênh và thiết lập ghi. File JSON dùng chung cho task Oddball. |

| Mã sự kiện | Ý nghĩa | Cách dùng |
|---|---|---|
| `S 5` | Hiện ký hiệu frequent/standard | Cắt một lượt, nhãn `standard = 0`. |
| `S 6` | Hiện ký hiệu target | Cắt một lượt, nhãn `target = 1`. |
| `S 7` | Hiện ký hiệu distractor | Bỏ qua khi tạo mẫu cho bài toán hai lớp. |
| `S 1` | Người tham gia bấm phím | Có thể dùng để phân tích thời gian phản ứng. Việc bấm hay quên bấm không đổi nhãn của kích thích đã hiện. |

## 3. Tiền xử lý

1. **Đọc và kiểm tra bản ghi bằng MNE.** Kiểm tra tần số lấy mẫu, tên kênh, thứ tự kênh và mã sự kiện của từng người. Ưu tiên **toàn bộ kênh EEG** cho lần chạy đầu. Mũ ghi có 128 điện cực, còn số kênh thực tế và kích thước đầu vào lấy từ bản ghi đã đọc. Bộ **16 kênh của báo cáo** là thí nghiệm bổ sung nếu có thời gian.
2. **Kiểm tra kênh xấu trước khi tính tham chiếu.** Kế hoạch là lấy trung bình các kênh EEG tốt làm mốc tham chiếu. Nếu một số người có kênh lỗi hoặc thiếu, chốt cách xử lý và kiểm tra rằng đầu vào cuối cùng có cùng tên và thứ tự kênh giữa các người.
3. **Lọc EEG liên tục.** Điểm xuất phát là bộ lọc **Butterworth IIR bậc 4, 0,5–30 Hz, chạy hai chiều** như báo cáo pilot.
4. **Cắt một epoch quanh mỗi lần hiện ký hiệu.** Dùng đoạn **−200 đến +1.000 ms** quanh `S 5` hoặc `S 6`. Ví dụ, `S 6` tại giây **48,015** của `sub-01` cho một lượt `target` từ giây **47,815 đến 49,015**. Mốc 48,015 giây trở thành `0 ms` trong lượt.
5. **Chỉnh nền.** Với mỗi kênh của mỗi lượt, tính trung bình tín hiệu từ **−200 đến 0 ms** rồi trừ giá trị đó khỏi cả lượt. Ví dụ, nếu mức nền của Pz là `3 µV` và số đo tại `+400 ms` là `10 µV`, giá trị sau chỉnh nền là `7 µV`.
6. **Loại lượt nhiễu và giảm tần số lấy mẫu.** Ngưỡng thử ban đầu lấy từ pilot. Một lượt bị loại nếu có kênh với biên độ đỉnh–đỉnh **trên 150 µV hoặc dưới 0,1 µV**. Sau đó giảm tần số lấy mẫu xuống **128 Hz**. Đây là số điểm đo mỗi giây trên mỗi kênh.
7. **Lấy đầu vào cho mô hình.** Cả ML và DL dùng phần **0 đến +1.000 ms sau chỉnh nền**. Mỗi lượt vẫn được giữ riêng để phân loại từng mẫu. Chỉ lấy trung bình qua nhiều lượt khi vẽ ERP.

**Cần kiểm tra trước khi chạy chính thức.** Lập bảng theo từng người gồm số lượt `standard` và `target` trước và sau khi loại nhiễu, cùng lý do loại. Khi dùng nhiều kênh hơn, quy tắc một kênh vượt ngưỡng sẽ có thể làm mất thêm lượt. Pilot có 84 trong 89 lượt bị loại thuộc `sub-07`, nên đây là bản ghi đáng xem kỹ. Quyết định có thử **ICA** hay không sau khi xem các lượt đã giữ và kiểm tra nhiễu mắt hoặc cơ còn sót.

## 4. Chia tập theo người

- Nếu đủ **42 người hợp lệ**, chia **34 train / 8 test**, gần tỷ lệ **80/20**. Toàn bộ lượt của một người nằm trong cùng một tập. Nếu số người thay đổi sau kiểm tra chất lượng, tính lại cách chia rồi lưu danh sách mã người của mỗi tập.
- **Lần chạy với hai tập** dùng cấu hình, số epoch và quy tắc dự đoán đã định trước. Tám người test dùng để chấm kết quả cuối.
- **Nếu cần chỉnh siêu tham số, early stopping hoặc ngưỡng theo macro-F1**, dành người làm validation bên trong nhóm 34 người, ví dụ **30 train / 4 validation**, và vẫn giữ nguyên 8 người test.
- Dùng cùng cách chia và cùng các lượt cho các mô hình. Các tham số chuẩn hóa và trọng số lớp được tính từ tập train tương ứng.

## 5. Trích xuất đặc trưng (Feature Extraction)

### Bước này làm gì?

Tiền xử lý cho bạn một đoạn EEG đã lọc, căn đúng giờ và chỉnh nền. Trích xuất đặc trưng lấy đoạn đó, tóm tắt nó thành **một hàng số cho LDA hoặc SVM**. Mỗi số trả lời một câu cụ thể, chẳng hạn *“Điện áp trung bình ở Pz từ 400 đến 450 ms sau khi ký hiệu hiện là bao nhiêu?”*

Bạn làm bước này sau tiền xử lý vì mốc `0 ms` của mọi lượt phải là lúc ký hiệu xuất hiện. Khi đó, **cùng tên kênh và cùng cửa sổ** ở hai lượt đều có ý nghĩa giống nhau. Cả hai mạng DL nhận trực tiếp đoạn EEG đã tiền xử lý và tự học đặc trưng trong lúc train.

### Tính thế nào cho một lượt?

1. Lấy phần EEG **0–1.000 ms sau chỉnh nền**. Với mỗi kênh, chia thành **20 cửa sổ liên tiếp 50 ms**, từ `0–50 ms` đến `950–1.000 ms`.
2. Chọn điểm theo thời gian của epoch. Dùng quy tắc **lấy đầu, bỏ cuối**. Ví dụ, `400–450 ms` lấy các điểm từ 400 ms đến trước 450 ms. Tại 128 Hz, mỗi cửa sổ có khoảng **6 hoặc 7 điểm**, nên dùng mốc thời gian của từng điểm khi chia.
3. Tính trung bình trong từng cửa sổ **của từng kênh trong lượt đó**. Nối các số theo một thứ tự kênh và cửa sổ cố định. Một lượt cho một hàng đặc trưng và giữ nhãn `standard` hoặc `target` của lượt đó.

Ví dụ dưới đây chỉ hiện hai kênh và hai cửa sổ của **một lượt target**. Các số đo là minh họa.

| Kênh | Cửa sổ | Các giá trị điện áp đã chỉnh nền (µV) | Một đặc trưng (µV) |
|---|---|---|---:|
| Pz | 400–450 ms | 2, 3, 4, 4, 5, 6 | **4** |
| Pz | 450–500 ms | 4, 5, 6, 6, 7, 8 | **6** |
| Fz | 400–450 ms | −2, −1, 0, 0, 1, 2 | **0** |
| Fz | 450–500 ms | −3, −2, −1, −1, 0, 1 | **−1** |

Khi làm hết các cửa sổ và kênh, **mỗi lượt có `số kênh × 20` đặc trưng**. Nếu bản ghi có đúng 128 kênh thì một lượt cho **2.560 số**. Nếu lấy nhiều lượt để vẽ phân bố đặc trưng **Pz `400–450 ms`**, bạn lấy đúng giá trị của kênh và cửa sổ đó ở mỗi lượt.

### Vì sao chọn đặc trưng này và cần kiểm tra gì?

Đường ERP của `target` trong bài báo gốc nhô lên rõ ở Pz vài trăm mili giây sau kích thích. Giữ riêng các cửa sổ giúp mô hình phân biệt được tín hiệu cao **ở đoạn nào** trong một giây. Slide nêu **band power, entropy và wavelet** như các ví dụ về cách trích đặc trưng. Mỗi cách đo một tính chất khác của EEG. **Điện áp trung bình 50 ms** là bộ đặc trưng đã chọn cho phép so sánh chính. [Hình 5 trong bài báo Nencki–Symfonia](https://doi.org/10.1093/gigascience/giac015)

- **Kiểm tra một lượt bằng tay.** Tính trung bình Pz trong `400–450 ms` và đối chiếu với số do code tạo ra.
- **Kiểm tra cấu trúc đầu vào.** Mỗi hàng ứng với một lượt, có đúng `số kênh × 20` số hữu hạn. Các tập dùng chung thứ tự kênh và cửa sổ.
- **Chuẩn hóa từ train.** Tính trung bình và độ lệch chuẩn của từng cột đặc trưng trên train, rồi áp dụng cùng phép biến đổi lên validation và test nếu có. LDA và SVM dùng cùng bộ cột đặc trưng.

Trong báo cáo, mô tả **cách tính, cửa sổ, số đặc trưng và lý do chọn**. Sau khi chạy, báo cáo LDA và SVM đạt macro-F1 nào với bộ đặc trưng này. Để kết luận cửa sổ 50 ms tốt hơn cách khác, cần so trực tiếp, chẳng hạn **50 ms và 100 ms** trên cùng mô hình và cùng cách chia người.

## 6. Nhánh ML

| Phương pháp | Cấu hình ban đầu |
|---|---|
| **Baseline lớp đông** | `DummyClassifier(strategy="most_frequent")`. Dự đoán lớp xuất hiện nhiều nhất trong train, dự kiến là `standard`. |
| **Shrinkage LDA** | `solver="lsqr"`, `shrinkage="auto"`, `priors=[0.5, 0.5]`. |
| **SVM RBF** | `kernel="rbf"`, `C=1.0`, `gamma="scale"`, `class_weight="balanced"`. |

Với cấu hình chỉ có train và test, dùng nhãn từ `predict()`. Nếu thử nhiều giá trị `C` hoặc ngưỡng khác, chọn trên validation theo người.

## 7. Nhánh deep learning

Cả hai mạng nhận trực tiếp **bảng số `kênh × thời điểm` của một lượt** trong khoảng **0–1.000 ms**. Mạng trả về một dự đoán `standard` hoặc `target` cho cả lượt.

1. **Compact EEG CNN kiểu EEGNet** chạy trước. Dùng kiến trúc gọn của pilot, với kích thước đầu vào khớp số kênh EEG thực tế.
2. **EEG-Inception** chạy tiếp. Mạng dùng các bộ lọc có nhiều độ dài để tìm dạng sóng ERP/P300 ở các khoảng thời gian khác nhau.

**Cấu hình train ban đầu cho cả hai mạng** là AdamW, learning rate `0,001`, batch size `64`, tối đa `35 epoch`. Nếu chỉ có train và test, chạy đủ 35 epoch như đã định trước. Nếu có validation, có thể chọn checkpoint hoặc dừng sớm bằng tập đó. Nếu dùng loss có trọng số cho lớp `target`, tính trọng số từ tỷ lệ lớp của train.

## 8. Đánh giá và visualization

**Đầu ra phân loại cho mỗi mô hình** gồm **ma trận nhầm lẫn 2 × 2 bằng số lượt** và **macro-F1** trên cùng tập test. Dùng thứ tự nhãn `standard`, `target`, với **hàng là nhãn thật** và **cột là nhãn dự đoán**. Macro-F1 là trung bình của F1 hai lớp, mỗi lớp có trọng số bằng nhau.

| Hình hoặc bảng | Cách làm |
|---|---|
| **Tín hiệu EEG ở Pz** | Vẽ một lượt riêng ở Pz, rồi vẽ hai đường ERP trung bình của `standard` và `target` ở Pz. Đánh dấu `0 ms` là lúc hiện ký hiệu. Chỉ vẽ Pz cho lần đầu. |
| **Phân bố đặc trưng** | Vẽ hai boxplot cho một đặc trưng ML, ví dụ điện áp trung bình ở Pz trong `400–450 ms`. Mỗi lượt góp một giá trị. |
| **Kết quả phân loại** | Trình bày ma trận nhầm lẫn và macro-F1 cho các mô hình trên cùng tập test. |
| **Giải thích mô hình DL** | Dùng một mạng đã train. Đo macro-F1 với tín hiệu đầy đủ, rồi lần lượt che từng cửa sổ **100 ms** và dự đoán lại trên cùng các lượt. Mỗi lần chỉ thay một cửa sổ, ví dụ thay bằng mức trung bình từng kênh tính từ train. Vẽ chênh lệch giữa điểm gốc và điểm khi che. |

Phép che giữ cố định trọng số và quy tắc dự đoán của mạng. Nếu che một đoạn làm điểm giảm, đoạn đó có ích cho dự đoán trong phép thử này. Đối chiếu với ERP ở Pz để xem thời điểm có phù hợp với phản ứng P300 hay không. Kết quả che theo thời gian cho biết **khoảng thời gian** có ích cho mạng. Khi diễn giải, lưu ý tín hiệu trong khoảng đó cũng có thể liên quan đến động tác bấm phím hoặc nhiễu còn sót.

## 9. Documentation và checklist triển khai

Báo cáo và thuyết trình bằng **tiếng Anh**. Báo cáo nên ghi nguồn dữ liệu, số người và số lượt thực tế, preprocessing, cách chia người, cấu hình mô hình và kết quả để trả lời ba câu hỏi nghiên cứu. Ghi công việc thực tế và phần trăm đóng góp của từng người, với tổng bằng **100%**.

Theo slide, gửi báo cáo **ít nhất một tuần trước thuyết trình** cho Guy Nagels (`guy.nagels@vub.be`), thầy Trần Quốc Sang và thầy Nguyễn Tiến Dũng. Bài trình bày dài **tối đa 10 phút**. Cả nhóm có mặt và chuẩn bị trả lời câu hỏi.

- [ ] Đọc thử bản ghi Oddball của `sub-01`. Cắt và vẽ một lượt `target` ở Pz để kiểm tra mốc và nhãn.
- [ ] Tiền xử lý toàn bộ người tham gia. Kiểm tra kênh xấu, số lượt bị loại và đầu ra sau cắt đoạn.
- [ ] Chốt số người hợp lệ. Lưu danh sách người của các tập trước khi train.
- [ ] Tạo đặc trưng 50 ms. Kiểm tra một lượt bằng tay và xác nhận các cột đúng thứ tự.
- [ ] Chạy baseline, LDA và SVM.
- [ ] Chạy Compact EEG CNN rồi EEG-Inception trên cùng cách chia người.
- [ ] Xuất ma trận nhầm lẫn và macro-F1. Hoàn thành các hình Pz, phân bố đặc trưng và che theo thời gian.
- [ ] Điền kết quả thực nghiệm vào báo cáo và bài thuyết trình. Kiểm tra có câu trả lời và kết quả minh họa cho từng RQ1–RQ3. Ghi rõ mọi thay đổi so với pilot.

## 10. Cấu trúc repo và cách chạy trên server

```text
TTKHTKVUD_26-27/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── experiment.yaml                  # Đường dẫn và thông số thí nghiệm
│   └── splits.json                      # Danh sách người của từng tập
├── data/
│   ├── raw/                             # Bản ghi EEG Oddball gốc
│   └── processed/                       # Epoch, nhãn và mã người
├── scripts/
│   ├── download_data.py
│   ├── build_data_report.py
│   ├── preprocess.py
│   ├── extract_features.py
│   ├── train_ml.py
│   ├── train_dl.py
│   ├── visualize.py
│   └── run_pipeline.py                  # Điều phối các script theo thứ tự
├── results/
│   ├── quality/
│   ├── metrics/
│   ├── checkpoints/
│   └── figures/
└── docs/
    ├── eeg-oddball-ke-hoach-trien-khai.md
    ├── report/
    └── presentation/
```

Dùng cùng cấu trúc code trên máy cá nhân và server. Ba đường dẫn **EEG gốc**, **EEG đã tiền xử lý** và **đầu ra** cấu hình qua `experiment.yaml`. Ví dụ, trên server Linux, `raw_dir` có thể trỏ tới `/mnt/data/oddball/raw` khi dữ liệu đã nằm trên ổ riêng. Giữ code và danh sách chia người trong Git. Đưa dữ liệu lớn và checkpoint vào `.gitignore`.

Bước tải đọc từ bản sao S3 công khai của OpenNeuro và mặc định lưu vào `data/raw/ds004621/`. Khi dùng ổ riêng, truyền `--output-dir` cho script tải. Manifest ghi nguồn, hash và số sự kiện theo người. Xem [lệnh chạy trên server](../scripts/README.md).

**Nên có `run_pipeline.py` khi bắt đầu chạy nhiều lần.** Script này gọi các bước đã có, dùng cùng config và danh sách chia người. Sau khi chuẩn bị dữ liệu gốc và chốt danh sách người, thứ tự chạy dự kiến là:

```text
preprocess → extract_features → train_ml → train_dl → visualize
```

Pipeline kiểm tra đầu vào trước mỗi bước. Ví dụ, trước khi train, kiểm tra danh sách người có trong dữ liệu đã xử lý và các tập dùng mã người riêng. Nếu một bước lỗi, báo tên bước và dừng để có thể sửa rồi chạy tiếp. Các script thành phần vẫn chạy riêng được để kiểm tra từng bước.

**Lệnh dự kiến khi script đã được triển khai**, chạy từ thư mục gốc của repo:

```bash
python scripts/run_pipeline.py --config configs/experiment.yaml
```

Khi đã có dữ liệu tiền xử lý, đặc trưng và kết quả ML phù hợp với lượt chạy hiện tại, có thể bắt đầu lại từ DL bằng `--from-stage train_dl`. Script tải dữ liệu được gọi riêng khi cần chuẩn bị bản ghi gốc.

## 11. Những điểm cần xác nhận trước khi chạy toàn bộ dữ liệu

| Việc cần nhớ | Kế hoạch hiện tại | Cần xác nhận và ghi lại khi chạy thử |
|---|---|---|
| **Người tham gia và nhãn** | Xử lý Oddball của cả 42 người. Lấy S5 và S6 làm hai lớp. | Kiểm kê file đọc được và đếm từng lớp trước/sau lọc. `sub-02`, `sub-14` và `sub-17` bị loại khỏi **thống kê hành vi** trong bài báo gốc do thường phản ứng với distractor. Quyết định về trial EEG của nhóm dựa vào tiêu chí dữ liệu đã định và được ghi rõ theo từng người. [Bài báo gốc, Bảng 5](https://doi.org/10.1093/gigascience/giac015) |
| **Số kênh và tham chiếu** | Chạy **toàn bộ kênh EEG trước**. Bộ 16 kênh để thử sau nếu có thời gian. Tính tham chiếu từ các kênh EEG tốt. | Kiểm tra số kênh thực tế, tên, thứ tự và danh sách kênh xấu. Mũ có 128 điện cực, danh mục NEMAR ghi 127 kênh. Kiểm tra FCz có mặt thế nào trong dữ liệu gốc và đối chiếu cách MNE xử lý điện cực tham chiếu gốc khi đổi mốc. Ghi cụ thể các kênh dùng để tính tham chiếu và số kênh cuối cùng vào mô hình. [Danh mục NEMAR](https://nemar.org/dataset/on004621) · [MNE về tham chiếu](https://mne.tools/stable/generated/mne.set_eeg_reference.html) |
| **Ngưỡng loại lượt** | Thử ngưỡng của pilot `150 µV` và `0,1 µV`. | So số lượt mất theo người, theo lớp và theo nguyên nhân. Xem riêng `sub-07`, người chiếm 84 trong 89 lượt bị loại ở pilot. Với toàn bộ kênh, kiểm tra xem có một kênh nào khiến nhiều lượt bị loại hay không. Ghi ngưỡng cuối và lý do nếu thay đổi. [MNE Epochs, các mục reject và flat](https://mne.tools/stable/generated/mne.Epochs.html) |
| **ICA** | Quyết định còn mở. Pilot chưa dùng ICA. | Vẽ thử các lượt giữ lại ở Pz và vài kênh trán để xem nhiễu mắt hoặc cơ còn rõ không. Nếu cần thử ICA, chạy trên một ít bản ghi, xem hình trước/sau và ghi thành phần nào bị bỏ cùng lý do. Chốt có áp dụng vào lần chạy chính hay không sau bước kiểm tra này. [MNE về ICA](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html) |
| **Biên thời gian và đặc trưng** | `−200 đến 0 ms` để chỉnh nền. `0 đến +1.000 ms` cho ML/DL. ML dùng 20 cửa sổ 50 ms. | Kiểm tra mốc 0, tần số sau resample, số điểm và đơn vị của một epoch. Đối chiếu một feature Pz tính tay. Kiểm tra CNN nhận đúng phần sau kích thích vì CNN trong pilot còn nhận đoạn trước kích thích. |
| **Chia người và chấm** | `34 train / 8 test` nếu đủ 42 người. Macro-F1 và ma trận nhầm lẫn. | Lưu danh sách người và kiểm tra hai tập không trùng. Nếu cần chọn ngưỡng, siêu tham số hoặc dừng sớm, tạo validation theo người bên trong 34 người. Giữ cùng danh sách test cho các mô hình. |
| **Nguồn dữ liệu và tài nguyên** | Dùng cùng cấu trúc repo trên máy cá nhân, server hoặc Kaggle. | Ghi phiên bản hoặc manifest của dữ liệu. Xác nhận đường dẫn đọc/ghi và dung lượng còn trống. Chạy thử một vài bản ghi và một batch của từng CNN để xem kích thước dữ liệu, RAM/VRAM và thời gian trước khi chạy cả bộ. |

Bắt đầu kiểm tra với **`sub-01`** để xác nhận số kênh, mốc cắt và đặc trưng mẫu, rồi xem thêm **`sub-07`** để kiểm tra xử lý nhiễu. Ghi quyết định đã dùng vào config hoặc báo cáo để lúc train và viết kết quả có thể tra lại.

## Nguồn đối chiếu

### Đề bài và báo cáo pilot

- **Slide môn học — `neuroimaging_EEG_20261006.pdf`.** Trang 138 nêu phép so sánh ML và DL. Trang 139 giới thiệu dataset, trang 140 có hình ERP Oddball và trang 141 liệt kê công cụ. Trang 142 nêu ba câu hỏi nghiên cứu. Trang 143 đưa ví dụ về feature extraction, mô hình ML và hình minh họa. Trang 144–145 nêu hạn gửi, cách trình bày và cách chấm.
- **`Report_EN.pdf`.** Mục 2 và bảng trang 3 cho biết phạm vi pilot 8 người và số lượt giữ hoặc loại. Mục 3–5 ghi preprocessing, đặc trưng 50 ms, kiến trúc Compact EEG CNN, thông số train và cách chia người. Mục 7–8 mô tả phép che tín hiệu và các giới hạn khi diễn giải. Mục 10 liệt kê các file dùng để tái lập.
- **Các lựa chọn của kế hoạch lần này** gồm ưu tiên toàn bộ kênh, chia 34/8 nếu đủ 42 người, cùng dùng khoảng 0–1.000 ms cho ML và DL, và lấy macro-F1 làm chỉ số chính. Bộ lọc 0,5–30 Hz, ngưỡng 150/0,1 µV và cấu hình train là điểm xuất phát từ pilot. Các lựa chọn này cần được ghi cùng kết quả thực tế khi triển khai.

### Dataset, file EEG và mã sự kiện

- **Dzianok và cộng sự (2022).** Dùng phần *Data acquisition* để tra thiết bị, tham chiếu FCz và tốc độ ghi. Dùng Bảng 3 để đối chiếu S5/S6/S7/S1, phần *Oddball task* để tra thiết kế, Bảng 5 để hiểu lý do loại ba người khỏi thống kê hành vi, và Hình 5 để đối chiếu ERP ở Pz. [Bài báo gốc GigaScience](https://doi.org/10.1093/gigascience/giac015)
- **Bản dữ liệu công khai.** OpenNeuro có snapshot 1.0.4. README cung cấp thông tin trích dẫn và DOI của bộ dữ liệu gốc trên GigaDB. Khi tải thật, lưu phiên bản hoặc nguồn tải cùng mã hash của các file đã dùng. [OpenNeuro ds004621 v1.0.4](https://openneuro.org/datasets/ds004621/versions/1.0.4) · [Repo dữ liệu và README](https://github.com/OpenNeuroDatasets/ds004621) · [DOI bộ dữ liệu gốc trên GigaDB](https://doi.org/10.5524/100990)
- **Đối chiếu file thật.** Metadata Oddball ghi tốc độ gốc 1.000 Hz và tham chiếu FCz. Bảng sự kiện của `sub-01` có S6 ở 48,015 giây. NEMAR liệt kê 127 kênh, còn bài báo mô tả mũ 128 điện cực. Khi dựng mô hình, lấy số kênh, tên và thứ tự từ bản ghi MNE đọc được. [Metadata Oddball](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004621/master/task-oddball_eeg.json) · [Bảng sự kiện sub-01](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004621/master/sub-01/eeg/sub-01_task-oddball_events.tsv) · [Danh mục NEMAR](https://nemar.org/dataset/on004621)
- **Đọc BrainVision và BIDS.** Xem cách MNE đọc bộ `.vhdr/.vmrk/.eeg`, ý nghĩa trường trong `events.tsv` và phần mô tả `channels.tsv`. Dùng Bảng 3 của bài báo để giải nghĩa các mã sự kiện của riêng dataset này. [MNE BrainVision](https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html) · [BIDS Events](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html) · [BIDS EEG](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html)

### Preprocessing và kiểm tra chất lượng

- **Kênh xấu và tham chiếu.** MNE mô tả cách đánh dấu kênh xấu và cách loại các kênh đã đánh dấu khỏi phép tính tham chiếu trung bình. Bài PREP giải thích vì sao một kênh nhiễu có thể làm lệch mốc tham chiếu chung. [MNE về kênh xấu](https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html) · [MNE về tham chiếu](https://mne.tools/stable/generated/mne.set_eeg_reference.html) · [Bigdely-Shamlo và cộng sự, PREP](https://doi.org/10.3389/fninf.2015.00016)
- **Lọc, cắt và chỉnh nền.** Tra cách lọc và giảm tần số lấy mẫu, quy ước biên epoch và baseline, cùng cách MNE áp dụng `reject` và `flat` theo biên độ đỉnh–đỉnh. Ngưỡng EEG truyền vào MNE tính bằng volt, nên 150 µV tương ứng với `150e-6` V và 0,1 µV tương ứng với `0.1e-6` V. [Hướng dẫn lọc và resample](https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html) · [MNE Epochs](https://mne.tools/stable/generated/mne.Epochs.html)
- **ICA nếu cần.** Dùng hướng dẫn MNE để xác định thành phần liên quan đến mắt hoặc cơ trước khi quyết định loại thành phần. Quyết định áp dụng dựa vào tín hiệu và mức nhiễu quan sát được sau preprocessing. [MNE về sửa nhiễu bằng ICA](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html)

### Feature extraction, chia người và ML

- **Đặc trưng trung bình theo thời gian.** Báo cáo pilot dùng 20 cửa sổ 50 ms. Speier và cộng sự dùng cách tương tự trong một thí nghiệm P300 speller, với khoảng 58,8 ms mỗi giá trị trung bình. Đây là tiền lệ cho cách biểu diễn, còn kết quả với đúng 50 ms sẽ được đo trên Nencki–Symfonia. [Channel Selection Methods for the P300 Speller, mục 2.2.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC4106671/)
- **Chia theo người, chuẩn hóa và chọn ngưỡng.** Dùng tài liệu về dữ liệu có nhóm để kiểm tra các tập theo mã người. Tra `StandardScaler` khi tính tham số trên train. Nếu chỉnh ngưỡng để tối ưu macro-F1, dùng người thuộc validation bên trong tập train. [Chia dữ liệu theo nhóm](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) · [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) · [Chọn ngưỡng dự đoán](https://scikit-learn.org/stable/modules/classification_threshold.html)
- **Cấu hình ba phương pháp ML.** Đối chiếu cách dự đoán lớp đông, tùy chọn shrinkage và priors của LDA, cùng kernel, `C`, `gamma` và trọng số lớp của SVM. [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) · [LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) · [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

### Deep learning và cách lưu mô hình

- **Compact EEG CNN.** Mục 4 của `Report_EN.pdf` mô tả kiến trúc nhóm dùng trong pilot. Paper EEGNet là nguồn cho ý tưởng học theo thời gian rồi kết hợp các kênh. Mục 2.1.1 của paper có thí nghiệm P300 trên cửa sổ 0–1 giây ở 128 Hz. [Lawhern và cộng sự, EEGNet](https://arxiv.org/pdf/1611.08024)
- **EEG-Inception.** Paper gốc giải thích các bộ lọc nhiều độ dài thời gian cho ERP. Tài liệu Braindecode mô tả bản triển khai PyTorch và các tham số đầu vào như số kênh, tần số lấy mẫu và số điểm thời gian. [Santamaría-Vázquez và cộng sự, EEG-Inception](https://doi.org/10.1109/TNSRE.2020.3048106) · [Braindecode EEGInceptionERP](https://braindecode.org/stable/generated/braindecode.models.EEGInceptionERP.html)
- **Train và checkpoint.** Tra tham số AdamW và cách đặt `pos_weight` nếu đầu ra là một logit nhị phân. Lưu checkpoint cùng số kênh, cấu hình và danh sách chia người để dùng lại khi làm phép che. [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) · [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) · [Lưu và nạp mô hình PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) · [PyTorch về seed và khả năng tái lập](https://docs.pytorch.org/docs/stable/notes/randomness.html)

### Metric, biểu đồ và giải thích mô hình

- **Chấm hai lớp.** Macro-F1 là trung bình đều của F1 từng lớp. Ma trận nhầm lẫn dùng số lượt, với hàng là nhãn thật và cột là nhãn dự đoán. [F1 trong scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) · [Confusion matrix trong scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- **Vẽ EEG và phân bố đặc trưng.** Dùng Hình 5 của bài báo gốc làm mẫu cách trình bày ERP. Với boxplot, mỗi lượt góp một giá trị điện áp trung bình của cùng kênh và cửa sổ. [Hình 5 trong bài báo Nencki–Symfonia](https://doi.org/10.1093/gigascience/giac015) · [Tài liệu Matplotlib về boxplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html)
- **Che theo thời gian.** Mục 7 của báo cáo pilot mô tả cách thay từng đoạn 100 ms bằng mức nền rồi dự đoán lại. Tài liệu Captum giải thích cơ chế thay một vùng đầu vào khi giữ mô hình cố định. Trong kế hoạch này, chênh lệch được tính bằng macro-F1. Đó là phép đo độ nhạy với vùng tín hiệu bị che. [Captum Occlusion](https://captum.ai/api/occlusion.html)

### Môi trường chạy và tái lập

- **Server hoặc Kaggle.** Đặt đường dẫn dữ liệu và kết quả trong config. Nếu chạy trên Kaggle, dùng tài liệu notebook để gắn dữ liệu và lưu đầu ra. Nếu dùng nhiều GPU trong cùng một lượt train, tra hướng dẫn PyTorch về DDP. [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) · [PyTorch DistributedDataParallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- **Trước lần chạy chính**, lưu phiên bản dữ liệu hoặc manifest, danh sách người của các tập, cấu hình và seed, phiên bản thư viện, số lượt giữ hoặc loại theo người và tên kênh đầu vào. Đây là các thông tin cần có để đối chiếu kết quả thật với pipeline đã mô tả.
