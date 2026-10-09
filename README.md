# Tiểu luận — Phát triển các hệ thống thông minh

Nguyễn Minh Vinh · B23DCCN934 · D23CTPM01.

Báo cáo, notebook đã chạy và web dự đoán cho ML, CNN, RNN. Tham khảo bố cục bài mẫu; các mô hình và kết quả trong repository được huấn luyện riêng.

## Sản phẩm

- [Báo cáo PDF 62 trang](report/tieuluan01_CT_08_VinhNM.934.pdf), [bản Markdown có thể sửa](report/Tieu_luan_Nguyen_Minh_Vinh.md).
- [6 notebook có output](notebooks): lịch sử/dataset; ML; CNN; RNN; tổng hợp/deploy.
- **38 checkpoint**: 18 bản đối chiếu NumPy/Keras/PyTorch, 16 mô hình so sánh kiến trúc CNN/RNN và 4 mô hình ML cổ điển.
- [Notebook 06](notebooks/06_architecture_comparison.ipynb) bổ sung kiến trúc như nhóm mô hình trong bài Tư, giữ dataset của tiểu luận hiện tại.
- Scratch có backward và SGD thật: MLP, CNN và RNN BPTT; kiểm tra gradient tại `results/gradient_checks.json`.
- Web có 6 bài toán, chọn 3 cách cài đặt đã train, nhận mẫu test hoặc input mới.

| Chương | Dataset | Mô hình chính | Phần báo cáo |
|---|---|---|---|
| Mở đầu | Phạm vi, mục tiêu và phương pháp | — | 4 trang |
| 1 — Lịch sử AI | CDC, MNIST, AAPL minh họa | — | 8 trang |
| 2 — ML | CDC Diabetes; giá nhà Việt Nam | MLP; thêm baseline tuyến tính/Random Forest | 12 trang |
| 3 — CNN | MNIST; EuroSAT | CNN cơ bản, VGG-style, MobileNet-style, ResNet-style | 14 trang |
| 4 — RNN | AAPL; Online Retail II | Simple RNN, LSTM, GRU, BiLSTM | 14 trang |

Có hai phép so sánh riêng: **đối chiếu framework** trên cùng kiến trúc và khởi tạo (notebook 02–04), và **so sánh thuật toán/kiến trúc khác nhau** (notebook 06). ML so Linear/Logistic Regression (Ridge cho hồi quy), Random Forest, MLP; CNN so 4 kiến trúc nhỏ; RNN so Simple RNN/LSTM/GRU/BiLSTM. Nhóm kiến trúc dùng PyTorch, Adam 0,001, 20 epoch, batch 128, seed 42 và cùng split. Các bản style không phải VGG-11/MobileNetV2/ResNet-18 đầy đủ. Không lấy chênh lệch với nhóm SGD cũ làm bằng chứng riêng cho kiến trúc.

## Kết quả và phạm vi

Nhóm kiến trúc bổ sung: VGG-style đạt Accuracy 97,75% MNIST và 70,65% EuroSAT. Bảng đủ bốn kiến trúc nằm tại `results/architectures/*/comparison.csv`, lựa chọn mặc định dựa trên validation. Mỗi kết quả có history, checkpoint PyTorch, bản NumPy xuất ra và dự đoán từng mẫu test. `results/architecture_verification.json` kiểm tra lại toàn bộ test của 20 mô hình bổ sung/cổ điển. Các số dưới đây là nhóm đối chiếu framework ban đầu, không trộn với nhóm kiến trúc mới.

Số đo test từ lần chạy seed 42: MNIST Accuracy 94,05%; EuroSAT 60,55%; CDC Macro-F1 0,6338; giá nhà RMSE 1,8760 tỷ VND; AAPL RMSE 3,9322 USD. Xem các bảng đầy đủ và kết quả khách hàng ở `results/*/comparison.csv`. Kết quả gần như trùng giữa framework là do đối chiếu cùng phép tính và điều kiện huấn luyện, đã kiểm chứng độc lập checkpoint native.

- CDC gộp **tiền tiểu đường hoặc tiểu đường** thành lớp dương; benchmark 25.000 mẫu sau chia theo nhóm predictor.
- Housing chia theo địa chỉ; 6 đặc trưng số, log-transform và thống kê chỉ fit trên train.
- CNN dùng ảnh 16×16 và tập con có nêu rõ số lượng; EuroSAT chưa chia theo vùng địa lý.
- Chuỗi được chia theo thời gian; AAPL chưa vượt baseline giữ giá phiên trước. Không suy diễn R² cao thành lợi thế đầu tư.
- Chỉ một seed; chưa đánh giá biến thiên nhiều seed hoặc calibration xác suất.

## Chạy web ngay, không cần tải data/train lại

Python 3.11:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-web.txt
python web/app.py
```

Mở http://127.0.0.1:5007. Web dùng NumPy để suy luận từ trọng số xuất ra của từng framework; checkpoint `.keras`/`.pth` gốc được giữ trong `results`. Không cần TensorFlow/PyTorch khi deploy.

## Tái lập notebook

```bash
python -m pip install -r requirements.txt
python tools/download_data.py
python tools/prepare.py
python tools/execute_notebooks.py
python tools/baselines.py
python tools/train_architectures.py
python tools/bundle_architectures.py
python tools/check_gradients.py
python tools/verify.py
python tools/verify_architectures.py
```

Cũng có thể mở từng notebook trong JupyterLab và chạy các cell từ trên xuống. Notebook có giải thích, code huấn luyện inline, bảng và hình output. `tools/build_notebooks.py` chỉ dùng để tạo lại cấu trúc notebook và sẽ xóa output cũ; không cần chạy khi đọc/nộp bài.

Data gốc/cache không commit để tránh repository quá lớn. [Manifest](docs/data_manifest.json) ghi URL, số byte và SHA-256 của sáu file đầu vào. Script tải 5 bộ công khai; **CSV nhà ở cần tải từ link Kaggle trong manifest hoặc dùng file `house_prices.csv` đã có ở Assignment 3**, đặt vào `data/house_prices.csv`. Nếu checksum khác, cần kiểm tra phiên bản thay vì coi là cùng lần thực nghiệm. Dữ liệu đã có sẵn trên máy thực hiện bài. Chỉ xem notebook hoặc chạy web không cần data gốc.

Tạo lại PDF trên Windows: `python tools/build_report.py` (dùng Times New Roman/Consolas trong Windows Fonts). Bản Markdown cho phép sửa trên nền tảng khác.

## Kiểm chứng và triển khai

`results/verification.json` ghi kiểm tra checkpoint, tính lại metric, output notebook và API; `results/gradient_checks.json` ghi kiểm tra sai phân hữu hạn và autograd. Không tạo số liệu giả khi thí nghiệm chưa chạy.

[Web đang chạy trên Render](https://tieuluan-vinh-intellilab.onrender.com) · [Hướng dẫn Render](docs/DEPLOY.md) · [Blueprint](render.yaml). Đã deploy gói Free và xác minh `/health` cùng 18 tuyến dự đoán cơ sở ngày 08/10/2026; bản bổ sung kiểm tra và triển khai ngày 09/10/2026. Kết quả kiểm tra: `results/render_verification.json`; trạng thái: `docs/deployment_status.json`. Dịch vụ Free có thể khởi động chậm sau thời gian không sử dụng.
