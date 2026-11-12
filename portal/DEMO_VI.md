# Hướng dẫn demo FedMedAI

Để cài đặt từng máy từ Git, xem [hướng dẫn setup chi tiết cho cả nhóm](TEAM_SETUP.md). Tài liệu đó có lệnh riêng cho server/client, môi trường Python, Firewall, pairing, cập nhật code và kiểm tra Jetson.

## 1. Hiểu cách chạy

- **Laptop server:** chạy portal Coordinator và Flower server. Người điều phối tạo phiên huấn luyện, chọn các máy tham gia, xem tiến độ và quản lý mô hình.
- **Mỗi laptop client / Jetson:** chạy portal Institution và Flower client. Người vận hành kiểm tra dữ liệu cục bộ, chấp nhận tham gia, xem tiến độ của máy và chạy dự đoán.
- **Trình duyệt:** chỉ là giao diện điều khiển. Đóng tab không dừng huấn luyện; phải giữ các dịch vụ Python đang chạy.

Website trên Jetson không cần màn hình riêng: mở `http://IP_JETSON:8001` từ laptop trên cùng mạng. Ảnh đưa vào Predict được gửi trực tiếp tới dịch vụ Institution đó.

## 2. Chuẩn bị trước buổi bảo vệ

Trên Windows, mở PowerShell ở thư mục `FedMed_20K`. Cài Node.js 22.12+ trên máy dùng để build. Thay các đường dẫn Python bên dưới bằng đường dẫn thật của máy:

```powershell
# Tạo môi trường web riêng và build giao diện (chỉ cần làm khi cài đặt/cập nhật).
.\scripts\setup_portal.ps1 -Python "C:\DuongDan\Python\python.exe"

# Laptop server: thay IP bằng IPv4 LAN thật của server.
.\scripts\start_portal.ps1 -Mode coordinator -Name "FedMedAI Research Network" `
  -TrainerPython "D:\FPT\KLTN\Code\.venv\Scripts\python.exe" -ServerAddress 192.168.1.10

# Laptop client: chạy lệnh này trên từng máy, đặt tên cơ sở khác nhau.
.\scripts\start_portal.ps1 -Mode institution -Name "Institution A" `
  -TrainerPython "C:\DuongDan\MoiTruongTraining\Scripts\python.exe"
```

Lần đầu chương trình yêu cầu tạo mật khẩu, tài khoản mặc định là `admin`. Mở `http://IP_SERVER:8000` cho Coordinator và `http://IP_CLIENT:8001` cho Institution. Giữ terminal dịch vụ mở. Nếu muốn chạy hai vai trò trên cùng máy để thử giao diện, truyền `-StateDir .portal-coordinator` và `-StateDir .portal-institution` riêng biệt.

Với Jetson, copy source và `web/dist` đã build; dùng `scripts/start_portal.sh institution` sau khi đặt `FEDMED_WEB_PYTHON` và `FEDMED_TRAINER_PYTHON`. Không dùng môi trường web để cài đè PyTorch/JetPack. Xem phần **Jetson compatibility gate** trong `portal/README.md`.

1. Cài và build theo `portal/README.md`. Copy cùng phiên bản source và thư mục `web/dist` sang các máy; mỗi máy tạo `.portal` riêng.
2. Kiểm tra IP của laptop server. Khởi tạo Coordinator bằng IP LAN thật, ví dụ `192.168.1.10`. Mở port 8000/8080 trên server và 8001 trên client trong mạng Private.
3. Tải sẵn BloodMNIST, OrganAMNIST và PathMNIST. Coordinator dùng Overview → Benchmark datasets → Prepare dataset; Institution dùng Dataset → Prepare dataset. Kiểm tra `data_root` trùng thư mục dataset của môi trường training.
4. Đăng nhập từng portal. Trong Overview, chạy Check environment. Jetson cần kiểm tra thêm một phiên huấn luyện thật; trạng thái import thành công chưa chứng minh CUDA/training hoạt động.
5. Coordinator tạo pairing token riêng cho từng máy. Trên Institution chọn Connect to coordinator rồi nhập URL và token. Token hết hạn sau 15 phút và dùng một lần.
6. Import checkpoint của một thí nghiệm đã huấn luyện đầy đủ. Chọn đúng dataset, architecture và thông tin checkpoint. Với FedBN/FedBNS, dùng checkpoint cục bộ được hệ thống lưu sau phiên training.
7. Nếu dùng mô hình global, vào Models trên Institution để tải và xác minh mô hình trước khi mất internet.

## 3. Kịch bản trình bày khoảng 8–10 phút

**Giới thiệu hệ thống:** “FedMedAI giúp các cơ sở cùng huấn luyện mô hình phân loại ảnh. Mỗi thiết bị thực hiện tính toán trên dữ liệu cục bộ. Server điều phối và tổng hợp cập nhật mô hình.”

**Màn hình Coordinator:** chỉ vào Institutions, tên máy, trạng thái Online, CPU/RAM và môi trường thực tế. Cho thấy có ba laptop client và một hoặc hai Nano nếu các thiết bị đã vượt qua kiểm tra.

**Màn hình Institution:** mở Dataset, giải thích số ảnh và phân bố nhãn được giao. MedMNIST là bộ benchmark công khai; đây là mô phỏng vai trò của các cơ sở, không phải dữ liệu bệnh nhân thực tế.

**Tạo phiên:** vào Training → New session. Chọn tập dữ liệu, thuật toán và roster. Để demo nhanh, dùng ba rounds và một local epoch; thời gian thực tế cần đo trước trên Nano. Partition dùng đúng số client đang chọn, khác với thí nghiệm 12-client đã chạy trên Kaggle.

**Chấp nhận và chạy:** từng Institution bấm Accept & check readiness. Coordinator chỉ bắt đầu được khi tất cả máy đã Ready. Mở biểu đồ accuracy và tiến độ client. Không phải mở terminal để thao tác trong bước demo này.

**Kết quả nghiên cứu:** vào Experiments để xem các kết quả đã import. Chỉ so sánh đúng dataset, model, partition và số client. BTA chọn round tốt nhất trên test nên dùng để khảo sát; kết quả SCR hiện tại chưa chứng minh cải thiện nhất quán.

**Dự đoán:** trên Institution, vào Models chọn checkpoint đã huấn luyện đầy đủ, sau đó Predict. Chọn mẫu benchmark hoặc ảnh patch phù hợp, nhấn Classify image. Giải thích nhãn dự đoán, ba model scores, thời gian inference và mã phiên bản mô hình. Các scores không phải độ tin cậy chẩn đoán đã được hiệu chuẩn.

## 4. Khi gặp sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| Không thấy client Online | Kiểm tra IP server, kết nối LAN, Firewall, pairing token và thông báo connection trên Institution. |
| Code / Flower không trùng | Sync đúng source; cài cùng phiên bản Flower đã kiểm thử trong môi trường training. |
| Dataset chưa có hoặc hash khác | Chuẩn bị lại đúng public benchmark file và kiểm tra `data_root`. |
| Client không Ready | Đọc lỗi của client. Chưa bấm Start cho đến khi đủ roster. |
| Client rớt mạng giữa phiên | Đợi timeout hoặc Cancel; sửa kết nối rồi tạo phiên mới. Không tiếp tục phiên bằng một roster khác. |
| Model chưa có | Import checkpoint hợp lệ hoặc hoàn thành training. Global checkpoint FedBN/FedBNS không đủ local BatchNorm. |
| Predict báo Busy | Đợi training/preparation trên máy đó kết thúc. |
| Nano không tương thích | Giữ trạng thái lỗi rõ ràng, kiểm tra môi trường riêng. Có thể trình bày pilot ba laptop với partition K=3 và nói rõ Nano chưa qua kiểm chứng. |

## 5. Nghiệm thu thiết bị thật

- [ ] Server laptop và ba laptop client hoạt động trên cùng LAN.
- [ ] Ít nhất một Nano hoàn thành preflight và một phiên Flower thật.
- [ ] Mọi client chấp nhận, cùng dataset/partition hash, đúng ID và đúng device type.
- [ ] Chạy thử năm thuật toán; kiểm tra checkpoint cục bộ FedBN/FedBNS.
- [ ] Dự đoán ảnh chạy trên Institution và không gửi ảnh qua Coordinator.
- [ ] Tắt internet sau khi chuẩn bị: website, dataset, model và training vẫn dùng được trên LAN.
- [ ] Kiểm thử mất kết nối, Cancel và restart; UI không báo thành công cho phiên lỗi.

Các kiểm thử local bằng nhiều process chứng minh luồng tích hợp phần mềm. Chỉ đánh dấu nghiệm thu LAN/Jetson sau khi chạy trên những thiết bị thật.

## 6. Theo dõi chi tiết quá trình học

Trong **Training**, chọn phiên rồi kéo xuống **Experiment monitor**:

- **Learning curves:** xem loss, accuracy, learning rate, gradient norm và độ thay đổi model. Với SCR, xem riêng weighted cross-entropy và head penalty. Chọn **Compare with another run** để đối chiếu hai phiên; chú ý cảnh báo nếu cấu hình hoặc partition không khớp.
- **Client diagnostics:** chọn client và round để tìm máy chạy chậm hoặc có kết quả kém; kiểm tra số mẫu và trọng số lớp của SCR.
- **Class performance:** xem confusion matrix, precision, recall, F1. Accuracy tổng cao vẫn có thể che giấu một lớp có recall thấp. Client không có ảnh test sẽ được bỏ qua, không xem là accuracy bằng 0.
- **System & timing:** theo dõi thời gian train/evaluate, CPU, RAM, bộ nhớ GPU của tiến trình và kích thước parameter arrays. Kích thước này không phải toàn bộ lưu lượng mạng.
- **Live logs:** lọc theo client, tìm lỗi/traceback và xem stdout/stderr thật. **CSV/JSON** dùng để phân tích hoặc lưu bằng chứng thí nghiệm.

BTA/BMCTA chọn round tốt nhất trên tập test nên chỉ dùng để khảo sát. Các metric mới chỉ có khi tiến trình đã ghi nhận chúng. Xem [MONITORING.md](MONITORING.md) để hiểu công thức, giới hạn, và lệnh restart an toàn sau khi phiên hiện tại kết thúc.
