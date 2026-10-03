# FedMedAI — Hướng Dẫn Thực Thi (Execution & Run Guide)

Tài liệu này hướng dẫn chi tiết cách chạy hệ thống **FedMedAI** được chia thành 3 kịch bản chính:
1. **Centralized run on PC** (Huấn luyện tập trung mô hình trên 1 máy tính để lấy mốc chuẩn baseline).
2. **FL run on multiple PC** (Huấn luyện Federated Learning qua mạng LAN giữa nhiều máy tính PC / Laptop hoặc mô phỏng trên 1 PC).
3. **FL run on Jetson device with Linux** (Triển khai Federated Learning thực tế trên cụm thiết bị biên NVIDIA Jetson Orin & Jetson Nano chạy hệ điều hành Linux).

Sau mỗi phần đều có **mục ghi chú liệt kê toàn bộ các tùy chọn (CLI options)** có thể truyền vào khi chạy.

---

## 0. Thiết lập môi trường (Prerequisites)

Trước khi thực hiện các bước bên dưới, hãy kích hoạt môi trường ảo:

* **Windows PowerShell:**
  ```powershell
  # Kích hoạt venv (mặc định tại D:\FPT\KLTN\FED\.venv)
  D:\FPT\KLTN\FED\.venv\Scripts\Activate.ps1
  ```
* **Linux (Ubuntu PC / Server):**
  ```bash
  source .venv/bin/activate
  ```
* **Kiểm tra kiến trúc mô hình CNN (~20,104 tham số):**
  * Windows: `python -m models.cnn`
  * Linux: `python3 -m models.cnn`

---

## 1. Centralized Run on PC (Huấn luyện tập trung trên PC)

Kịch bản này huấn luyện mô hình CNN trên toàn bộ tập dữ liệu BloodMNIST (không phân chia client) để xác định độ chính xác chuẩn (upper-bound benchmark).

### 1.1. Chạy với cấu hình mặc định (100 epochs, batch size 32, lr 0.001)

* **Windows PowerShell:**
  ```powershell
  python -m experiments.train_centralized
  ```
* **Linux (Bash):**
  ```bash
  python3 -m experiments.train_centralized
  ```

### 1.2. Chạy với tùy chỉnh số epoch, batch size, learning rate

* **Windows PowerShell:**
  ```powershell
  # Ví dụ: Huấn luyện nhanh 30 epochs, batch size 64, learning rate 0.0005
  python -m experiments.train_centralized --epochs 30 --batch_size 64 --lr 0.0005
  ```
* **Linux (Bash):**
  ```bash
  python3 -m experiments.train_centralized --epochs 30 --batch_size 64 --lr 0.0005
  ```

### 1.3. Chạy nền trên Linux (Background Execution với `nohup`)

Tránh bị ngắt tiến trình khi đóng terminal hoặc mất kết nối SSH:
```bash
nohup python3 -m experiments.train_centralized --epochs 100 --batch_size 32 > centralized.log 2>&1 &

# Theo dõi tiến độ huấn luyện thời gian thực:
tail -f centralized.log
```

> 📁 **Kết quả đầu ra:** Lưu tại thư mục `results/centralized/<timestamp>/` gồm:
> - `best_model.pth`: Trọng số mô hình đạt accuracy cao nhất trên tập validation.
> - `training_curves.png`: Đồ thị Loss & Accuracy qua các epoch.
> - `confusion_matrix.png`: Ma trận nhầm lẫn trên tập test.
> - `per_class_metrics.png`: Biểu đồ Precision, Recall, F1-Score từng lớp tế bào.
> - `centralized_results.json`: Báo cáo chi tiết metrics và siêu tham số huấn luyện.
> - `summary_card.png`: Ảnh tổng hợp tóm tắt kết quả thí nghiệm.

---

### 📝 Ghi chú: Danh sách tùy chọn cho Centralized Run (`experiments/train_centralized.py`)

| Tùy chọn (Option) | Kiểu dữ liệu | Mặc định | Ý nghĩa & Hướng dẫn sử dụng |
| :--- | :---: | :---: | :--- |
| `--config` | `str` | `configs/experiment.yaml` | Đường dẫn file cấu hình YAML chứa các siêu tham số tập trung (mục `centralized:`). |
| `--epochs` | `int` | `100` (YAML) | Số lượng epoch huấn luyện tối đa. Quá trình có tích hợp Early Stopping tự động dừng nếu validation loss không cải thiện. |
| `--batch_size` | `int` | `32` (YAML) | Kích thước mini-batch cho DataLoader. Tăng lên `64` hoặc `128` nếu PC có GPU VRAM lớn (>= 6GB) để tăng tốc, hoặc giảm xuống `16` nếu chạy trên CPU yếu. |
| `--lr` hoặc `--learning_rate` | `float` | `0.001` (YAML) | Tốc độ học ban đầu cho Adam optimizer. |
| `--lr_patience` | `int` | `3` (YAML) | Số epochs val loss đi ngang trước khi `ReduceLROnPlateau` tự động giảm learning rate. |
| `--lr_factor` | `float` | `0.5` (YAML) | Hệ số nhân giảm learning rate khi gặp plateau (`new_lr = lr * factor`). |
| `--early_stop_patience` | `int` | `8` (YAML) | Số epochs val loss đi ngang trước khi Early Stopping dừng sớm quá trình huấn luyện. |
| `--min_lr` | `float` | `1e-6` (YAML) | Giới hạn dưới tối thiểu của learning rate. |

**Ví dụ lệnh kết hợp nhiều tùy chọn hoặc dùng file YAML khác:**
```powershell
python -m experiments.train_centralized --config configs/experiment.yaml --epochs 50 --batch_size 64 --lr 0.0005
```

---

## 2. FL Run on Multiple PC (Federated Learning trên nhiều PC trong mạng LAN)

Kịch bản này triển khai Federated Learning trên **nhiều máy tính PC / Laptop kết nối chung một mạng LAN** (Wi-Fi hoặc switch Ethernet), hoặc mô phỏng nhiều client chạy trên các terminal của cùng 1 PC.

Flower giao tiếp qua **gRPC trên socket TCP**, hoàn toàn tương thích chéo giữa Windows và Linux.

```
       ┌──────────────────────────────────────────────┐
       │   PC Server (e.g. Windows / Linux)           │
       │   IP LAN: 192.168.1.50 : 8080                │
       └───────┬──────────────────────────────┬───────┘
               │                              │
               ▼                              ▼
  ┌─────────────────────────┐    ┌─────────────────────────┐
  │   PC Client 0           │    │   PC Client 1           │
  │   IP LAN: 192.168.1.51  │    │   IP LAN: 192.168.1.52  │
  └─────────────────────────┘    └─────────────────────────┘
```

### 2.1. Bước chuẩn bị phân vùng dữ liệu (Data Partitioning)

Tạo file phân vùng Dirichlet Non-IID trên máy chủ (hoặc trên từng máy):

* **Tạo phân vùng cho 3 máy PC (ví dụ: $\alpha = 0.3$, seed 42):**
  * Windows: `python -m experiments.run_partition --num_clients 3 --alpha 0.3 --seed 42`
  * Linux: `python3 -m experiments.run_partition --num_clients 3 --alpha 0.3 --seed 42`

* **Sao chép file phân vùng:** Copy file vừa tạo trong `data/partitions/` sang các máy PC client (hoặc chạy lệnh trên với cùng `--seed 42` ở mỗi máy). Tập dữ liệu MedMNIST sẽ tự động tải về khi client chạy lần đầu.

### 2.2. Thiết lập trên PC làm Máy Chủ (FL Server)

1. **Tìm địa chỉ IP LAN của máy chủ:**
   * **Windows:** Mở PowerShell/CMD gõ `ipconfig` -> Tìm dòng `IPv4 Address` (ví dụ: `192.168.1.50`).
   * **Linux:** Mở Terminal gõ `hostname -I` (ví dụ: `192.168.1.50`).

2. **Mở cổng 8080 trên tường lửa (Firewall) máy chủ:**
   * **Windows (PowerShell Run as Administrator):**
     ```powershell
     New-NetFirewallRule -DisplayName "Flower FL Server 8080" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow
     ```
     *(Hoặc bấm Cho phép/Allow khi hộp thoại Windows Defender Firewall xuất hiện).*
   * **Linux:**
     ```bash
     sudo ufw allow 8080/tcp
     ```

3. **Khởi động FL Server trên PC Server:**
   * **Chiến lược FedAvg (Mặc định):**
     ```powershell
     # Chờ 2 clients kết nối qua cổng 8080, chạy 20 rounds:
     python -m server.server --strategy fedavg --rounds 20 --min_clients 2 --host 0.0.0.0 --port 8080
     ```
   * **Chiến lược FedProx (Chống phân kỳ do dữ liệu Non-IID giữa các PC):**
     ```powershell
     python -m server.server --strategy fedprox --proximal_mu 0.1 --rounds 20 --min_clients 2
     ```
   * **Chiến lược FedNova (Chuẩn hóa khi các PC có tốc độ tính toán chênh lệch):**
     ```powershell
     python -m server.server --strategy fednova --rounds 20 --min_clients 2
     ```
   * **Chiến lược FedBN (Bảo tồn Batch Normalization cục bộ chống Domain Shift):**
     ```powershell
     python -m server.server --strategy fedbn --rounds 20 --min_clients 2
     ```
   * **Chiến lược SCAFFOLD (Sử dụng biến đối ngẫu Control Variates chống Client Drift):**
     ```powershell
     python -m server.server --strategy scaffold --rounds 20 --min_clients 2
     ```

   * **Lựa chọn chế độ điều chỉnh Learning Rate (`--lr_mode`):**
     ```powershell
     # 1. client_loss (Mặc định): Server chỉ gửi initial LR ở Round 1; từng client tự giảm LR khi local loss đi ngang
     python -m server.server --strategy fedavg --lr_mode client_loss --learning_rate 0.001 --client_lr_patience 2 --client_lr_factor 0.5

     # 2. server_decay: Server đồng bộ step decay giảm đều LR của tất cả clients sau mỗi N rounds
     python -m server.server --strategy fedavg --lr_mode server_decay --learning_rate 0.001 --lr_decay_steps 10 --lr_decay_gamma 0.5

     # 3. fixed: Giữ nguyên tốc độ học cố định suốt toàn bộ các rounds (Flower baseline cổ điển)
     python -m server.server --strategy fedavg --lr_mode fixed --learning_rate 0.001
     ```

### 2.3. Khởi động các PC Client kết nối về PC Server

Giả sử IP của PC Server là `192.168.1.50`:

* **Trên PC Client 1 (Client ID = 0):**
  * Windows:
    ```powershell
    python -m client.client --client_id 0 --server_address 192.168.1.50:8080 --num_clients 2 --alpha 0.3 --strategy fedavg --lr_mode client_loss
    ```
  * Linux:
    ```bash
    python3 -m client.client --client_id 0 --server_address 192.168.1.50:8080 --num_clients 2 --alpha 0.3 --strategy fedavg --lr_mode client_loss
    ```

* **Trên PC Client 2 (Client ID = 1):**
  * Windows:
    ```powershell
    python -m client.client --client_id 1 --server_address 192.168.1.50:8080 --num_clients 2 --alpha 0.3 --strategy fedavg --lr_mode client_loss
    ```
  * Linux:
    ```bash
    python3 -m client.client --client_id 1 --server_address 192.168.1.50:8080 --num_clients 2 --alpha 0.3 --strategy fedavg --lr_mode client_loss
    ```

*(Nếu chạy 1 client ngay trên chính máy chủ, có thể dùng `--server_address 127.0.0.1:8080`).*

---

### 📝 Ghi chú: Danh sách tùy chọn cho FL trên PC

#### A. Tùy chọn cho Server (`server/server.py`)
| Tùy chọn (Option) | Kiểu dữ liệu | Mặc định | Ý nghĩa & Hướng dẫn sử dụng |
| :--- | :---: | :---: | :--- |
| `--strategy` | `str` | `fedavg` (YAML) | Thuật toán tổng hợp trọng số: `fedavg`, `fedprox`, `fednova`, `fedbn`, `scaffold`. |
| `--rounds` | `int` | `50` (YAML) | Tổng số vòng (rounds) Federated Learning cần huấn luyện. |
| `--min_clients` | `int` | `3` (YAML) | Số lượng client tối thiểu phải kết nối trước khi Server bắt đầu mỗi round. **Phải khớp với số PC client tham gia**. |
| `--local_epochs` | `int` | `5` (YAML) | Số epoch huấn luyện cục bộ tại mỗi client trong từng round. |
| `--lr_mode` | `str` | `client_loss` (YAML) | Chế độ quản trị Learning Rate: `client_loss` (mỗi client tự giảm trên loss cục bộ), `server_decay` (server giảm theo chu kỳ round), `fixed` (cố định). |
| `--learning_rate` | `float` | `0.001` (YAML) | Tốc độ học khởi tạo (initial base learning rate). |
| `--client_lr_patience`| `int` | `2` (YAML) | Số epoch local loss đi ngang trước khi client tự giảm LR (dùng khi `--lr_mode client_loss`). |
| `--client_lr_factor` | `float` | `0.5` (YAML) | Hệ số nhân giảm learning rate của client (`new_lr = lr * factor`). |
| `--client_lr_min` | `float` | `1e-6` (YAML) | Ngưỡng learning rate tối thiểu tại client. |
| `--lr_decay_steps` | `int` | `10` (YAML) | Chu kỳ số round để server giảm learning rate (dùng khi `--lr_mode server_decay`). |
| `--lr_decay_gamma` | `float` | `0.5` (YAML) | Hệ số nhân giảm learning rate của server ở mỗi chu kỳ (`--lr_mode server_decay`). |
| `--proximal_mu` | `float` | `0.1` (YAML) | Hệ số phạt proximal $\mu$ khi dùng thuật toán `fedprox` (ràng buộc trọng số local không lệch quá xa global model). |
| `--early_stop_patience`| `int` | `10` (YAML) | Dừng sớm FL nếu sau $N$ round accuracy đánh giá không tăng (`0` để tắt). |
| `--server_eval` | `flag` | `False` | Bật đánh giá tập trung mô hình toàn cục trên tập test sau mỗi round tại Server. |
| `--host` | `str` | `0.0.0.0` | Địa chỉ IP máy chủ lắng nghe (`0.0.0.0` để lắng nghe từ tất cả card mạng LAN). |
| `--port` | `int` | `8080` | Cổng mạng TCP gRPC giao tiếp giữa client và server. |

#### B. Tùy chọn cho Client (`client/client.py`)
| Tùy chọn (Option) | Kiểu dữ liệu | Mặc định | Ý nghĩa & Hướng dẫn sử dụng |
| :--- | :---: | :---: | :--- |
| `--client_id` | `int` | *(Bắt buộc)* | ID đại diện của máy client (0, 1, 2, ...). **Mỗi PC phải có ID riêng biệt**. |
| `--server_address` | `str` | `127.0.0.1:8080` | Địa chỉ `<IP_SERVER>:<PORT>` của PC Server trong mạng LAN. |
| `--num_clients` | `int` | `3` (YAML) | Tổng số client tương ứng với file phân vùng dữ liệu. |
| `--alpha` | `float` | `0.3` | Hệ số Dirichlet $\alpha$ để tìm đúng file phân vùng (`0.1`, `0.3`, `1.0`). |
| `--partition_path` | `str` | `None` | Đường dẫn trực tiếp đến file JSON phân vùng (nếu không dùng đường dẫn mặc định). |
| `--strategy` | `str` | `fedavg` (YAML) | Tên chiến lược để client kích hoạt cơ chế tương ứng (`scaffold`, `fedbn`, `fedprox`, `fednova`, `fedavg`). |
| `--local_epochs` | `int` | `5` (YAML) | Số epoch train cục bộ (sẽ bị Server ghi đè nếu Server chỉ định). |
| `--lr_mode` | `str` | `client_loss` (YAML) | Chế độ LR của client: `client_loss`, `server_decay`, `fixed`. |
| `--learning_rate` | `float` | `0.001` (YAML) | Tốc độ học ban đầu của client. |
| `--lr_patience` | `int` | `2` (YAML) | Số epoch local loss đi ngang trước khi client tự giảm LR (khi ở chế độ `client_loss`). |
| `--lr_factor` | `float` | `0.5` (YAML) | Hệ số nhân giảm learning rate của client. |
| `--lr_min` | `float` | `1e-6` (YAML) | Giới hạn learning rate tối thiểu của client. |
| `--batch_size` | `int` | `None` | Kích thước batch. Mặc định tự đọc từ `configs/jetson.yaml` (PC: 32). Có thể truyền ví dụ `--batch_size 64` để ép buộc. |
| `--device_type` | `str` | `None` | Ghi đè loại thiết bị / tên nhãn định danh (ví dụ: `pc`, `jetson_orin`, `jetson_nano`, `PC1`). Nếu không truyền, hệ thống sẽ tự động tra cứu từ `configs/jetson.yaml`. |
| `--no_save_metrics` | `flag` | `False` | Tắt tự động ghi log số liệu metrics per-round & per-epoch ra CSV cục bộ tại client. |

**Ví dụ lệnh Server & Client trên PC:**
```powershell
# Server PC (chờ 3 clients, 30 rounds, chiến lược SCAFFOLD, client_loss LR mode, đánh giá toàn cục):
python -m server.server --strategy scaffold --lr_mode client_loss --rounds 30 --min_clients 3 --server_eval

# Client PC (ID 0, kết nối tới 192.168.1.50, batch 32, strategy scaffold):
python -m client.client --client_id 0 --server_address 192.168.1.50:8080 --num_clients 3 --alpha 0.3 --strategy scaffold --batch_size 32
```


> 📁 **Kết quả đầu ra của FL Server (FL Run Artifacts):**  
> Tự động lưu tại thư mục `results/federated/<strategy>/<timestamp>/` gồm đầy đủ định dạng số liệu, đồ thị và báo cáo:
>
> 1. **Các file số liệu định dạng CSV (.csv):**
>    - `client_round_metrics.csv`: Chi tiết từng client qua từng round (round, client_id, device_type, num_samples, local_epochs, training_time_seconds, epoch_time_avg, min/max epoch time, train_loss, train_accuracy, weight_size_kb, cpu_percent, ram_percent, ram_used_mb, gpu_memory_mb).
>    - `client_epoch_metrics.csv`: Bóc tách chi tiết đến từng epoch của từng client (round, client_id, device_type, epoch, epoch_time_seconds, cumulative_epoch_time_seconds, train_loss, train_accuracy, learning_rate).
>    - `round_metrics.csv`: Tổng hợp cấp hệ thống cho từng round (round, round_time_seconds, total_elapsed_seconds, num_clients_reporting, train_loss_avg, train_accuracy_avg, eval_loss, eval_accuracy, client_train_time_avg, client_train_time_min, client_train_time_max, straggler_time_seconds, epoch_time_avg, total_weight_size_kb, server_overhead_seconds).
>
> 2. **Các biểu đồ trực quan so sánh (.png):**
>    - `client_training_time_comparison.png`: Đồ thị so sánh thời gian huấn luyện per-round giữa các client, biểu đồ cột thời gian trung bình kèm sai số, và biểu đồ độ trễ trễ hạn (straggler latency penalty = max_time - min_time).
>    - `client_epoch_time_comparison.png`: Biểu đồ hộp (Boxplot) phân phối thời gian tính toán mỗi epoch của từng client và biểu đồ so sánh thời gian epoch trung bình giữa các lớp phần cứng (PC vs Orin vs Nano).
>    - `fl_training_curves.png`: Đường cong hội tụ Loss & Accuracy qua các round (Weighted Client Train vs Global Centralized Test) và biểu đồ phân tán độ chính xác của từng client.
>    - `round_time_breakdown.png`: Biểu đồ cột xếp chồng phân rã thời gian mỗi round thành 3 thành phần: Thời gian tính toán client nhanh nhất, thời gian chờ do straggler, và overhead truyền thông mạng / tổng hợp server.
>    - `fl_summary_card.png`: Thẻ tóm tắt trực quan (Dashboard Summary Card) tổng kết siêu tham số, hiệu năng mô hình, tài nguyên và độ chênh lệch tính toán phần cứng.
>
> 3. **Báo cáo định dạng JSON chi tiết (.json):**
>    - `fl_results.json`: Chứa toàn bộ metadata, executive summary, thống kê tổng hợp của từng client, phân tích độ trễ phần cứng dị thể (hardware heterogeneity analysis), lịch sử huấn luyện round-by-round đầy đủ và đường dẫn đến các artifacts.
>
> 4. **Số liệu cục bộ lưu tại từng Client:**  
>    Tự động ghi tại `results/clients/client_<id>/` gồm:
>    - `client_<id>_round_metrics.csv`: Lịch sử thời gian và loss/acc của riêng client đó.
>    - `client_<id>_epoch_metrics.csv`: Thời gian thực thi từng epoch cục bộ.

---

## 3. FL Run on Jetson Device with Linux (Federated Learning trên cụm thiết bị NVIDIA Jetson)

Kịch bản này triển khai hệ thống Federated Learning thực tế trên cụm **10 thiết bị biên nhúng chạy Linux JetPack**:
- **5 thiết bị NVIDIA Jetson AGX Orin** (Client 0 → 4): Phần cứng mạnh (16-32GB RAM, 12-core ARM, GPU Ampere).
- **5 thiết bị NVIDIA Jetson Nano** (Client 5 → 9): Phần cứng giới hạn (4GB RAM chia sẻ CPU/GPU, 4-core ARM A57).
- **Máy chủ FL Server**: Có thể là Laptop/PC hoặc 1 node Server riêng trong mạng LAN.

```
                  ┌─────────────────────────────────────┐
                  │    FL Server (PC / Ubuntu Server)   │
                  │    IP: 192.168.1.15 : 8080          │
                  └──────────────────┬──────────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           │ (LAN 1 Gbps)                                      │
           ▼                                                   ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│  5x NVIDIA Jetson Orin        │               │  5x NVIDIA Jetson Nano        │
│  Clients 0, 1, 2, 3, 4        │               │  Clients 5, 6, 7, 8, 9        │
│  Batch 32 | 2 workers | MAXN  │               │  Batch 16 | 0 workers | 10W   │
└───────────────────────────────┘               └───────────────────────────────┘
```

### 3.1. Phân phối dữ liệu tới 10 thiết bị Jetson qua SCP

Từ máy tính chủ (đã có thư mục `data/partitions/`), phân phối phân vùng 10 clients:

* **Tự động gửi qua script:**
  * Windows: `python -m scripts.distribute_data --alpha 0.3 --num_clients 10`
  * Linux: `python3 -m scripts.distribute_data --alpha 0.3 --num_clients 10`
* **Hoặc gửi thủ công bằng `scp`:**
  ```bash
  # Gửi sang Orin 0:
  scp data/partitions/partition_seed42_alpha0.3_clients10.json jetson@192.168.1.50:~/FedMedAI/data/partitions/
  # Gửi sang Nano 5:
  scp data/partitions/partition_seed42_alpha0.3_clients10.json nano@192.168.1.60:~/FedMedAI/data/partitions/
  ```

### 3.2. Cấu hình phần cứng & Chế độ năng lượng trên Jetson (Bắt buộc)

Trước khi chạy, kích hoạt chế độ xung nhịp và năng lượng tối đa trên từng máy Jetson để đo lường độ trễ chuẩn xác:

* **Trên 5 máy Jetson Orin (Client 0 → 4):**
  ```bash
  sudo nvpmodel -m 0          # Chế độ MAXN (công suất tối đa)
  sudo jetson_clocks          # Khóa xung nhịp CPU và GPU ở mức cao nhất
  ```
* **Trên 5 máy Jetson Nano (Client 5 → 9):**
  ```bash
  sudo nvpmodel -m 0          # Chế độ 10W MAX (tránh thắt cổ chai 5W)
  sudo jetson_clocks          # Khóa xung nhịp tối đa
  ```

### 3.3. Giám sát tài nguyên phần cứng bằng `tegrastats` hoặc `jtop`

Thu thập dữ liệu tiêu thụ điện năng, mức chiếm dụng RAM, tải GPU/CPU trên Jetson:

```bash
# Ghi log tài nguyên mỗi 1 giây (1000ms) vào file trong quá trình train:
sudo tegrastats --interval 1000 --logfile ~/FedMedAI/tegrastats_client.log &

# Theo dõi trực quan dạng dashboard:
jtop

# Dừng ghi log tegrastats sau khi hoàn thành:
sudo pkill tegrastats
```

### 3.4. Khởi động FL Server (chờ 10 clients)

Trên máy chủ Server (ví dụ IP: `192.168.1.15`):

```bash
# 1. Khởi động FedAvg cho 10 clients, 50 rounds:
python3 -m server.server --strategy fedavg --host 0.0.0.0 --port 8080 --min_clients 10 --rounds 50

# 2. Hoặc FedProx để giảm phân kỳ do dữ liệu Non-IID trên 10 thiết bị:
python3 -m server.server --strategy fedprox --proximal_mu 0.1 --min_clients 10 --rounds 50

# 3. Hoặc FedNova để xử lý vấn đề không đồng nhất phần cứng (Orin nhanh hơn Nano):
python3 -m server.server --strategy fednova --min_clients 10 --rounds 50

# 4. Hoặc FedBN để bảo tồn Batch Normalization cục bộ chống Domain Shift:
python3 -m server.server --strategy fedbn --min_clients 10 --rounds 50

# 5. Hoặc SCAFFOLD với Control Variates để khắc phục Client Drift giữa các Jetson:
python3 -m server.server --strategy scaffold --min_clients 10 --rounds 50

# Tùy chỉnh chế độ Learning Rate trên Server:
# - Chế độ client_loss (mặc định): Từng Jetson tự điều chỉnh tốc độ học theo local loss
python3 -m server.server --strategy fedavg --lr_mode client_loss --learning_rate 0.001 --client_lr_patience 2

# - Chế độ server_decay: Giảm đều LR toàn cụm sau mỗi 10 rounds
python3 -m server.server --strategy fedavg --lr_mode server_decay --learning_rate 0.001 --lr_decay_steps 10 --lr_decay_gamma 0.5

# - Chế độ fixed: Giữ cố định LR suốt 50 rounds
python3 -m server.server --strategy fedavg --lr_mode fixed --learning_rate 0.001
```

*(Mẹo: Dùng `tmux new -s fl_server` trên Linux Server để phiên chạy không bị gián đoạn).*

### 3.5. Khởi động Client trên từng máy Jetson

Giả sử IP máy chủ là `192.168.1.15:8080`:

* **Trên 5 máy Jetson Orin (Client 0 → 4):**
  ```bash
  # Orin 0:
  python3 -m client.client --client_id 0 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3 --strategy fedavg --lr_mode client_loss

  # Orin 1:
  python3 -m client.client --client_id 1 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3 --strategy fedavg --lr_mode client_loss

  # Orin 2:
  python3 -m client.client --client_id 2 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3 --strategy fedavg --lr_mode client_loss

  # Orin 3:
  python3 -m client.client --client_id 3 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3 --strategy fedavg --lr_mode client_loss

  # Orin 4:
  python3 -m client.client --client_id 4 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3 --strategy fedavg --lr_mode client_loss
  ```

* **Trên 5 máy Jetson Nano (Client 5 → 9):**
  ```bash
  # Nano 5:
  python3 -m client.client --client_id 5 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3 --strategy fedavg --lr_mode client_loss

  # Nano 6:
  python3 -m client.client --client_id 6 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3 --strategy fedavg --lr_mode client_loss

  # Nano 7:
  python3 -m client.client --client_id 7 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3 --strategy fedavg --lr_mode client_loss

  # Nano 8:
  python3 -m client.client --client_id 8 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3 --strategy fedavg --lr_mode client_loss

  # Nano 9:
  python3 -m client.client --client_id 9 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3 --strategy fedavg --lr_mode client_loss
  ```

### 3.6. Tự động hóa qua SSH Script (Chạy & Dừng toàn bộ cụm 10 Jetson từ xa)

Để không phải SSH thủ công vào 10 máy, tạo và chạy script điều khiển từ máy chủ Linux:

* **Kích hoạt đồng loạt cụm (`start_cluster.sh`):**
  ```bash
  #!/bin/bash
  SERVER_IP="192.168.1.15:8080"
  STRATEGY="fedavg"
  LR_MODE="client_loss"
  ORIN_IPS=("192.168.1.50" "192.168.1.51" "192.168.1.52" "192.168.1.53" "192.168.1.54")
  NANO_IPS=("192.168.1.60" "192.168.1.61" "192.168.1.62" "192.168.1.63" "192.168.1.64")

  echo "=== Kích hoạt 5 Jetson Orin ==="
  for i in {0..4}; do
    IP=${ORIN_IPS[$i]}
    echo "Khởi động Client $i trên Orin ($IP)..."
    ssh -n jetson@$IP "cd ~/FedMedAI && nohup python3 -m client.client --client_id $i --server_address $SERVER_IP --num_clients 10 --alpha 0.3 --strategy $STRATEGY --lr_mode $LR_MODE > ~/FedMedAI/client_$i.log 2>&1 &"
  done

  echo "=== Kích hoạt 5 Jetson Nano ==="
  for i in {0..4}; do
    CID=$((i + 5))
    IP=${NANO_IPS[$i]}
    echo "Khởi động Client $CID trên Nano ($IP)..."
    ssh -n nano@$IP "cd ~/FedMedAI && nohup python3 -m client.client --client_id $CID --server_address $SERVER_IP --num_clients 10 --alpha 0.3 --strategy $STRATEGY --lr_mode $LR_MODE > ~/FedMedAI/client_$CID.log 2>&1 &"
  done
  echo "Tất cả 10 Jetson clients đã được kích hoạt!"
  ```

* **Dừng đồng loạt cụm (`stop_cluster.sh`):**
  ```bash
  for IP in "${ORIN_IPS[@]}"; do ssh -n jetson@$IP "pkill -f 'client.client'"; done
  for IP in "${NANO_IPS[@]}"; do ssh -n nano@$IP "pkill -f 'client.client'"; done
  echo "Đã dừng tất cả clients trên 10 Jetson."
  ```

---

### 📝 Ghi chú: Danh sách tùy chọn & Cấu hình phần cứng Jetson

#### A. Tự động nhận diện cấu hình phần cứng theo `client_id`
Hệ thống tự động tra cứu ID trong [`configs/jetson.yaml`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/configs/jetson.yaml) để chọn cấu hình phù hợp với RAM và CPU của từng bo mạch:

| Client ID | Thiết bị | Batch Size | DataLoader Workers | Pin Memory | Lý do kỹ thuật |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0 – 4** | **Jetson AGX Orin** | `32` | `2` | `False` | 12-core CPU đủ mạnh cho 2 workers; Unified Memory không cần pin. |
| **5 – 9** | **Jetson Nano** | `16` | `0` | `False` | RAM 4GB giới hạn nên giảm batch 16; workers = 0 (main process) tránh tràn bộ nhớ OOM. |

*(Nếu muốn ghi đè thủ công, bạn có thể truyền trực tiếp cờ `--batch_size <giá_trị>` trên dòng lệnh client).*

#### B. Danh sách tùy chọn lệnh Client trên Jetson (`client/client.py`)
| Tùy chọn (Option) | Kiểu dữ liệu | Mặc định | Ý nghĩa & Hướng dẫn sử dụng |
| :--- | :---: | :---: | :--- |
| `--client_id` | `int` | *(Bắt buộc)* | ID thiết bị: `0..4` cho Orin, `5..9` cho Nano. |
| `--server_address` | `str` | `127.0.0.1:8080` | Địa chỉ IP:Port của FL Server trong mạng LAN (ví dụ `192.168.1.15:8080`). |
| `--num_clients` | `int` | `10` (YAML) | Tổng số client trong cụm (mặc định cho cụm Jetson là 10). |
| `--alpha` | `float` | `0.3` | Tham số Dirichlet $\alpha$ của phân vùng (`0.1`, `0.3`, `1.0`). |
| `--partition_path` | `str` | `None` | Chỉ định file phân vùng cụ thể nếu lưu ở vị trí khác. |
| `--strategy` | `str` | `fedavg` (YAML) | Thuật toán client tham gia: `fedavg`, `fedprox`, `fednova`, `fedbn`, `scaffold`. |
| `--batch_size` | `int` | Tự động | Ghi đè kích thước batch nếu không muốn dùng giá trị tự động từ config. |
| `--lr_mode` | `str` | `client_loss` (YAML) | Chế độ quản lý learning rate: `client_loss`, `server_decay`, `fixed`. |
| `--learning_rate` | `float` | `0.001` (YAML) | Tốc độ học khởi tạo. |
| `--lr_patience` | `int` | `2` (YAML) | Số epoch local loss đi ngang trước khi client tự giảm LR (ở chế độ `client_loss`). |
| `--lr_factor` | `float` | `0.5` (YAML) | Hệ số nhân giảm learning rate của client. |
| `--lr_min` | `float` | `1e-6` (YAML) | Ngưỡng learning rate tối thiểu tại client. |
| `--local_epochs` | `int` | `5` (YAML) | Số epoch huấn luyện cục bộ mỗi round. |
| `--device_type` | `str` | `None` (Tự động) | Ép buộc lớp thiết bị: `pc`, `jetson_orin`, `jetson_nano`. |
| `--no_save_metrics` | `flag` | `False` | Tắt tự động ghi log số liệu metrics per-round & per-epoch ra CSV cục bộ tại client. |

#### C. Lệnh hệ thống quản lý phần cứng Jetson
| Lệnh hệ thống | Tùy chọn / Cú pháp | Mục đích sử dụng |
| :--- | :--- | :--- |
| `nvpmodel` | `-m 0` (MAXN / High Power) | Đặt chế độ công suất tiêu thụ tối đa cho bo mạch Jetson. |
| `jetson_clocks` | *(không có cờ hoặc `--show`)* | Ép xung nhịp CPU, GPU và EMC memory controller chạy ở tần số trần cố định. |
| `tegrastats` | `--interval <ms> --logfile <path>` | Ghi log liên tục thông số phần cứng (RAM, VRAM, CPU load, công suất W, nhiệt độ °C). |
| `jtop` | *(chạy trong terminal)* | Trình giám sát tương tác trực quan thời gian thực của Jetson stats. |
| `pkill` | `-f <tên_tiến_trình>` | Dừng tiến trình client (`pkill -f 'client.client'`) hoặc tegrastats (`pkill tegrastats`). |
| `tmux` | `new -s <tên>` / `attach -t <tên>` | Quản lý terminal chạy ngầm, không bị tắt khi mất kết nối mạng. |

---

## 4. Automated Multi-Strategy & Multi-Partition Simulation Benchmark (Mô phỏng tự động Benchmark toàn diện trên PC)

Kịch bản này cung cấp công cụ tự động hóa **100% cục bộ trên 1 máy tính PC** ([`experiments/run_simulation.py`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/experiments/run_simulation.py)) để chạy quét toàn bộ lưới ma trận (Grid Search Benchmark) giữa:
- **Tất cả các thuật toán FL:** `FedAvg`, `FedProx`, `FedNova`, `FedBN`, `SCAFFOLD`.
- **Tất cả các mức độ dị thể dữ liệu Non-IID:** Dirichlet $\alpha \in \{1.0, 0.3, 0.1\}$.
- **File cấu hình độc lập chuyên dụng:** [`configs/simulation.yaml`](file:///D:/FPT/KLTN/FED/FedMedAI_Study/configs/simulation.yaml).

> 💡 **Đặc tính kỹ thuật nổi bật:**
> 1. **100% Localhost Execution:** Tự động tạo Server và số lượng Client mong muốn (`num_clients: 3`, `5`, `10`...) dưới dạng các tiến trình con độc lập chạy qua loopback `127.0.0.1`. Không cần thiết bị Jetson ngoài và không cần kết nối mạng bên ngoài.
> 2. **Tự động cấp phát cổng động (Dynamic Ephemeral Port):** Mỗi lượt chạy được cấp một socket port trống riêng biệt, loại bỏ hoàn toàn lỗi xung đột cổng (`Address already in use`).
> 3. **Tự động kiểm tra & sinh phân hoạch Dirichlet:** Nếu thiếu file phân hoạch `partition_seed42_alpha{alpha}_clients{num_clients}.json`, hệ thống tự động phân chia tập BloodMNIST và vẽ biểu đồ phân bố lưu vào `data/partitions/`.
> 4. **Giám sát đa tiến trình chủ động (Watchdog Monitor):** Theo dõi liên tục cả Server và toàn bộ Client; nếu có lỗi ở bất kỳ client nào, hệ thống lập tức trích xuất log lỗi và thu hồi tiến trình, tránh bị treo vô tận.
> 5. **Tự động tổng hợp kết quả & biểu đồ so sánh xuất bản:** Xuất 5 biểu đồ so sánh chất lượng cao (300 DPI), 2 file CSV tổng hợp, file JSON benchmark và báo cáo tóm tắt Markdown.

---

### 4.1. Chạy mô phỏng với cấu hình mặc định trong `configs/simulation.yaml`

Lệnh này sẽ tự động duyệt toàn bộ ma trận (5 thuật toán $\times$ 3 mức độ $\alpha$ = 15 lượt chạy):

* **Windows PowerShell:**
  ```powershell
  python -m experiments.run_simulation
  ```
* **Linux (Bash):**
  ```bash
  python3 -m experiments.run_simulation
  ```

---

### 4.2. Chạy với file cấu hình tùy biến hoặc ghi đè tham số dòng lệnh (CLI)

Bạn có thể dễ dàng chạy thử nhanh một tập con các chiến lược, số round, hoặc số client:

* **Ví dụ 1: Chạy kiểm thử nhanh (Fast Verification) với 1 round, 2 clients, FedAvg:**
  ```powershell
  python -m experiments.run_simulation --rounds 1 --local_epochs 1 --num_clients 2 --batch_size 128 --strategies fedavg --alphas 1.0
  ```

* **Ví dụ 2: So sánh FedAvg vs FedProx trên dữ liệu dị thể trung bình và cao ($\alpha = 0.3, 0.1$):**
  ```powershell
  python -m experiments.run_simulation --rounds 20 --num_clients 5 --strategies fedavg fedprox --alphas 0.3 0.1
  ```

* **Ví dụ 3: Đổi chế độ quản lý Learning Rate sang `server_decay`:**
  ```powershell
  python -m experiments.run_simulation --rounds 15 --lr_mode server_decay --strategies fedavg fednova scaffold --alphas 0.3
  ```

---

### 4.3. Cấu trúc kết quả đầu ra của Simulation Benchmark

Toàn bộ kết quả được lưu tập trung vào thư mục `results/simulation/<timestamp>/`:

```
results/simulation/<timestamp>/
├── simulation_summary.csv             # Bảng CSV tổng hợp metrics cốt lõi của tất cả các lần chạy
├── simulation_rounds_history.csv     # Bảng CSV toàn bộ timeline round-by-round (accuracy, loss, time)
├── simulation_results.json           # File JSON đầy đủ siêu tham số và kết quả máy đọc
├── simulation_report.md              # Báo cáo Markdown có bảng xếp hạng Leaderboard trực quan
├── compare_accuracy_curves.png       # Đồ thị hội tụ Accuracy qua các round (chia theo từng alpha)
├── compare_loss_curves.png           # Đồ thị hội tụ Loss qua các round (chia theo từng alpha)
├── compare_accuracy_barchart.png     # Biểu đồ cột so sánh Best Test Accuracy của các thuật toán
├── compare_runtime_barchart.png      # Biểu đồ cột so sánh tổng thời gian huấn luyện và thời gian/round
├── compare_accuracy_heatmap.png      # Ma trận Heatmap 2D (Strategy x Alpha) làm nổi bật thuật toán tối ưu
└── alpha_<alpha>/                    # Thư mục chi tiết từng lần chạy
    └── <strategy>/                   # Artifacts chi tiết: fl_results.json, round_metrics.csv,
        ├── clients/                  # Log và metrics của từng client cục bộ
        └── best_model.pth            # Trọng số tốt nhất đạt được của chiến lược đó
```

---

### 📝 Ghi chú: Danh sách tùy chọn cấu hình Simulation (`configs/simulation.yaml` & CLI)

| Tùy chọn CLI | Khóa trong `configs/simulation.yaml` | Kiểu dữ liệu | Mặc định | Ý nghĩa & Hướng dẫn sử dụng |
| :--- | :--- | :---: | :---: | :--- |
| `--config` | *(đường dẫn)* | `str` | `configs/simulation.yaml` | Đường dẫn file YAML cấu hình mô phỏng. |
| `--num_clients` | `num_clients` | `int` | `3` | Số lượng client ảo mô phỏng cùng chạy cục bộ trên PC. |
| `--alphas` | `alphas` | `list[float]` | `[1.0, 0.3, 0.1]` | Danh sách mức độ phân tán nhãn Dirichlet $\alpha$ cần kiểm nghiệm. |
| `--strategies` | `strategies` | `list[str]` | `[fedavg, fedprox, fednova, fedbn, scaffold]` | Danh sách các thuật toán tổng hợp cần chạy benchmark. |
| `--rounds` | `rounds` | `int` | `10` | Số vòng giao tiếp FL (rounds) cho mỗi lượt chạy. |
| `--local_epochs` | `local_epochs` | `int` | `3` | Số epoch huấn luyện cục bộ tại mỗi client trong 1 round. |
| `--batch_size` | `batch_size` | `int` | `32` | Kích thước mini-batch của DataLoader trên client. |
| `--num_workers` | `num_workers` | `int` | `0` | Số tiến trình con nạp dữ liệu cho DataLoader (mặc định an toàn trên Windows là `0` để tránh lỗi IPC/multiprocessing). |
| `--lr` hoặc `--learning_rate` | `learning_rate` | `float` | `0.001` | Tốc độ học khởi tạo cho Optimizer. |
| `--lr_mode` | `lr_mode` | `str` | `client_loss` | Chế độ quản lý LR: `client_loss` (tự giảm theo loss cục bộ), `server_decay` (giảm toàn cục theo chu kỳ round), hoặc `fixed` (cố định). |
| *(YAML)* | `client_lr_patience` | `int` | `2` | Số epoch loss đi ngang trước khi client tự giảm LR (chế độ `client_loss`). |
| *(YAML)* | `client_lr_factor` | `float` | `0.5` | Hệ số nhân giảm LR của client (`new_lr = lr * factor`). |
| *(YAML)* | `lr_decay_steps` | `int` | `3` | Số round giữa mỗi lần giảm LR toàn cục (chế độ `server_decay`). |
| *(YAML)* | `proximal_mu` | `float` | `0.01` | Trọng số ràng buộc proximal $\mu$ cho thuật toán FedProx. |
| `--server_eval` | `server_eval` | `bool` | `True` | Đánh giá mô hình toàn cục sau mỗi round trên 3,421 ảnh test set BloodMNIST. |
| `--output_dir` | `output_dir` | `str` | `results/simulation/<timestamp>` | Thư mục lưu trữ kết quả và các biểu đồ so sánh. |
| `--timeout_per_run` | `timeout_per_run` | `int` | `1800` | Thời gian chờ tối đa (giây) cho 1 lượt chạy trước khi tự hủy để tránh treo. |

