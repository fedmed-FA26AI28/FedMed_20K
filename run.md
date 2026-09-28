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
| `--epochs` | `int` | `100` | Số lượng epoch huấn luyện tối đa. Quá trình có tích hợp Early Stopping (patience = 8 epochs) tự động dừng nếu validation loss không cải thiện. |
| `--batch_size` | `int` | `32` | Kích thước mini-batch cho DataLoader. Tăng lên `64` hoặc `128` nếu PC có GPU VRAM lớn (>= 6GB) để tăng tốc, hoặc giảm xuống `16` nếu chạy trên CPU yếu. |
| `--lr` hoặc `--learning_rate` | `float` | `0.001` | Tốc độ học ban đầu cho Adam optimizer. Bộ điều chỉnh `ReduceLROnPlateau` sẽ tự động giảm 50% lr khi val loss đi ngang 3 epochs. |

**Ví dụ lệnh kết hợp nhiều tùy chọn:**
```powershell
python -m experiments.train_centralized --epochs 50 --batch_size 64 --lr 0.0005
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

### 2.3. Khởi động các PC Client kết nối về PC Server

Giả sử IP của PC Server là `192.168.1.50`:

* **Trên PC Client 1 (Client ID = 0):**
  * Windows:
    ```powershell
    python -m client.client --client_id 0 --server_address 192.168.1.50:8080 --num_clients 2 --alpha 0.3
    ```
  * Linux:
    ```bash
    python3 -m client.client --client_id 0 --server_address 192.168.1.50:8080 --num_clients 2 --alpha 0.3
    ```

* **Trên PC Client 2 (Client ID = 1):**
  * Windows:
    ```powershell
    python -m client.client --client_id 1 --server_address 192.168.1.50:8080 --num_clients 2 --alpha 0.3
    ```
  * Linux:
    ```bash
    python3 -m client.client --client_id 1 --server_address 192.168.1.50:8080 --num_clients 2 --alpha 0.3
    ```

*(Nếu chạy 1 client ngay trên chính máy chủ, có thể dùng `--server_address 127.0.0.1:8080`).*

---

### 📝 Ghi chú: Danh sách tùy chọn cho FL trên PC

#### A. Tùy chọn cho Server (`server/server.py`)
| Tùy chọn (Option) | Kiểu dữ liệu | Mặc định | Ý nghĩa & Hướng dẫn sử dụng |
| :--- | :---: | :---: | :--- |
| `--strategy` | `str` | `fedavg` | Thuật toán tổng hợp trọng số: `fedavg`, `fedprox`, `fednova`. |
| `--rounds` | `int` | `50` | Tổng số vòng (rounds) Federated Learning cần huấn luyện. |
| `--min_clients` | `int` | `3` | Số lượng client tối thiểu phải kết nối trước khi Server bắt đầu mỗi round. **Phải khớp với số PC client tham gia**. |
| `--local_epochs` | `int` | `5` | Số epoch huấn luyện cục bộ tại mỗi client trong từng round. |
| `--learning_rate` | `float` | `0.001` | Tốc độ học mà Server chỉ thị cho tất cả clients sử dụng. |
| `--proximal_mu` | `float` | `0.1` | Hệ số phạt proximal $\mu$ khi dùng thuật toán `fedprox` (ràng buộc trọng số local không lệch quá xa global model). |
| `--early_stop_patience`| `int` | `10` | Dừng sớm FL nếu sau $N$ round accuracy đánh giá không tăng (`0` để tắt). |
| `--server_eval` | `flag` | `False` | Bật đánh giá tập trung mô hình toàn cục trên tập test sau mỗi round tại Server. |
| `--host` | `str` | `0.0.0.0` | Địa chỉ IP máy chủ lắng nghe (`0.0.0.0` để lắng nghe từ tất cả card mạng LAN). |
| `--port` | `int` | `8080` | Cổng mạng TCP gRPC giao tiếp giữa client và server. |

#### B. Tùy chọn cho Client (`client/client.py`)
| Tùy chọn (Option) | Kiểu dữ liệu | Mặc định | Ý nghĩa & Hướng dẫn sử dụng |
| :--- | :---: | :---: | :--- |
| `--client_id` | `int` | *(Bắt buộc)* | ID đại diện của máy client (0, 1, 2, ...). **Mỗi PC phải có ID riêng biệt**. |
| `--server_address` | `str` | `127.0.0.1:8080` | Địa chỉ `<IP_SERVER>:<PORT>` của PC Server trong mạng LAN. |
| `--num_clients` | `int` | `3` | Tổng số client tương ứng với file phân vùng dữ liệu. |
| `--alpha` | `float` | `0.3` | Hệ số Dirichlet $\alpha$ để tìm đúng file phân vùng (`0.1`, `0.3`, `1.0`). |
| `--partition_path` | `str` | `None` | Đường dẫn trực tiếp đến file JSON phân vùng (nếu không dùng đường dẫn mặc định). |
| `--local_epochs` | `int` | `5` | Số epoch train cục bộ (sẽ bị Server ghi đè nếu Server chỉ định). |
| `--learning_rate` | `float` | `0.001` | Tốc độ học ban đầu của client. |
| `--batch_size` | `int` | `None` | Kích thước batch. Mặc định tự đọc từ `configs/jetson.yaml` (PC: 32). Có thể truyền ví dụ `--batch_size 64` để ép buộc. |

**Ví dụ lệnh Server & Client trên PC:**
```powershell
# Server PC (chờ 3 clients, 30 rounds, chiến lược FedProx, đánh giá toàn cục):
python -m server.server --strategy fedprox --proximal_mu 0.05 --rounds 30 --min_clients 3 --server_eval

# Client PC (ID 0, kết nối tới 192.168.1.50, batch 32):
python -m client.client --client_id 0 --server_address 192.168.1.50:8080 --num_clients 3 --alpha 0.3 --batch_size 32
```

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
# Khởi động FedAvg cho 10 clients, 50 rounds:
python3 -m server.server --strategy fedavg --host 0.0.0.0 --port 8080 --min_clients 10 --rounds 50

# Hoặc FedProx để giảm phân kỳ do dữ liệu Non-IID trên 10 thiết bị:
python3 -m server.server --strategy fedprox --proximal_mu 0.1 --min_clients 10 --rounds 50

# Hoặc FedNova để xử lý vấn đề không đồng nhất phần cứng (Orin nhanh hơn Nano):
python3 -m server.server --strategy fednova --min_clients 10 --rounds 50
```

*(Mẹo: Dùng `tmux new -s fl_server` trên Linux Server để phiên chạy không bị gián đoạn).*

### 3.5. Khởi động Client trên từng máy Jetson

Giả sử IP máy chủ là `192.168.1.15:8080`:

* **Trên 5 máy Jetson Orin (Client 0 → 4):**
  ```bash
  # Orin 0:
  python3 -m client.client --client_id 0 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3

  # Orin 1:
  python3 -m client.client --client_id 1 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3

  # Orin 2:
  python3 -m client.client --client_id 2 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3

  # Orin 3:
  python3 -m client.client --client_id 3 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3

  # Orin 4:
  python3 -m client.client --client_id 4 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3
  ```

* **Trên 5 máy Jetson Nano (Client 5 → 9):**
  ```bash
  # Nano 5:
  python3 -m client.client --client_id 5 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3

  # Nano 6:
  python3 -m client.client --client_id 6 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3

  # Nano 7:
  python3 -m client.client --client_id 7 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3

  # Nano 8:
  python3 -m client.client --client_id 8 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3

  # Nano 9:
  python3 -m client.client --client_id 9 --server_address 192.168.1.15:8080 --num_clients 10 --alpha 0.3
  ```

### 3.6. Tự động hóa qua SSH Script (Chạy & Dừng toàn bộ cụm 10 Jetson từ xa)

Để không phải SSH thủ công vào 10 máy, tạo và chạy script điều khiển từ máy chủ Linux:

* **Kích hoạt đồng loạt cụm (`start_cluster.sh`):**
  ```bash
  #!/bin/bash
  SERVER_IP="192.168.1.15:8080"
  ORIN_IPS=("192.168.1.50" "192.168.1.51" "192.168.1.52" "192.168.1.53" "192.168.1.54")
  NANO_IPS=("192.168.1.60" "192.168.1.61" "192.168.1.62" "192.168.1.63" "192.168.1.64")

  echo "=== Kích hoạt 5 Jetson Orin ==="
  for i in {0..4}; do
    IP=${ORIN_IPS[$i]}
    echo "Khởi động Client $i trên Orin ($IP)..."
    ssh -n jetson@$IP "cd ~/FedMedAI && nohup python3 -m client.client --client_id $i --server_address $SERVER_IP --num_clients 10 --alpha 0.3 > ~/FedMedAI/client_$i.log 2>&1 &"
  done

  echo "=== Kích hoạt 5 Jetson Nano ==="
  for i in {0..4}; do
    CID=$((i + 5))
    IP=${NANO_IPS[$i]}
    echo "Khởi động Client $CID trên Nano ($IP)..."
    ssh -n nano@$IP "cd ~/FedMedAI && nohup python3 -m client.client --client_id $CID --server_address $SERVER_IP --num_clients 10 --alpha 0.3 > ~/FedMedAI/client_$CID.log 2>&1 &"
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
| `--num_clients` | `int` | `10` | Tổng số client trong cụm (mặc định cho cụm Jetson là 10). |
| `--alpha` | `float` | `0.3` | Tham số Dirichlet $\alpha$ của phân vùng (`0.1`, `0.3`, `1.0`). |
| `--partition_path` | `str` | `None` | Chỉ định file phân vùng cụ thể nếu lưu ở vị trí khác. |
| `--batch_size` | `int` | Tự động | Ghi đè kích thước batch nếu không muốn dùng giá trị tự động từ config. |
| `--learning_rate` | `float` | `0.001` | Tốc độ học khởi tạo. |
| `--local_epochs` | `int` | `5` | Số epoch huấn luyện cục bộ mỗi round. |

#### C. Lệnh hệ thống quản lý phần cứng Jetson
| Lệnh hệ thống | Tùy chọn / Cú pháp | Mục đích sử dụng |
| :--- | :--- | :--- |
| `nvpmodel` | `-m 0` (MAXN / High Power) | Đặt chế độ công suất tiêu thụ tối đa cho bo mạch Jetson. |
| `jetson_clocks` | *(không có cờ hoặc `--show`)* | Ép xung nhịp CPU, GPU và EMC memory controller chạy ở tần số trần cố định. |
| `tegrastats` | `--interval <ms> --logfile <path>` | Ghi log liên tục thông số phần cứng (RAM, VRAM, CPU load, công suất W, nhiệt độ °C). |
| `jtop` | *(chạy trong terminal)* | Trình giám sát tương tác trực quan thời gian thực của Jetson stats. |
| `pkill` | `-f <tên_tiến_trình>` | Dừng tiến trình client (`pkill -f 'client.client'`) hoặc tegrastats (`pkill tegrastats`). |
| `tmux` | `new -s <tên>` / `attach -t <tên>` | Quản lý terminal chạy ngầm, không bị tắt khi mất kết nối mạng. |
