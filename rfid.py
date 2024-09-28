import threading
import serial
import time
import re


class RFIDReader:
    no_reader = False

    def __init__(self, port="COM4", baudrate=38400, antenna=1):
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
            )
            self.detection_count = 0  # 検出カウント
            self.predicted_power = 0  # 予測電力
            self.antenna = antenna  # 使用するアンテナ番号
            self.running = False
            self.lock = threading.Lock()
        except Exception:
            self.no_reader = True

    def send_command(self, command):
        if self.no_reader:
            return ""
        try:
            self.ser.write(command.encode())
            time.sleep(0.01)
            response = self.ser.read(self.ser.in_waiting)
            return response.decode("ascii")
        except serial.SerialException as e:
            print(f"Serial exception: {e}")
            return ""

    def read_epc(self):
        """指定されたアンテナからEPCを読み取ります。"""
        command = "\x0AQ\x0D"
        response = self.send_command(command)
        response = re.sub(r"[\n\t\r]", "", response).lstrip("Q")
        if len(response) > 1:
            return response
        return None

    def predict_rf_power(self, detection_count):
        """検出回数に基づいて電波強度を百分率で推定します。"""
        max_detections = 16  # 1秒あたりの最大検出回数
        detection_ratio = min(detection_count / max_detections, 1.0)

        # 電波強度の範囲を仮定 (-100dBmから0dBm)
        min_power = -100
        max_power = 0

        predicted_power = min_power + detection_ratio * (max_power - min_power)

        # 電波強度を百分率に変換し、整数に丸める
        percentage = int((predicted_power - min_power) / (max_power - min_power) * 100)

        return percentage

    def check_antenna_status(self):
        """指定されたアンテナが接続されているか確認します。"""
        command = "\x0AN7,22\x0D"
        response = self.send_command(command)
        if response.strip():
            return True
        else:
            return False

    def start_reading(self):
        """RFIDリーディングを開始します。"""
        if not self.no_reader:
            self.running = True
            self.thread = threading.Thread(target=self.monitor_rfid)
            self.thread.daemon = True
            self.thread.start()

    def stop_reading(self):
        """RFIDリーディングを停止します。"""
        if not self.no_reader:
            self.running = False
            if self.thread.is_alive():
                self.thread.join()

    def monitor_rfid(self):
        """RFIDタグを監視し、検出回数に基づいて電波強度を予測します。"""
        if self.no_reader:
            return
        start_time = time.time()
        try:
            while self.running:
                epc = self.read_epc()
                if epc:
                    with self.lock:
                        self.detection_count += 1

                # 0.5秒ごとに検出頻度に基づいて電波強度を予測
                current_time = time.time()
                if current_time - start_time >= 0.5:
                    with self.lock:
                        self.predicted_power = self.predict_rf_power(
                            self.detection_count
                        )
                        self.detection_count = 0
                    start_time = current_time
                time.sleep(0.01)  # 短い間隔でループを継続
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Error in monitor_rfid: {e}")
        finally:
            self.ser.close()

    def get_predicted_power(self):
        """予測された電波強度を取得します。"""
        if self.no_reader:
            return 0
        with self.lock:
            return self.predicted_power

    def close(self):
        """RFIDリーディングを停止し、シリアルポートを閉じます。"""
        if not self.no_reader:
            self.stop_reading()
            self.ser.close()
