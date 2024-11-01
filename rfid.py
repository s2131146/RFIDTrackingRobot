import threading
import serial
import time
import re
import logger
from constants import Commands

RFID_CARD = "Q3000E280699500006003A248807119C3"
RFID_ACTIVE = "Q3000E28068940000501AC621013A59D0"


class RFIDReader:
    # アンテナIDと名前のマッピングをクラス属性として定義
    ANTENNA_NAMES = {1: "CENTER", 2: "LEFT", 3: "RIGHT", 4: "REAR"}

    def __init__(self, port="COM4", baudrate=38400, antenna=1):
        self.port = port
        self.baudrate = baudrate
        self.current_antenna = antenna
        self.ser = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        self.detection_counts = {1: 0, 2: 0, 3: 0, 4: 0}  # REAR, LEFT, RIGHT, CENTER
        self.detection_counts_sec = {1: 0, 2: 0, 3: 0, 4: 0}
        self.predict_direction = Commands.GO_CENTER  # 初期方向設定
        self.last_direction = Commands.STOP_TEMP  # 前回の方向
        self.prevent_transition = True  # 方向転換を一時的に防ぐフラグ
        self.last_move_direction = Commands.GO_LEFT

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
            )
            logger.logger.info("RFID Reader initialized.")
        except Exception as e:
            logger.logger.error(f"Failed to initialize RFID Reader: {e}")
            self.no_reader = True

    def check_antenna_status(self):
        antenna_status = [False, False, False, False]

        for antenna_id in range(1, 5):
            self.switch_antenna(antenna_id)
            response = self.send_command("Q")

            if response and "Q" in response:
                antenna_status[antenna_id - 1] = True

        return antenna_status

    def send_command(self, command, skip_tag=False):
        """シリアル通信でコマンドを送信し、応答を受信"""
        if not hasattr(self, "no_reader") or not self.no_reader:
            try:
                if not skip_tag:
                    full_command = f"\x0A{command}\x0D"
                else:
                    full_command = command
                self.ser.write(full_command.encode())
                time.sleep(0.05)  # 0.016秒のスリープ
                response = self.ser.read_all().decode(errors="ignore")
                response = re.sub(r"[\n\t\r]", "", response)
                return response
            except serial.SerialException as e:
                logger.logger.error(f"Serial exception: {e}")
                return ""
        else:
            return ""

    def switch_antenna(self, antenna_id):
        """アンテナを切り替えるためのコマンドを送信"""
        if antenna_id in [1, 2]:
            command = "N9,20"
        elif antenna_id in [3, 4]:
            command = "N9,22"

        if antenna_id in [1, 4]:
            command2 = "N9,10"
        elif antenna_id in [2, 3]:
            command2 = "N9,11"

        self.send_command(command)
        self.send_command(command2)
        with self.lock:
            self.current_antenna = antenna_id

    def antenna_to_direction(self, counts):
        """アンテナ検出回数に基づいて移動方向を決定します。"""
        rear = counts.get(1, 0)
        left = counts.get(2, 0)
        right = counts.get(3, 0)
        center = counts.get(4, 0)

        # 検出回数が3以上のアンテナをリストアップ
        antennas_with_ge1 = [ant for ant, cnt in counts.items() if cnt >= 1]
        total_ge1 = len(antennas_with_ge1)

        # 各アンテナの最大検出回数とそれを持つアンテナ
        max_count = max(counts.values(), default=0)
        antennas_with_max = [ant for ant, cnt in counts.items() if cnt == max_count]

        # 5. 3つ以上のアンテナが1回以上検出された場合
        if total_ge1 >= 3:
            return Commands.STOP_TEMP

        # 1. すべてのアンテナが0の場合
        if rear == 0 and left == 0 and right == 0 and center == 0:
            return Commands.STOP_TEMP

        # 2. 一つのアンテナのみが検出され、他が0の場合
        non_zero_counts = [cnt for cnt in counts.values() if cnt > 0]
        if len(non_zero_counts) == 1:
            antenna = next(ant for ant, cnt in counts.items() if cnt > 0)
            direction = self.ANTENNA_NAMES[antenna]
            return self.direction_to_command(direction)

        # 3. CENTER、REARとLEFT、RIGHTの検出回数が同じ場合
        if center == rear and left == right and center > 0 and left > 0:
            return Commands.STOP_TEMP

        # 4. LEFTとRIGHTの検出回数が異なる場合、検出回数が多い方の方向に設定
        if left != right:
            if left > right:
                return Commands.GO_LEFT
            else:
                return Commands.GO_RIGHT

        if center >= 3 and rear >= 3:
            if center > rear:
                return Commands.GO_CENTER
            else:
                return Commands.GO_BACK
        elif left >= 3 and right >= 3:
            if left > right:
                return Commands.GO_LEFT
            else:
                return Commands.GO_RIGHT

        # 7. 最も検出回数が多い方向へ移動。ただし、各方向が3回以上の場合は前へ
        if max_count > 0 and total_ge1 == 0:
            antenna = antennas_with_max[0]
            direction = self.ANTENNA_NAMES[antenna]
            return self.direction_to_command(direction)

        return Commands.STOP_TEMP

    def direction_to_command(self, direction):
        """方向名からコマンドに変換"""
        mapping = {
            "REAR": Commands.ROTATE_LEFT,
            "LEFT": Commands.GO_LEFT,
            "RIGHT": Commands.GO_RIGHT,
            "CENTER": Commands.GO_CENTER,
        }
        return mapping.get(direction, Commands.STOP_TEMP)

    def read_epc(self):
        """EPCを読み取る"""
        epc_command = "Q"  # EPC読み取りコマンド
        response = self.send_command(epc_command)

        if RFID_CARD in response:
            with self.lock:
                self.detection_counts[self.current_antenna] += 1

    def monitor_rfid(self):
        """RFIDタグを監視し、検出回数に基づいて方向を予測します。"""
        try:
            while self.running:
                # 1. CENTER、LEFT、RIGHTアンテナでEPC読み取り
                for antenna_id in [1, 2, 3]:
                    self.switch_antenna(antenna_id)
                    for _ in range(5):
                        self.read_epc()

                # 2. 他のアンテナで一度も検出されなかった場合にのみREARアンテナを読み取る
                with self.lock:
                    counts_copy = self.detection_counts.copy()
                    detected_in_other_antennas = any(
                        counts_copy[ant] > 0 for ant in [1, 2, 3]
                    )

                if not detected_in_other_antennas:
                    # REARアンテナでEPC読み取り
                    self.switch_antenna(4)
                    for _ in range(5):
                        self.read_epc()

                with self.lock:
                    counts_copy = self.detection_counts.copy()
                    self.detection_counts_sec = counts_copy
                    new_direction = self.antenna_to_direction(counts_copy)
                    if (
                        (
                            new_direction == Commands.GO_LEFT
                            and self.predict_direction == Commands.GO_RIGHT
                        )
                        or (
                            new_direction == Commands.GO_RIGHT
                            and self.predict_direction == Commands.GO_LEFT
                        )
                        or (
                            new_direction == Commands.GO_CENTER
                            and self.predict_direction == Commands.GO_BACK
                        )
                        or new_direction == Commands.STOP_TEMP
                    ):
                        if self.prevent_transition:
                            self.predict_direction = new_direction
                            self.prevent_transition = False
                        else:
                            self.prevent_transition = True
                    else:
                        self.prevent_transition = False
                        self.predict_direction = new_direction

                    if (
                        self.predict_direction == Commands.GO_LEFT
                        or self.predict_direction == Commands.GO_RIGHT
                    ):
                        self.last_move_direction = self.predict_direction
                    self.last_direction = new_direction
                    for key in self.detection_counts:
                        self.detection_counts[key] = 0

                # 方向を直接設定
                with self.lock:
                    # アンテナ名と検出回数を文字列に変換
                    counts_str = ", ".join(
                        [
                            f"{self.ANTENNA_NAMES[ant]}={cnt}"
                            for ant, cnt in counts_copy.items()
                        ]
                    )
                    # 方向と検出回数をログに記録
                    logger.logger.info(
                        f"Direction: {self.predict_direction}, Counts: {counts_str}"
                    )

        except Exception as e:
            logger.logger.error(f"Error in monitor_rfid: {e}")

    def start_reading(self):
        """RFIDリーディングを開始します。"""
        if not hasattr(self, "no_reader") or not self.no_reader:
            self.running = True
            self.thread = threading.Thread(target=self.monitor_rfid)
            self.thread.daemon = True
            self.thread.start()
            logger.logger.info("Started RFID monitoring thread.")

    def stop_reading(self):
        """RFIDリーディングを停止します。"""
        if not hasattr(self, "no_reader") or not self.no_reader:
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join()
            logger.logger.info("Stopped RFID monitoring thread.")

    def get_detection_counts(self):
        """各アンテナの検出回数を取得します。"""
        with self.lock:
            return self.detection_counts_sec.copy()

    def get_rotate_direction(self):
        with self.lock:
            return self.last_move_direction

    def get_direction(self):
        """現在の進行方向を取得します。"""
        with self.lock:
            return self.predict_direction

    def close(self):
        """RFIDリーディングを停止し、シリアルポートを閉じます。"""
        self.stop_reading()
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.logger.info("Closed serial connection.")


def main():
    reader = RFIDReader(port="COM4", baudrate=38400)
    reader.start_reading()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        reader.close()
        logger.logger.info("RFID Reader stopped.")


if __name__ == "__main__":
    main()
