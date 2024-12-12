from enum import Enum
import threading
import serial
import time
import re
import logger
from constants import Commands, Position
from timer import timer
from obstacles import Obstacles


class RFIDAntenna(Enum):
    CENTER = 1
    RIGHT = 2
    REAR = 3
    LEFT = 4


RFID_CARD = "Q3000E280699500006003A248807119C3"
RFID_ACTIVE = "Q3000E28068940000501AC621013A59D0"

# アンテナIDと名前のマッピングをクラス属性として定義
ANTENNA_MAPS = {antenna.value: antenna.name for antenna in RFIDAntenna}
ANTENNA_NAMES = [ANTENNA_MAPS.get(i, "") for i in ANTENNA_MAPS]

SHOW_DEBUG = False


class RFIDReader:

    def __init__(self, obs=None, port="COM4", baudrate=38400, antenna=1):
        self.port = port
        self.baudrate = baudrate
        self.current_antenna = antenna
        self.ser = None
        self.running = False
        self.thread = None
        self.timer = timer()
        self.lock = threading.Lock()
        self.obs: Obstacles = obs

        self.detection_counts = {1: 0, 2: 0, 3: 0, 4: 0}  # REAR, LEFT, RIGHT, CENTER
        self.detection_counts_sec = {1: 0, 2: 0, 3: 0, 4: 0}
        self.predict_direction = Commands.GO_CENTER  # 初期方向設定
        self.last_direction = Commands.STOP_TEMP  # 前回の方向
        self.last_move_direction = Commands.GO_LEFT

        self.signal_strength = 40  # 初期値を40に設定
        self.rear_detection_streak = 0
        self.no_reader = False

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
            )
            self.setup_command()
            logger.logger.info("RFID Reader initialized.")
        except Exception as e:
            logger.logger.error(f"Failed to initialize RFID Reader: {e}")
            self.no_reader = True

    def setup_command(self):
        self.send_command("N1,26")  # アンテナの電波強度設定
        r1 = self.send_command("J000")  # Region設定
        r2 = self.send_command("N5,05")  # EU 865~868

        logger.logger.info(f"RFID Reader Settings: {r1}, {r2}")

    def set_signal_strength(self, strength):
        """電波の強さを設定"""
        if 18 <= strength <= 40 and strength != self.signal_strength:
            command_value = hex(strength - 2).upper()[2:]
            command = f"N1,{command_value}"
            self.send_command(command)
            self.signal_strength = strength
            time.sleep(0.3)
            logger.logger.info("Set RFID Antennas signal strengh to %d", strength)

    def adjust_signal_strength(self):
        """検出回数に基づいて電波の強さを動的に変更"""
        with self.lock:
            # 最も検出されたアンテナの回数と数を取得
            max_detection = max(self.detection_counts_sec.values(), default=0)
            num_with_4_or_more = sum(
                1 for count in self.detection_counts_sec.values() if count >= 4
            )

        # 条件に基づいて目標電波強度を設定
        if max_detection == 0:
            target_strength = 40
        elif (
            self.detection_counts_sec[RFIDAntenna.CENTER.value] > 1
            and self.detection_counts_sec[RFIDAntenna.REAR.value] > 1
        ) or (
            self.detection_counts_sec[RFIDAntenna.LEFT.value] > 1
            and self.detection_counts_sec[RFIDAntenna.RIGHT.value] > 1
        ):
            target_strength = self.signal_strength - 4  # 強度を下げる
        elif max_detection >= 4 and num_with_4_or_more <= 2:
            target_strength = self.signal_strength  # 強度を変えない
        elif max_detection >= 4 and num_with_4_or_more > 2:
            target_strength = self.signal_strength - 4  # 強度を下げる
        elif max_detection <= 3:
            target_strength = self.signal_strength + 6  # 強度を上げる
        else:
            target_strength = self.signal_strength  # 何もしない

        # 電波強度を変更
        self.set_signal_strength(target_strength)

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
                time.sleep(
                    0.016
                )  # 最小0.016秒、ただし正常に取れない可能性あり 安定は0.05
                response = self.ser.read_all().decode(errors="ignore")
                if SHOW_DEBUG:
                    logger.logger.debug(
                        f"Send: {repr(full_command)}, Response: {repr(response)}"
                    )
                response = re.sub(r"[\n\t\r]", "", response)

                if response == "":
                    return self.send_command(command, skip_tag)
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

        self.send_command("N7,22")
        self.send_command("N7,11")
        self.send_command(command)
        self.send_command(command2)

        with self.lock:
            self.current_antenna = antenna_id

    prev_lr = Commands.GO_LEFT

    def antenna_to_direction(self, counts):
        """アンテナ検出回数に基づいて移動方向を決定します。"""
        rear = counts.get(RFIDAntenna.REAR.value, 0)
        left = counts.get(RFIDAntenna.LEFT.value, 0)
        right = counts.get(RFIDAntenna.RIGHT.value, 0)
        center = counts.get(RFIDAntenna.CENTER.value, 0)

        non_zero_counts = [cnt for cnt in counts.values() if cnt > 0]
        only_one = len(non_zero_counts) == 1

        if self.obs.any_left() and not only_one:
            left = max(left - 3, 0)
        if self.obs.any_right() and not only_one:
            right = max(right - 3, 0)

        if not (rear >= 3 and center < 1 and left < 2 and right < 2):
            rear = 0

        self.detection_counts_sec = {1: center, 2: right, 3: rear, 4: left}
        counts = self.detection_counts_sec.copy()

        antennas_with_ge3 = [ant for ant, cnt in counts.items() if cnt >= 3]
        total_ge3 = len(antennas_with_ge3)

        # 3回以上のアンテナが3つ以上で停止
        if total_ge3 >= 3:
            logger.logger.info("RFID Dir: Total ge 3")
            return Commands.STOP_TEMP

        # 各アンテナの最大検出回数とそれを持つアンテナ
        max_count = max(counts.values(), default=0)
        antennas_with_max = [ant for ant, cnt in counts.items() if cnt == max_count]

        if rear > 0:
            self.rear_detection_streak += 1
        else:
            self.rear_detection_streak = 0

        if self.rear_detection_streak >= 2:
            logger.logger.info("RFID Dir: REAR rotate")
            return Position.convert_to_rotate(self.prev_lr)
        else:
            rear = 0

        # 1. すべてのアンテナが0の場合
        if rear == 0 and left == 0 and right == 0 and center == 0:
            logger.logger.info("RFID Dir: All zero")
            return Commands.STOP_TEMP

        # 2. 一つのアンテナのみが検出され、他が0の場合
        if only_one:
            antenna = next(ant for ant, cnt in counts.items() if cnt > 0)
            direction = ANTENNA_MAPS[antenna]
            logger.logger.info("RFID Dir: Only one antenna detected")
            dir = self.direction_to_command(direction)
            if max_count >= 4:
                dir = Position.convert_to_rotate(dir)
            return dir

        # 3. CENTER、REARとLEFT、RIGHTの検出回数が同じ場合
        if center == rear and left == right and center > 0 and left > 0:
            logger.logger.info("RFID Dir: Same diagonal")
            return self.last_direction

        # 斜めの場合
        if center > 0 and (left >= 1 or right >= 1):
            logger.logger.info("RFID Dir: Diagonal")
            if left > right:
                return Commands.GO_LEFT
            elif right > left:
                return Commands.GO_RIGHT
            else:
                return Commands.GO_CENTER

        # 7. 最も検出回数が多い方向へ移動。ただし、各方向が3回以上の場合は前へ
        if max_count > 0:
            logger.logger.info("RFID Dir: Max detected")
            antenna = antennas_with_max[0]
            direction = ANTENNA_MAPS[antenna]
            dir = self.direction_to_command(direction)
            if max_count >= 4:
                dir = Position.convert_to_rotate(dir)
            return dir

        logger.logger.info("RFID Dir: No match")
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

        if (RFID_CARD in response) or (RFID_ACTIVE in response):
            with self.lock:
                self.detection_counts[self.current_antenna] += 1

    def confirm_direction(self):
        if Position.invert(self.delayed_direction) == self.predict_direction:
            self.predict_direction = self.delayed_direction

    def start_timer(self, new_direction):
        self.delayed_direction = new_direction
        self.timer.register("direction_timer")

    def process_direction(self):
        counts_copy = self.detection_counts.copy()
        new_direction = self.antenna_to_direction(counts_copy)
        counts_copy = self.detection_counts_sec
        left = counts_copy.get(RFIDAntenna.LEFT.value, 0)
        right = counts_copy.get(RFIDAntenna.RIGHT.value, 0)
        center = counts_copy.get(RFIDAntenna.CENTER.value, 0)

        # 回転途中に更新が間に合わない場合に直進
        if new_direction != Commands.GO_CENTER and new_direction != Commands.STOP_TEMP:
            if (
                self.predict_direction in [Commands.ROTATE_LEFT, Commands.GO_LEFT]
                and center >= left
                or self.predict_direction in [Commands.ROTATE_RIGHT, Commands.GO_RIGHT]
                and center >= right
            ):
                new_direction = Commands.GO_CENTER

        if (
            Commands.is_go(self.last_direction)
            and Position.invert(new_direction) == self.predict_direction
        ):
            self.start_timer(new_direction)
        else:
            self.timer.remove("direction_timer")
            self.predict_direction = new_direction

        if self.timer.passed("direction_timer", 1, remove=True):
            self.confirm_direction()

    def monitor_rfid(self):
        """RFIDタグを監視し、検出回数に基づいて方向を予測します。"""
        try:
            while self.running:
                # 1. CENTER、LEFT、RIGHTアンテナでEPC読み取り
                for antenna_id in [1, 2, 3, 4]:
                    self.switch_antenna(antenna_id)
                    for _ in range(5):
                        self.read_epc()

                with self.lock:
                    self.process_direction()

                new_direction = self.predict_direction
                if self.predict_direction != Commands.STOP_TEMP:
                    self.last_move_direction = self.predict_direction

                self.last_direction = new_direction
                if (
                    new_direction == Commands.GO_LEFT
                    or new_direction == Commands.GO_RIGHT
                ):
                    self.prev_lr = new_direction

                for key in self.detection_counts:
                    self.detection_counts[key] = 0

                if self.get_max_count() == 0:
                    self.timer.register("no_rfid")
                else:
                    self.timer.remove("no_rfid")

                if self.timer.passed("no_rfid", 2, remove=True):
                    self.send_command("N1,26")

                counts_str = ", ".join(
                    [
                        f"{ANTENNA_MAPS[ant]}={cnt}"
                        for ant, cnt in self.get_detection_counts().items()
                    ]
                )

                logger.logger.debug(
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

    def is_zero_detection(self):
        return self.get_max_count() == 0

    def get_max_count(self):
        return max(self.get_detection_counts().values(), default=0)

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
    reader = RFIDReader(port="COM8")
    reader.start_reading()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        reader.close()
        logger.logger.info("RFID Reader stopped.")


if __name__ == "__main__":
    main()
