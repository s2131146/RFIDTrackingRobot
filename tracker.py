import os
import threading
import cv2
import time
import math
import tkinter as tk
import numpy as np
import asyncio
from enum import Enum

from constants import Commands
from rfid import RFIDReader
import tracker_socket as ts
import gui
import logger as l

from obstacles import Obstacles
from target_processor import TargetProcessor

# カメラ起動高速化
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

from ultralytics import YOLO

# シリアルポートの設定
SERIAL_PORT = "COM3"
SERIAL_BAUD = 19200
SERIAL_SEND_INTERVAL = 0.03
SERIAL_SEND_MOTOR_INTERVAL = 0.5

TCP_PORT = 8001

# キャプチャウィンドウサイズ
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CAM_NO = 0

DEBUG_SERIAL = True
DEBUG_USE_WEBCAM = False
DEBUG_DETECT_OBSTACLES = True

# Cascade 関連のフラグとリストを削除

logger = l.logger


class Tracker:
    """Trackerクラス: 人物検出と距離推定（画像面積ベース）、モーター制御を行う"""

    # 動作モード
    class Mode(Enum):
        CAM_ONLY = 1
        DUAL = 2
        RFID_ONLY = 3

    def __init__(self):
        self.root = self.video_capture = None
        self.frame_width = FRAME_WIDTH
        self.frame_height = FRAME_HEIGHT
        self.d_x = self.d_y = self.face_x = self.face_center_x = 0
        self.max_motor_power_threshold = self.motor_power_l = self.motor_power_r = 0
        self.interval_serial_send_start_time = self.seg = self.serial_sent = 0
        self.rfid_accessing = False
        self.rfid_reader = None
        self.lost_target_command = Commands.STOP_TEMP
        self.stop_exec_cmd = False

        self.default_speed = 250
        self._def_spd_bk = 0

        self.gui: gui.GUI

        self.serial = ts.TrackerSocket(
            SERIAL_PORT,
            SERIAL_BAUD,
            SERIAL_SEND_INTERVAL,
            DEBUG_SERIAL,
            TCP_PORT,
            DEBUG_USE_WEBCAM,
        )

        self.stop = False
        self.target_last_seen_time = None  # 対象が最後に検出された時刻

        # 自動停止の距離閾値（占有率）
        self.AUTO_STOP_OCCUPANCY_RATIO = 0.6  # 60%

        # 近接の閾値（占有率）
        self.CLOSE_OCCUPANCY_RATIO = 0.5  # 40%以上で「近い」と判断

        # モーター出力の最低限の値（対象が近くない場合）
        self.MIN_MOTOR_POWER = 80  # 80%

        # 自動停止状態を管理するフラグ
        self.auto_stop = False

        self.RFID_ENABLED = False
        self.RFID_ONLY_MODE = True

        # YOLOモデルのロード
        self.model = YOLO("models\\yolov8n.pt")

        # 障害物検出クラスのインスタンス作成
        self.obstacles = Obstacles(FRAME_WIDTH, FRAME_HEIGHT, logger)

        # 対象検出クラスのインスタンス作成
        self.target_processor = TargetProcessor(
            FRAME_WIDTH, FRAME_HEIGHT, self.model, logger, self
        )

        self.stop_temp = False
        self.detected_obstacles = []
        self.occupancy_ratio = 0
        self.is_close = False

        self.target_position_str = "X"
        self.target_x = 0
        self.safe_x = 0

        self.received_serial = None
        self.disconnect = False

        self.fps = 0
        self.total_fps = 0
        self.avr_fps = 0
        self.fps_count = 0
        self.bkl = 0
        self.bkr = 0

        self.send_check_start_time = 0
        self.sent_count = 0

        self.initialized = False
        self.first_disable_tracking = True

    def update_mode(self):
        self.init()

    def set_mode(self, mode):
        self.mode = mode
        logger.info(f"Mode set to {self.mode}")

        if self.mode == self.Mode.CAM_ONLY.name:
            self.RFID_ENABLED = False
            self.RFID_ONLY_MODE = False
            self.default_speed = 250
        elif self.mode == self.Mode.DUAL.name:
            self.RFID_ENABLED = True
            self.RFID_ONLY_MODE = False
            self.default_speed = 250
        elif self.mode == self.Mode.RFID_ONLY.name:
            self.RFID_ONLY_MODE = True
            self.RFID_ENABLED = True
            self.default_speed = 100

    def elapsed_str(self, start_time):
        """経過時間の文字列を取得

        Args:
            start_time (float): 計測開始時刻

        Returns:
            str: 経過時間の文字列
        """
        return "{}ms".format(math.floor((time.time() - start_time) * 1000))

    def print_d(self, text):
        """デバッグ用画面出力

        Args:
            text (str): 出力文字列
        """
        text_lines = text.split("\n")
        for index, line in enumerate(text_lines, start=1):
            cv2.putText(
                self.frame,
                line,
                (5, 15 * index),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 0),
            )

    def print_key_binds(self):
        """キーバインドを出力

        Args:
            frame (MatLike): フレーム
        """
        if self.stop:
            s = "START"
        else:
            s = Commands.STOP
        text = "PRESS X TO {}.\nPRESS Q TO EXIT.\n{}".format(s, self.seg)
        text_lines = text.split("\n")
        text_lines.reverse()
        for index, line in enumerate(text_lines, start=1):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = math.floor(self.frame_width - text_size[0] - 15)
            text_y = math.floor(self.frame_height - (15 * index)) + 5
            cv2.putText(
                self.frame,
                line,
                (text_x, text_y),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 0),
            )

    def print_serial_stat(self):
        """シリアル通信状況の出力

        Args:
            frame (MatLike): フレーム
        """
        text = f"SERIAL SENT {str(self.sent_count).zfill(2)}"
        text_y = math.floor(self.frame_height - 10)
        cv2.putText(
            self.frame, text, (5, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0)
        )

    def print_stop(self):
        """ストップ出力

        Args:
            frame (MatLike): フレーム
        """
        text = "STOP"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 10

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        text_x = (int)(self.frame_width - text_size[0]) // 2
        text_y = (int)(self.frame_height + text_size[1]) // 2

        cv2.putText(
            self.frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness
        )

        padding = 50
        top_left = (text_x - padding, text_y - text_size[1] - padding)
        bottom_right = (text_x + text_size[0] + padding, text_y + padding)
        cv2.rectangle(self.frame, top_left, bottom_right, (0, 255, 255), thickness)

    def send(self, data):
        if isinstance(data, tuple):
            command, value = data
            data = f"{command}:{value}"

        if not Commands.contains(self.serial.get_command(data)):
            logger.warning(f"Invalid command: {data}")
            return False

        if data == Commands.START:
            return True

        if (
            self.prev_command == data
            and data != Commands.CHECK
            and data != Commands.STOP
            and data != Commands.STOP_TEMP
        ):
            return True

        if data != Commands.CHECK:
            self.prev_command = data

        ret = self.serial.send_serial(data)
        if ret:
            self.print_serial_stat()
        self.exec_sent_command()

        if not Commands.is_ignore(data):
            self.gui.queue.add("s", self.serial.serial_sent)

        return True

    def exec_sent_command(self):
        c = self.serial.command_sent.upper()
        if c == Commands.STOP:
            self.stop = True
            self.gui.update_stop()
            logger.info("Motor stopped")
        elif c == Commands.START:
            self.start_motor()
            logger.info("Motor started")
        elif c == Commands.STOP_TEMP:
            self.stop_temp = True

    def calculate_motor_power(self, target_x, obs=False):
        """モーター出力を計算

        Args:
            target_x (int): 対象座標X
            obs (bool): 障害物検知のフラグ
        """
        # 定義された占有率と対応する基礎速度のポイント
        # 占有率: 0% -> 100%, 15% -> 100%, 40% -> 80%, 60% -> 60%, 100% -> 0%
        occupancy_thresholds = [0.0, 0.15, 0.40, 0.60, 1.0]
        base_speeds = [100, 100, 80, 60, 0]

        occupancy = self.occupancy_ratio

        # 線形補間を使用して基礎速度を計算
        base_speed = np.interp(occupancy, occupancy_thresholds, base_speeds)

        # 速度をクランプ（0%〜100%）
        base_speed = max(0, min(100, base_speed))

        # 中央15%の閾値
        central_threshold = 0.15 * self.frame_width

        if abs(target_x) <= central_threshold:
            # 中央15%以内に対象がいる場合、モーター出力を基礎速度に設定
            self.motor_power_l = base_speed
            self.motor_power_r = base_speed
        else:
            # 中央15%を超える場合、対象の位置に基づいてモーター出力を調整
            # フレームの半幅
            half_width = self.frame_width / 2

            # ターゲットの位置を正規化（-1.0 ~ 1.0）
            normalized_error = target_x / (0.85 * half_width)  # 0.85は調整可能

            # -1.0 ~ 1.0 にクランプ
            normalized_error = max(-1.0, min(1.0, normalized_error))

            # 方向調整の強さを設定（0〜100）
            steer_strength = 50  # 調整可能

            if normalized_error < 0:
                # 左に曲がる場合
                self.motor_power_l = max(
                    0, min(100, base_speed - abs(normalized_error) * steer_strength)
                )
                self.motor_power_r = max(
                    0, min(100, base_speed + abs(normalized_error) * steer_strength)
                )
            elif normalized_error > 0:
                # 右に曲がる場合
                self.motor_power_r = max(
                    0, min(100, base_speed - abs(normalized_error) * steer_strength)
                )
                self.motor_power_l = max(
                    0, min(100, base_speed + abs(normalized_error) * steer_strength)
                )
            else:
                # 中央の場合
                self.motor_power_l = base_speed
                self.motor_power_r = base_speed

        # 障害物が検出された場合、再帰的にモーター出力を調整
        if not self.no_obs and not obs:
            self.calculate_motor_power(-self.safe_x, obs=True)

    def get_target_pos_str(self, target_center_x):
        return self.target_processor.get_target_pos_str(target_center_x)

    def stop_motor(self, no_target=False):
        if no_target:
            self.send(Commands.STOP_TEMP)
        else:
            self.send(Commands.STOP)

    def start_motor(self):
        if self.serial.ser_connected:
            self.stop = False
            self.gui.update_stop()
            self.send(Commands.START)

    def main_loop(self):
        logger.info("Starting main loop")
        self.initialize_loop_variables()

        while True:
            if not self.initialized:
                continue
            self.update_seg()
            frame_start_time = time.time()
            self.reset_detection_flags()
            self.capture_frame()
            if self.gui.var_enable_tracking.get():
                self.first_disable_tracking = True
                self.process_obstacles_and_targets()
            elif self.first_disable_tracking:
                self.first_disable_tracking = False
                self.send(Commands.STOP_TEMP)
            self.update_motor_power()
            self.update_rfid_power()
            self.send_commands_if_needed()
            self.receive_serial_data()
            self.update_fps(frame_start_time)
            self.handle_disconnection()
            self.update_debug_info()
            self.update_gui()

    def initialize_loop_variables(self):
        self.fps = 0
        self.total_fps = 0
        self.avr_fps = 0
        self.fps_count = 0
        self.bkl = 0
        self.bkr = 0
        self.received_serial = None
        self.disconnect = False
        self.stop_temp = False
        self.target_last_seen_time = time.time()
        self.target_detection_start_time = None
        self.time_detected = 0
        self.max_motor_power_threshold = self.frame_width / 4
        self.motor_power_l, self.motor_power_r = 0, 0

        self.interval_serial_send_start_time = 0
        self.interval_serial_send_motor_start_time = 0
        self.send_check_start_time = 0

        self.seg = 0
        self.sent_count = 0

    def update_motor_power(self):
        if not self.stop and not self.stop_temp:
            self.gui.root.after(
                0, self.gui.update_motor_values, self.motor_power_l, self.motor_power_r
            )

    def update_rfid_power(self):
        if self.RFID_ENABLED and self.rfid_accessing:
            detection_counts = self.rfid_reader.get_detection_counts()
            self.gui.update_rfid_values(detection_counts)
        else:
            self.gui.update_rfid_values({1: 0, 2: 0, 3: 0, 4: 0})

    def update_seg(self):
        self.seg += 1
        if self.seg > 9:
            self.seg = 0
        self.gui.update_seg(self.seg)

    def reset_detection_flags(self):
        self.target_detected = False
        self.target_position_str = "X"
        self.stop_exec_cmd = False
        self.target_x = 0
        self.no_obs = True
        self.safe_x = 0

    def capture_frame(self):
        if self.RFID_ONLY_MODE and self.RFID_ENABLED:
            self.frame = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255
        else:
            if DEBUG_USE_WEBCAM:
                ret, frame = self.video_capture.read()
                if not ret:
                    logger.warning("Failed to read frame from webcam.")
                    self.frame = (
                        np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255
                    )
                else:
                    self.frame = cv2.flip(frame, 1)
            else:
                if (
                    hasattr(self.serial, "color_image")
                    and self.serial.color_image is not None
                ):
                    self.frame = np.copy(self.serial.color_image)
                else:
                    self.frame = (
                        np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255
                    )

    def execute_rfid_direction(self, rotate=False):
        """RFIDの移動方向を取得して移動指示を送信する"""
        self.rfid_accessing = True
        if not self.stop or (self.auto_stop and self.mode == Tracker.Mode.DUAL):
            if rotate:
                direction = self.rfid_reader.get_rotate_direction()
            else:
                direction = self.rfid_reader.get_direction()
            if direction != Commands.STOP_TEMP and self.auto_stop:
                self.start_motor()
            self.send(direction)

    def process_obstacles_and_targets(self):
        current_time = time.time()

        if not DEBUG_USE_WEBCAM and DEBUG_DETECT_OBSTACLES:
            self.frame, self.no_obs, self.safe_x = self.obstacles.process_obstacles(
                self.frame, self.serial.depth_image
            )

        detected_targets = self.target_processor.detect_targets(
            self.frame, self.RFID_ONLY_MODE, self.RFID_ENABLED
        )

        self.rfid_accessing = False

        if not self.RFID_ONLY_MODE:
            if len(detected_targets) > 1:
                # 二人以上検出された場合の処理
                logger.info(f"Multiple targets detected: {len(detected_targets)}")
                if self.RFID_ENABLED:
                    self.execute_rfid_direction()
            elif len(detected_targets) == 1:
                # 一人のみ検出された場合の既存の処理
                if self.target_detection_start_time is None:
                    self.target_detection_start_time = current_time
                else:
                    self.time_detected = current_time - self.target_detection_start_time

                    if self.time_detected >= 0.3:
                        self.target_last_seen_time = current_time
                        self.stop_temp = False
                        result = self.target_processor.process_target(
                            detected_targets, self.frame
                        )
                        if result is not None:
                            (target_center_x, self.target_x) = result
                            self.target_position_str = self.get_target_pos_str(
                                target_center_x
                            )
                            self.calculate_motor_power(self.target_x)
            elif len(detected_targets) == 0:
                if self.target_detection_start_time is not None:
                    self.target_detection_start_time = None

                time_since_last_seen = current_time - self.target_last_seen_time
                if time_since_last_seen > 1.0 and not self.stop_temp:
                    logger.info("Target lost")
                    if self.RFID_ENABLED:
                        self.execute_rfid_direction()
                    else:
                        self.stop_exec_cmd = True
                        self.send(self.lost_target_command)
            else:
                if self.RFID_ENABLED:
                    self.execute_rfid_direction()

        if self.RFID_ONLY_MODE and self.RFID_ENABLED:
            self.target_position_str = self.rfid_reader.get_direction()
            if not self.stop:
                self.send(self.target_position_str)

        if self._def_spd_bk != self.default_speed:
            self.send((Commands.SET_DEFAULT_SPEED, self.default_speed))
            self._def_spd_bk = self.default_speed

    def send_commands_if_needed(self):
        current_time = time.time()
        if current_time - self.interval_serial_send_start_time > SERIAL_SEND_INTERVAL:
            self.sent_count += 1
            self.interval_serial_send_start_time = current_time
            self.send(Commands.CHECK)

        if (
            current_time - self.interval_serial_send_motor_start_time
            > SERIAL_SEND_MOTOR_INTERVAL
        ):
            self.interval_serial_send_motor_start_time = current_time
            if not self.stop and not self.stop_temp and not self.stop_exec_cmd:
                self.send((Commands.L_SPEED, self.motor_power_l))
                self.bkl = self.motor_power_l
                self.send((Commands.R_SPEED, self.motor_power_r))
                self.bkr = self.motor_power_r

        if current_time - self.send_check_start_time > 1:
            self.sent_count = 0
            self.send_check_start_time = current_time

    def receive_serial_data(self):
        received = self.serial.get_received_queue()
        if received:
            self.gui.queue.add_all("r", received)
            self.received_serial = received[-1]
            # ログに受信データを記録
            logger.info(f"Received Serial Data: {received[-1]}")

    def update_fps(self, frame_start_time):
        fps_end_time = time.time()
        time_taken = fps_end_time - frame_start_time
        if time_taken > 0:
            self.fps = 1 / time_taken

        self.fps_count += 1
        self.total_fps += self.fps
        if self.fps_count >= 10:
            self.avr_fps = self.total_fps / self.fps_count
            self.fps_count = 0
            self.total_fps = 0

    def handle_disconnection(self):
        """接続状態を監視し、切断時の処理を行う"""
        if not self.serial.ser_connected:
            self.stop = True
            self.print_stop()
            if self.serial.test_connect():
                asyncio.run(self.serial.connect_socket())
            if not self.disconnect:
                self.gui.update_stop(connected=False)
                self.disconnect = True
                logger.warning("Disconnected from serial")
        else:
            if self.disconnect:
                self.gui.update_stop(connected=True)
                logger.info("Reconnected to serial")
            self.disconnect = False

    def update_debug_info(self):
        debug_text = (
            f"{self.seg} {math.floor(self.frame_width)} x {math.floor(self.frame_height)} "
            f"{'O' if self.target_position_str != 'X' else 'X'} x: {self.d_x} y: {self.d_y} "
            f"avr_fps: {math.floor(self.avr_fps)} fps: {math.floor(self.fps)} "
            f"target_position: {self.target_position_str}\n"
            f"target_x: {self.target_x} mpl: {self.motor_power_l} mpr: {self.motor_power_r} bkl: {self.bkl} bkr: {self.bkr}\n"
            f"connected: {self.serial.ser_connected} port: {SERIAL_PORT} baud: {SERIAL_BAUD} data: {self.serial.serial_sent}\n"
            f"ip: {self.serial.tcp_ip} STOP: {self.stop or self.stop_temp} serial_received: {self.received_serial}\n"
            f"obstacle_detected: {not self.no_obs} safe_x: {self.safe_x} "
            f"no_detection: {round(time.time() - self.target_last_seen_time, 2):.2f} "
            f"detecting: {round(self.time_detected, 2):.2f}\n"
            f"self.RFID_ENABLED: {self.RFID_ENABLED} RFID_only: {self.RFID_ONLY_MODE} def_speed: {self.default_speed}\n"
            f"Auto_Stop: {self.auto_stop} close: {self.is_close} occupancy: {self.occupancy_ratio:.2%}"
        )
        self.print_d(debug_text)
        self.print_key_binds()
        if self.stop:
            self.print_stop()

    def update_gui(self):
        if self.frame is not None:
            self.gui.update_frame(self.frame)

    def close(self):
        logger.info("Closing application")
        self.stop_motor()
        self.serial.close()

        if DEBUG_USE_WEBCAM:
            self.video_capture.release()

        cv2.destroyAllWindows()

    def init(self):
        re_init = self.initialized
        self.initialized = False
        if not re_init:
            logger.info("Starting up...")
        time_startup = time.time()

        if self.video_capture is not None:
            self.video_capture.release()
        if self.rfid_reader is not None:
            self.rfid_reader.close()
            self.update_rfid_power()

        self.set_mode(self.gui.mode_var.get())

        self.frame_width = FRAME_WIDTH
        self.frame_height = FRAME_HEIGHT

        if DEBUG_USE_WEBCAM and not self.RFID_ONLY_MODE:
            self.video_capture = cv2.VideoCapture(CAM_NO)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            self.frame_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.frame_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.frame = None

        if self.RFID_ENABLED:
            logger.info("Initializing RFID Reader")
            self.rfid_reader = RFIDReader(port="COM4", baudrate=38400)
            logger.info(
                f"RFID Antenna Status: {self.rfid_reader.check_antenna_status()}"
            )
            self.rfid_reader.start_reading()

        self.d_x, self.d_y = 0, 0
        self.face_x, self.face_center_x = 0, 0
        self.target_detection_start_time = None
        self.time_detected = 0
        self.prev_command = Commands.STOP_TEMP

        self.max_motor_power_threshold = self.frame_width / 4
        self.motor_power_l, self.motor_power_r = 0, 0

        self.interval_serial_send_start_time = 0
        self.interval_serial_send_motor_start_time = 0
        self.send_check_start_time = 0
        self.occupancy_ratio = 0
        self.is_close = False

        self.seg = 0
        self.sent_count = 0

        logger.info(
            "Startup completed. Elapsed time: {}".format(self.elapsed_str(time_startup))
        )

        if not re_init:
            tracker_thread = threading.Thread(target=self.run_loop_serial)
            tracker_thread.daemon = True
            tracker_thread.start()

        self.initialized = True

    def run_loop_serial(self):
        asyncio.run(self.serial.loop_serial())

    def start(self):
        tracker_thread = threading.Thread(target=self.main_loop)
        tracker_thread.daemon = True
        tracker_thread.start()

        self.root.mainloop()
        self.close()


def main():
    tracker = Tracker()
    logger.info("Initializing GUI")
    tracker.root = tk.Tk()
    tracker.gui = gui.GUI(tracker, tracker.root)
    tracker.gui.queue.wait_for("init")
    tracker.init()
    tracker.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("An error occurred in the main execution loop.")
