import os
import subprocess
import traceback
import psutil

# カメラ起動高速化
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import threading
import cv2
import time
import math
import tkinter as tk
import numpy as np
import asyncio

from enum import Enum
from typing import Optional, Tuple
from ultralytics import YOLO

from obstacles import Obstacles
from rfid import RFIDAntenna, RFIDReader
from constants import Commands, Position
import gui
import logger as l
import tracker_socket as ts

# シリアルポートの設定
ROBOT_SERIAL_PORT = "COM3"
ROBOT_SERIAL_BAUD = 19200
ROBOT_SERIAL_SEND_INTERVAL = 0.03
ROBOT_SERIAL_SEND_MOTOR_INTERVAL = 0.5

RFID_SERIAL_PORT = "COM8"

TCP_PORT = 8001

# キャプチャウィンドウサイズ
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

WEB_CAM_NO = 0

DEBUG_SLOW_MOTOR = True
DEBUG_SERIAL = True
DEBUG_USE_ONLY_WEBCAM = False
DEBUG_DETECT_OBSTACLES = True
DEBUG_INVERT_MOTOR = False
DEBUG_ENABLE_TRACKING = False
DEBUG_SLOW_SPEED = 150

# 壁回避時の決め打ちパラメータ
DELAY_SPEED_FAST = 4
DELAY_SPEED_SLOW = 6

logger = l.logger


class Tracker:
    """Trackerクラス: 人物検出と距離推定（画像面積ベース）、モーター制御を行う"""

    class Mode(Enum):
        CAM_ONLY = 1
        DUAL = 2
        RFID_ONLY = 3

    def __init__(self):
        # GUIのroot
        self.root: tk.Tk
        self.gui: gui.GUI

        # Webカメラ関連
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.frame_width = FRAME_WIDTH
        self.frame_height = FRAME_HEIGHT

        # 左右のモーター出力
        self.motor_power_l = self.motor_power_r = 0

        # CHECKコマンド送信インターバル管理用
        self.interval_serial_send_start_time = 0

        # RFID関連
        self.rfid_accessing = False
        self.rfid_reader: Optional[RFIDReader] = None

        # 対象を見失った際に実行するコマンド（最後の対象の位置確認と兼用）
        self.lost_target_command = Commands.STOP_TEMP

        # 現在の最大スピード
        self.default_speed = 250
        self._def_spd_bk = 0

        self.serial = ts.TrackerSocket(
            ROBOT_SERIAL_PORT,
            ROBOT_SERIAL_BAUD,
            ROBOT_SERIAL_SEND_INTERVAL,
            DEBUG_SERIAL,
            TCP_PORT,
            DEBUG_USE_ONLY_WEBCAM,
        )

        # 現在停止中か
        self.stop = False

        # 対象が最後に検出された時刻
        self.target_last_seen_time = None

        # 自動停止の距離閾値（占有率）
        self.AUTO_STOP_OCCUPANCY_RATIO = 0.6

        # 近接の閾値（占有率）
        self.CLOSE_OCCUPANCY_RATIO = 0.5

        # 自動停止状態を管理するフラグ
        self.auto_stop = False
        self.auto_stop_obs = False

        self.RFID_ENABLED = False
        self.RFID_ONLY_MODE = True

        # YOLOモデルのロード
        self.model = YOLO("models\\yolov8n.pt")

        from target_processor import TargetProcessor, Target

        # 障害物検出クラスのインスタンス作成
        self.obstacles = Obstacles(FRAME_WIDTH, FRAME_HEIGHT, logger)

        # 対象検出クラスのインスタンス作成
        self.target_processor = TargetProcessor(
            FRAME_WIDTH, FRAME_HEIGHT, self.model, logger, self
        )

        # 一時停止中か
        self.stop_temp = False
        self.detected_obstacles = []
        self.occupancy_ratio = 0

        # 障害物が目の前か
        self.is_close_obs = False

        # 現在選択中の対象
        self.target: Optional[Target] = None

        # 最後に対象を見つけた時刻
        self.target_last_seen_time = time.time()

        # GUI反映用受信データ
        self.received_serial = None

        # シリアル通信を一時切断中か
        self.disconnect = False

        self.fps = 0
        self.total_fps = 0
        self.avr_fps = 0
        self.fps_count = 0
        self.bkl = 0
        self.bkr = 0

        # モーター出力を停止するフラグ
        self.stop_exec_cmd = False
        self.stop_exec_cmd_gui = False

        # 1秒間にCHECKした回数記憶用
        self.send_check_start_time = 0
        self.sent_count = 0

        # 初期化完了フラグ
        self.initialized = False

        # Trackingが無効化されてから初めての実行
        self.first_disable_tracking = True

        # 転回して対象を探すか
        self.find_target_rotate = False

        # 転回してから経過した時間
        self.find_target_start_time = time.time()

        # 後退してから経過した時間
        self.obs_backoff_start_time = None

        self.init_detection_flags()
        self.init_obstacles_vals()

    def init_obstacles_vals(self):
        # 障害物の位置
        self.obs_pos = Obstacles.OBS_NONE

        # 壁が存在
        self.wall = None

        # 壁を回避
        self.avoid_wall = False

        # 壁に平行に進むか
        self.wall_parallel = False

        # 壁に近すぎる
        self.too_close_wall = False

        # 壁を避けた時刻
        self.last_wall_avoid_time = time.time()

        # 壁を検出した時刻
        self.last_wall_detect_time = time.time()

        # 最後に検出した壁の位置
        self.last_wall_detect = Position.NONE

        # 最後に避けた壁の位置
        self.last_wall_avoid = Position.NONE

        # 壁に隠れた対象を追いかけるか
        self.tracking_target_invisible = False

        # 最も近い壁の距離
        self.closest_wall_depth = 0

        # 壁の情報を更新しないか
        self.reset_to_backup = False

    def update_mode(self):
        """モードの更新"""
        self.init()

    def set_mode(self, mode):
        """動作モードを設定"""
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

    def draw_red_border(self, thickness=5):
        """録画時の赤い枠線を描画"""
        height, width, _ = self.frame.shape

        cv2.line(self.frame, (0, 0), (width, 0), (0, 0, 255), thickness)
        cv2.line(
            self.frame, (0, height - 1), (width, height - 1), (0, 0, 255), thickness
        )
        cv2.line(self.frame, (0, 0), (0, height), (0, 0, 255), thickness)
        cv2.line(
            self.frame, (width - 1, 0), (width - 1, height), (0, 0, 255), thickness
        )

    def elapsed_str(self, start_time):
        """経過時間をミリ秒単位の文字列で取得"""
        return "{}ms".format(math.floor((time.time() - start_time) * 1000))

    def print_d(self, text):
        """デバッグ情報をフレームに描画"""
        overlay = self.frame.copy()
        alpha = 0.6  # 半透明の背景の透明度

        text_lines = text.split("\n")
        for index, line in enumerate(text_lines, start=1):
            (text_width, text_height), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1
            )
            top_left = (5, 15 * index - text_height - baseline // 2)
            bottom_right = (5 + text_width, 15 * index + baseline)
            cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), -1)
            cv2.putText(
                overlay,
                line,
                (5, 15 * index),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
        cv2.addWeighted(overlay, alpha, self.frame, 1 - alpha, 0, self.frame)

    def print_key_binds(self):
        """キーバインド情報をフレームに描画"""
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
        """シリアル通信のステータスをフレームに描画"""
        text = f"SERIAL SENT {str(self.sent_count).zfill(2)}"
        text_y = math.floor(self.frame_height - 10)
        cv2.putText(
            self.frame, text, (5, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0)
        )

    def print_stop(self):
        """フレームに停止状態を表示"""
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
        """データをシリアル通信で送信"""
        if not self.gui.var_enable_serial.get():
            return

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
            and data != Commands.DETACH_MOTOR
            and data != Commands.SPD_UP
            and data != Commands.SPD_DOWN
        ):
            return True

        if data != Commands.CHECK and data != Commands.STOP_TEMP:
            self.prev_command = data

        ret = self.serial.send_serial(data)
        if ret:
            self.print_serial_stat()
        self.exec_sent_command()

        if not Commands.is_ignore(data):
            self.gui.queue.add("s", self.serial.serial_sent)

        return True

    def exec_sent_command(self):
        """送信したコマンドに基づいて内部状態を更新"""
        c = self.serial.command_sent.upper()
        if c == Commands.STOP:
            self.stop = True
            self.motor_power_l = 0
            self.motor_power_r = 0
            self.gui.update_stop()
            logger.info("Motor stopped")
        elif c == Commands.START:
            self.start_motor()
            logger.info("Motor started")
        elif c == Commands.STOP_TEMP:
            self.motor_power_l = 0
            self.motor_power_r = 0
            self.stop_temp = True

    def apply_motor_wall(self, wall_direction: str) -> Tuple[int, int]:
        """壁の位置に基づいて、壁に沿って進むためのモーター出力を計算"""
        base_power = max(self._calculate_base_speed(), 40)
        low_power = 40 if not self.too_close_wall else 0
        high_motor = base_power if not DEBUG_INVERT_MOTOR else low_power
        low_motor = low_power if not DEBUG_INVERT_MOTOR else base_power
        is_fix_pos = Obstacles.is_fix(self.wall_parallel)
        wall_direction = (
            Position.invert(wall_direction)
            if self.wall_parallel == Obstacles.OBS_PARALLEL_FIX_INVERT
            else wall_direction
        )

        if is_fix_pos:
            low_motor = base_power * 0.6

        if self.target_position != Position.NONE:
            wall_direction = Position.invert(self.target_position)

        if wall_direction == Position.LEFT and (
            is_fix_pos or self.motor_power_r > low_motor
        ):
            self.motor_power_l = high_motor
            self.motor_power_r = low_motor
        elif wall_direction == Position.RIGHT and (
            is_fix_pos or self.motor_power_l > low_motor
        ):
            self.motor_power_l = low_motor
            self.motor_power_r = high_motor

    def calculate_motor_power(self, target_x=0):
        """対象の位置に基づいてモーター出力を計算"""
        if self.obs_detected:
            self._adjust_for_obstacles(target_x)
            return

        base_speed = self._calculate_base_speed()
        self._adjust_motor_power_based_on_target_position(target_x, base_speed)

    def _adjust_for_obstacles(self, target_x):
        """障害物が検出された場合にモーター出力を調整"""
        if self.obs_pos == Obstacles.OBS_LEFT:
            target_x = 160 if target_x > 0 else -160
        elif self.obs_pos == Obstacles.OBS_RIGHT:
            target_x = -160 if target_x < 0 else 160
        elif self.obs_pos == Obstacles.OBS_CENTER:
            self.send(Commands.ROTATE_LEFT)
            self.stop_exec_cmd = True

    def _calculate_base_speed(self):
        """占有率に基づいて基礎速度を計算"""
        occupancy_thresholds = [0.0, 0.15, 0.40, 0.60, 1.0]
        base_speeds = [100, 100, 80, 60, 0]
        occupancy = self.occupancy_ratio
        base_speed = np.interp(occupancy, occupancy_thresholds, base_speeds)
        return int(max(0, min(100, base_speed)))

    def _adjust_motor_power_based_on_target_position(self, target_x, base_speed):
        """対象の位置に基づいてモーター出力を調整"""
        central_threshold = 0.15 * self.frame_width

        if abs(target_x) <= central_threshold:
            self.motor_power_l = base_speed
            self.motor_power_r = base_speed
        else:
            self._calculate_steering_power(target_x, base_speed)

    def _calculate_steering_power(self, target_x, base_speed):
        """左右のモーター出力を調整して旋回を行う"""
        half_width = self.frame_width / 2
        normalized_error = target_x / (0.85 * half_width)
        normalized_error = max(-1.0, min(1.0, normalized_error))
        steer_strength = 50

        if normalized_error < 0:
            self.motor_power_l = max(
                0, min(100, base_speed - abs(normalized_error) * steer_strength)
            )
            self.motor_power_r = max(
                0, min(100, base_speed + abs(normalized_error) * steer_strength)
            )
        elif normalized_error > 0:
            self.motor_power_r = max(
                0, min(100, base_speed - abs(normalized_error) * steer_strength)
            )
            self.motor_power_l = max(
                0, min(100, base_speed + abs(normalized_error) * steer_strength)
            )
        else:
            self.motor_power_l = base_speed
            self.motor_power_r = base_speed

        if DEBUG_INVERT_MOTOR:
            temp = self.motor_power_l
            self.motor_power_l = self.motor_power_r
            self.motor_power_r = temp

    def get_target_pos_str(self, target_center_x):
        return self.target_processor.get_target_pos_str(target_center_x)

    def stop_motor(self, no_target=False):
        """モーターを停止"""
        if no_target:
            self.send(Commands.STOP_TEMP)
        else:
            self.send(Commands.STOP)

    def start_motor(self):
        if self.serial.ser_connected:
            self.stop = False
            self.stop_temp = False
            self.gui.update_stop()
            self.send(Commands.START)

    last_obstacle_time = time.time()

    def process_obstacles(self, detected_targets):
        """障害物の処理を行う"""
        if not DEBUG_USE_ONLY_WEBCAM and self.gui.var_enable_obstacles.get():
            if time.time() - self.last_obstacle_time >= 0.1:
                self._process_obstacles(detected_targets)
                self.last_obstacle_time = time.time()
        else:
            self.init_obstacles_vals()

    def process_motor(self):
        """対象の位置に基づいてモーターを制御"""
        self._control_motor_for_target()

        self.motor_power_l = int(self.motor_power_l)
        self.motor_power_r = int(self.motor_power_r)

        self._send_target_position_if_rfid_mode()
        self._send_default_speed_if_changed()

    def main_loop(self):
        """メインループを実行"""
        logger.info("Starting main loop")

        while True:
            if not self.initialized:
                continue

            self.update_seg()
            frame_start_time = time.time()
            self.init_detection_flags()
            self.capture_frame()

            detected_targets = []
            if self.gui.var_enable_tracking.get():
                if not self.first_disable_tracking:
                    self.stop_temp = False
                self.first_disable_tracking = True
                detected_targets = self.process_targets()
            elif self.first_disable_tracking:
                self.first_disable_tracking = False
                self.send(Commands.STOP_TEMP)

            # 一周しても見つからない場合は停止（音を鳴らすかも）
            if self.stop_exec_cmd and time.time() - self.find_target_start_time > 8:
                self.stop_exec_cmd = False
                self.send(Commands.STOP_TEMP)

            self.process_obstacles(detected_targets)
            self.process_motor()
            self.update_motor_power()
            self.update_rfid_power()
            self.send_commands_if_needed()
            self.receive_serial_data()
            self.update_fps(frame_start_time)
            self.handle_disconnection()
            if self.gui.recording:
                self.draw_red_border()
            self.update_debug_info()
            self.update_gui()

    def update_motor_power(self):
        self.gui.root.after(
            0,
            self.gui.update_motor_values,
            self.motor_power_l,
            self.motor_power_r,
        )

    def update_rfid_power(self):
        if self.RFID_ENABLED:
            detection_counts = self.rfid_reader.get_detection_counts()
            self.gui.update_rfid_values(detection_counts)
        else:
            self.gui.update_rfid_values({1: 0, 2: 0, 3: 0, 4: 0})

    def update_seg(self):
        self.seg += 1
        if self.seg > 9:
            self.seg = 0
        self.gui.update_seg(self.seg)

    def init_detection_flags(self):
        self.target_position = Position.NONE
        self.target_x = -1
        self.obs_detected = False

    def capture_frame(self):
        ret, self.frame = self.video_capture.read()
        if not ret:
            logger.warning("Failed to read frame from webcam.")
            self.frame = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255

    def apply_rfid_direction(self, multi_targets=False):
        """RFIDの方向情報を適用（デュアルモード）"""
        if not self.gui.var_enable_tracking.get():
            return
        self.rfid_accessing = True
        if not self.stop or (self.auto_stop and self.mode == Tracker.Mode.DUAL):
            if multi_targets:
                detected_count = self.rfid_reader.get_detection_counts()
                side = max(
                    detected_count[RFIDAntenna.RIGHT.value],
                    detected_count[RFIDAntenna.LEFT.value],
                )

                if detected_count[RFIDAntenna.CENTER.value] < side:
                    direction = self.rfid_reader.get_rotate_direction()
            else:
                direction = self.rfid_reader.get_direction()
            if direction != Commands.STOP_TEMP and self.auto_stop:
                self.start_motor()

            self.lost_target_command = Position.convert_to_rotate(direction)
            logger.info(
                f"RFID Direction applied to lost command: {self.lost_target_command}"
            )

    def process_targets(self):
        """対象の検出と追跡を処理"""
        current_time = time.time()

        detected_targets = self._detect_and_select_targets(current_time)
        if len(detected_targets) > 0:
            self.target = detected_targets[0]
        else:
            self.target = None

        if (
            len(detected_targets) > 0
            and detected_targets[0].y2 - detected_targets[0].y1
            > self.frame_height * 0.8
        ):
            if not self.find_target_rotate:
                self.find_target_rotate = True
        else:
            self.find_target_rotate = False

        return detected_targets

    def _detect_and_select_targets(self, current_time):
        """対象を検出して選択する内部処理"""
        detected_targets = self.target_processor.detect_targets(
            self.frame, self.RFID_ONLY_MODE, self.RFID_ENABLED
        )
        self.rfid_accessing = False
        self.tracking_target_invisible = False

        selected_target = None

        if not self.RFID_ONLY_MODE:
            if len(detected_targets) > 0:
                selected_target = self._select_and_process_multiple_targets(
                    detected_targets
                )
                if selected_target is None:
                    self._handle_no_targets(current_time)
                else:
                    self.stop_temp = False
                    self.stop_exec_cmd = False
            else:
                self._handle_no_targets(current_time)
        elif not Commands.is_rotate(self.rfid_reader.get_direction()):
            self.stop_exec_cmd = False

        return [selected_target] if selected_target else []

    def _select_and_process_multiple_targets(self, detected_targets):
        """複数の対象が検出された場合の処理を行う"""
        selected_target = self.target_processor.select_target(detected_targets)
        self.target_processor.draw_all_targets(
            detected_targets, selected_target, self.frame
        )
        if selected_target:
            result = self.target_processor.process_target([selected_target], self.frame)
            if result is not None:
                target_center_x, self.target_x, _ = result
                self.target_position = self.get_target_pos_str(target_center_x)

        if self.RFID_ENABLED:
            self.apply_rfid_direction()

        return selected_target

    @classmethod
    def map_value_to_scale(
        cls,
        x,
        in_min=DEBUG_SLOW_SPEED,
        in_max=400,
        out_min=DELAY_SPEED_FAST,
        out_max=DELAY_SPEED_SLOW,
    ):
        """値を指定されたスケールにマッピング"""
        if x < in_min:
            x = in_min
        elif x > in_max:
            x = in_max
        return out_max + (out_min - out_max) * (x - in_min) / (in_max - in_min)

    def exec_lost_target_command(self):
        """対象を見失った際のコマンドを実行"""
        self.send(self.lost_target_command)
        self.find_target_start_time = time.time()
        self.stop_exec_cmd = True
        self.motor_power_l = 0
        self.motor_power_r = 0

    lost_target_avoid_wall = Position.NONE

    force_avoid_wall_start_time = time.time()

    def _handle_no_targets(self, current_time):
        """対象が検出されなかった場合の処理を行う"""
        if (
            self.last_wall_detect != Position.NONE
            and Position.convert_to_rotate(self.last_wall_detect)
            == self.lost_target_command
            and current_time - self.last_wall_avoid_time
            < self.map_value_to_scale(self.default_speed)
            and not self.find_target_rotate
        ):
            if self.lost_target_avoid_wall == Position.NONE:
                self.lost_target_avoid_wall = self.last_wall_detect
            if self.lost_target_command == self.wall:
                self.force_avoid_wall_start_time = time.time()
            if (
                self.lost_target_avoid_wall == self.wall
                and time.time() - self.force_avoid_wall_start_time < 1.5
            ):
                self.last_wall_avoid_time = time.time()

            self.tracking_target_invisible = True
            return

        self.last_wall_avoid = Position.NONE

        if self.target_detection_start_time is not None:
            self.target_detection_start_time = None

        time_since_last_seen = current_time - self.target_last_seen_time
        if time_since_last_seen > 1.0 and not self.stop_temp:
            self.find_target_rotate = False
            if self.RFID_ENABLED:
                self.apply_rfid_direction()
            elif not self.stop and not self.stop_exec_cmd_gui:
                self.exec_lost_target_command()

    def _process_obstacles(self, detected_targets):
        """障害物の処理を行う内部関数"""
        result = self.obstacles.process_obstacles(
            self.serial.depth_image,
            self.target_x,
            detected_targets,
            self.lost_target_command,
        )

        bk_wall = self.wall
        if len(result) == 2:
            if not self.auto_stop_obs:
                self.stop_motor()
                self.auto_stop_obs = True
            _, self.depth_frame = result
            self.is_close_obs = True
            return
        else:
            self.is_close_obs = False
            if self.auto_stop_obs:
                self.auto_stop_obs = False
                self.start_motor()

            self.auto_stop_obs = False
            (
                self.depth_frame,
                wall,
                self.avoid_wall,
                self.wall_parallel,
                self.too_close_wall,
                self.closest_wall_depth,
            ) = result

        self.reset_to_backup = self._update_wall_position(wall, bk_wall)
        if self.reset_to_backup and bk_wall != self.target_position:
            self.wall = bk_wall
            self.wall_parallel = Obstacles.OBS_PARALLEL_FULL

    in_lost_detect_wall_time = time.time()

    def _update_wall_position(self, wall, bk_wall):
        """壁の位置に基づいて状態を更新"""
        if wall == Obstacles.OBS_CENTER:
            self.wall = "BOTH"
        elif wall == Obstacles.OBS_LEFT:
            self.wall = Position.LEFT
        elif wall == Obstacles.OBS_RIGHT:
            self.wall = Position.RIGHT
        else:
            self.wall = Position.NONE

        skip_parallel = (
            self.lost_target_command != Commands.STOP_TEMP
            and self.lost_target_command != Position.convert_to_rotate(self.wall)
        )

        if (
            self.wall == self.last_wall_detect
            and self.wall == bk_wall
            or self.wall == "BOTH"
        ):
            self.last_wall_detect_time = time.time()

        if not self.tracking_target_invisible or (
            (self.avoid_wall or self.wall_parallel) and self.tracking_target_invisible
        ):
            self.last_wall_avoid_time = time.time()

        if self.avoid_wall or (self.wall_parallel and not skip_parallel):
            self.last_wall_detect = self.wall

            if self.avoid_wall:
                self.last_wall_avoid = self.wall

            if self.target_position != Position.NONE and self.wall == Position.NONE:
                self.last_wall_avoid = Position.NONE
                self.last_wall_detect = Position.NONE

        return False

    def _control_motor_for_target(self):
        """対象の位置に基づいてモーターを制御"""
        base_speed = self._calculate_base_speed()
        self.motor_power_l = base_speed
        self.motor_power_r = base_speed

        if self.target_x != -1:
            self.calculate_motor_power(self.target_x)

        skip_parallel = self.target_position != Position.CENTER and (
            Position.convert_to_rotate(self.wall) != self.lost_target_command
            and self.lost_target_command != Commands.STOP_TEMP
            or (
                self.tracking_target_invisible
                and self.lost_target_command == Commands.STOP_TEMP
            )
        )

        if (self.wall != Obstacles.OBS_NONE and self.avoid_wall) or (
            Obstacles.is_fix(self.wall_parallel) and not skip_parallel
        ):
            self.apply_motor_wall(self.wall)

        if self.wall_parallel == Obstacles.OBS_PARALLEL_FULL and not skip_parallel:
            base_speed = self._calculate_base_speed()
            self.motor_power_l = base_speed
            self.motor_power_r = base_speed

    _prev_send_rfid_command = Commands.CHECK

    def _send_target_position_if_rfid_mode(self):
        """RFIDモードの場合、対象の方向を送信"""
        if (
            self.RFID_ONLY_MODE
            and self.RFID_ENABLED
            and self.gui.var_enable_tracking.get()
            and not self.stop_exec_cmd
            and not self.stop_exec_cmd_gui
        ):
            max_power = 100
            min_power = 0
            self.target_position = self.rfid_reader.get_direction()
            self.motor_power_l = max_power
            self.motor_power_l = max_power
            if (
                self._prev_send_rfid_command == Commands.STOP_TEMP
                and self.target_position == Commands.STOP_TEMP
            ):
                return
            self.stop_temp = False
            self._prev_send_rfid_command = self.target_position
            if self.target_position == Commands.GO_LEFT:
                self.motor_power_l = min_power
                self.lost_target_command = Commands.ROTATE_LEFT
            elif self.target_position == Commands.GO_RIGHT:
                self.motor_power_r = min_power
                self.lost_target_command = Commands.ROTATE_RIGHT
            elif self.target_position == Commands.STOP_TEMP:
                self.send(Commands.STOP_TEMP)
            elif Commands.is_rotate(self.target_position):
                self.stop_exec_cmd = True
                self.find_target_start_time = time.time()
                self.send(self.target_position)
                self.lost_target_command = self.target_position
            elif self.target_position == Commands.GO_CENTER:
                self.lost_target_command = Commands.STOP_TEMP

    def _send_default_speed_if_changed(self):
        """デフォルト速度に変更がある場合に送信"""
        if self.gui.var_slow.get() or self.mode == self.Mode.RFID_ONLY.name:
            self.default_speed = DEBUG_SLOW_SPEED
        if self._def_spd_bk != self.default_speed:
            self.send((Commands.SET_DEFAULT_SPEED, self.default_speed))
            self._def_spd_bk = self.default_speed

    def send_commands_if_needed(self):
        """送信できるならモーター情報やCHECKを送信"""
        current_time = time.time()
        if (
            current_time - self.interval_serial_send_start_time
            > ROBOT_SERIAL_SEND_INTERVAL
        ):
            self.sent_count += 1
            self.interval_serial_send_start_time = current_time
            self.send(Commands.CHECK)

        # 障害物が目の前なら、後退して転回
        if (
            self.auto_stop_obs
            and self.is_close_obs
            and not self.stop_exec_cmd
            and self.gui.var_enable_tracking.get()
            and not (
                self.target_position == Position.CENTER
                or self.lost_target_command == Commands.STOP_TEMP
            )
        ):
            self.stop_exec_cmd = True
            self.obs_backoff_start_time = time.time()
            self.send(Commands.GO_BACK)

        if (
            self.stop_exec_cmd
            and self.obs_backoff_start_time is not None
            and time.time() - self.obs_backoff_start_time > 1.5
            and self.gui.var_enable_tracking.get()
        ):
            self.obs_backoff_start_time = None
            self.exec_lost_target_command()

        if (
            current_time - self.interval_serial_send_motor_start_time
            > ROBOT_SERIAL_SEND_MOTOR_INTERVAL
        ):
            self.interval_serial_send_motor_start_time = current_time
            if (
                not self.stop
                and not self.stop_temp
                and not self.stop_exec_cmd
                and not self.stop_exec_cmd_gui
                and self.gui.var_enable_tracking.get()
            ):
                self.send((Commands.L_SPEED, self.motor_power_l))
                self.bkl = self.motor_power_l
                self.send((Commands.R_SPEED, self.motor_power_r))
                self.bkr = self.motor_power_r

        if current_time - self.send_check_start_time > 1:
            self.sent_count = 0
            self.send_check_start_time = current_time

    def receive_serial_data(self):
        """シリアル通信からデータを受信"""
        received = self.serial.get_received_queue()
        if received:
            self.gui.queue.add_all("r", received)
            self.received_serial = received[-1]

    def update_fps(self, frame_start_time):
        """FPS（フレーム毎秒）を更新"""
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
        """デバッグ情報を更新し、フレームに描画"""
        delay_find_map = self.map_value_to_scale(self.default_speed)
        delay_find = delay_find_map - (time.time() - self.last_wall_avoid_time)

        delay = time.time() - self.last_wall_detect_time

        rfid_power = 0
        if self.rfid_reader:
            rfid_power = self.rfid_reader.signal_strength

        center = (
            self.obstacles.is_target_in_center_area(self.target.bboxes)
            if self.target
            else False
        )

        debug_text = (
            f"{self.seg} {math.floor(self.frame_width)} x {math.floor(self.frame_height)} "
            f"{'O' if self.target_position != 'X' and self.target_position != Commands.STOP_TEMP else 'X'} "
            f"avr_fps: {math.floor(self.avr_fps)} fps: {math.floor(self.fps)} "
            f"target_position: {self.target_position}\n"
            f"target_x: {self.target_x} mpl: {self.motor_power_l} mpr: {self.motor_power_r} bkl: {self.bkl} bkr: {self.bkr} in_center: {center}\n"
            f"connected: {self.serial.ser_connected} port: {ROBOT_SERIAL_PORT} baud: {ROBOT_SERIAL_BAUD} data: {self.serial.serial_sent}\n"
            f"ip: {self.serial.tcp_ip} STOP: {self.stop or self.stop_temp} serial_received: {self.received_serial}\n"
            f"obstacle_detected: {self.obs_detected} obs_pos: {self.obs_pos} wall: {self.wall} stop_obs: {self.auto_stop_obs}\n"
            f"avoid: {self.avoid_wall} no_detection: {round(time.time() - self.target_last_seen_time, 2):.2f} "
            f"detecting: {round(self.time_detected, 2):.2f} track_out: {self.tracking_target_invisible}\n"
            f"RFID: {self.RFID_ENABLED} RFID_only: {self.RFID_ONLY_MODE} RFID_Power: {rfid_power} def_spd: {self.default_speed} parallel: {self.wall_parallel}\n"
            f"auto_stop: {self.auto_stop} close: {self.is_close_obs} occupancy: {self.occupancy_ratio:.2%} exec_stop: {self.stop_exec_cmd}\n"
            f"no_wall: {delay:.2f} last_wall_detect: {self.last_wall_detect} close_wall: {self.too_close_wall} closest: {self.closest_wall_depth} lost: {self.lost_target_command}\n"
            f"map: {delay_find_map:.2f} dfind: {delay_find:.2f} rb_wall: {self.reset_to_backup} w_avoid: {self.last_wall_avoid} stmp: {self.stop_temp} find: {self.find_target_rotate}\n"
        )
        self.print_d(debug_text)
        self.print_key_binds()
        if self.stop:
            self.print_stop()

    def update_gui(self):
        if self.frame is not None:
            self.gui.update_frame(self.frame, self.depth_frame)

    def close(self):
        self.stop_motor()
        self.serial.close()

        if DEBUG_USE_ONLY_WEBCAM:
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

        self.video_capture = cv2.VideoCapture(WEB_CAM_NO)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.frame_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.frame = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255
        self.depth_frame = np.ones((0, 0, 3), dtype=np.uint8) * 255

        if self.RFID_ENABLED:
            logger.info("Initializing RFID Reader...")
            self.rfid_reader = RFIDReader(port=RFID_SERIAL_PORT)
            logger.info(
                f"RFID Antenna Status: {self.rfid_reader.check_antenna_status()}"
            )
            self.rfid_reader.start_reading()

        self.target_detection_start_time = None
        self.time_detected = 0
        self.prev_command = Commands.STOP_TEMP

        self.max_motor_power_threshold = self.frame_width / 4
        self.motor_power_l, self.motor_power_r = 0, 0

        self.interval_serial_send_start_time = 0
        self.interval_serial_send_motor_start_time = 0
        self.send_check_start_time = 0
        self.occupancy_ratio = 0
        self.is_close_obs = False

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

    @classmethod
    def is_process_running(cls, process_name):
        """指定したプロセスが実行中か確認"""
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline")
                if cmdline and process_name in cmdline:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    @classmethod
    def run_robot_if_needed(cls):
        """必要に応じてロボット側のプログラムを起動"""
        if not cls.is_process_running("tracker_robot_side.py"):
            logger.info("Robot process is not running. Starting...")
            subprocess.Popen(["python", "tracker_robot_side.py"])


def main():
    Tracker.run_robot_if_needed()
    tracker = Tracker()
    logger.info("Initializing GUI")
    tracker.root = tk.Tk()
    tracker.gui = gui.GUI(tracker, tracker.root)
    tracker.gui.queue.wait_for("init")
    tracker.init()
    tracker.start()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error(traceback.format_exc())
