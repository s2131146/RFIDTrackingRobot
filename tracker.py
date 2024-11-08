import os
from typing import Tuple

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

from constants import Commands
from rfid import RFIDReader
import tracker_socket as ts
import gui
import logger as l

from obstacles import Obstacles

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

DEBUG_SLOW_MOTOR = True
DEBUG_SERIAL = True
DEBUG_USE_ONLY_WEBCAM = False
DEBUG_DETECT_OBSTACLES = True

# Cascade 関連のフラグとリストを削除

logger = l.logger


class Tracker:
    """Trackerクラス: 人物検出と距離推定（画像面積ベース）、モーター制御を行う"""

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
            DEBUG_USE_ONLY_WEBCAM,
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
        self.auto_stop_obs = False

        self.RFID_ENABLED = False
        self.RFID_ONLY_MODE = True

        # YOLOモデルのロード
        self.model = YOLO("models\\yolov8n.pt")

        # 障害物検出クラスのインスタンス作成
        self.obstacles = Obstacles(FRAME_WIDTH, FRAME_HEIGHT, logger)

        from target_processor import TargetProcessor

        # 対象検出クラスのインスタンス作成
        self.target_processor = TargetProcessor(
            FRAME_WIDTH, FRAME_HEIGHT, self.model, logger, self
        )

        self.stop_temp = False
        self.detected_obstacles = []
        self.occupancy_ratio = 0
        self.is_close = False

        self.target_position_str = "X"
        self.target_x = -1
        self.obs_pos = Obstacles.OBS_POS_NONE
        self.wall = None
        self.avoid_wall = False

        self.last_target_direction = (
            None  # ターゲットが最後にいた方向 ('LEFT' または 'RIGHT')
        )
        self.following_wall = False  # 壁沿い移動中かどうか
        self.wall_follow_direction = None  # 壁沿い移動の方向 ('LEFT' または 'RIGHT')

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
        overlay = self.frame.copy()  # 元のフレームをコピーしてオーバーレイを作成
        alpha = 0.6  # 半透明の背景の透明度（0.0は完全透明、1.0は完全不透明）

        text_lines = text.split("\n")
        for index, line in enumerate(text_lines, start=1):
            # 文字のサイズを取得
            (text_width, text_height), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1
            )

            # 背景の四角形の座標を計算
            top_left = (5, 15 * index - text_height - baseline // 2)
            bottom_right = (5 + text_width, 15 * index + baseline)

            # 半透明の白い背景をオーバーレイに描画
            cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), -1)

            # 文字をオーバーレイに描画
            cv2.putText(
                overlay,
                line,
                (5, 15 * index),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        # 元のフレームとオーバーレイを合成して半透明の背景を作成
        cv2.addWeighted(overlay, alpha, self.frame, 1 - alpha, 0, self.frame)

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
            self.motor_power_l = 0
            self.motor_power_r = 0
            self.lost_target_command = Commands.STOP_TEMP
            self.gui.update_stop()
            logger.info("Motor stopped")
        elif c == Commands.START:
            self.start_motor()
            logger.info("Motor started")
        elif c == Commands.STOP_TEMP:
            self.motor_power_l = 0
            self.motor_power_r = 0
            self.stop_temp = True

    def is_obstacle_cleared(self):
        """障害物がクリアされたかを判断"""
        # フレーム内に障害物が検出されていない場合
        return not self.obs_detected

    def turn_towards_target(self):
        """障害物回避後に対象に向かってターンする"""
        if self.lost_target_command == Commands.ROTATE_LEFT:
            # 左折
            self.motor_power_l = 50  # 左モーターを低速に
            self.motor_power_r = 100  # 右モーターを高速に
        elif self.lost_target_command == Commands.ROTATE_RIGHT:
            # 右折
            self.motor_power_l = 100  # 左モーターを高速に
            self.motor_power_r = 50  # 右モーターを低速に
        else:
            # 中央の場合
            self.motor_power_l = self.default_speed
            self.motor_power_r = self.default_speed

    def apply_motor_wall(self, wall_direction: str) -> Tuple[int, int]:
        """
        壁の位置に基づき、壁沿いに進むためのモーター出力を計算します。

        Args:
            wall_direction (str): 壁の方向（LEFT, RIGHT, CENTER, NONE）

        Returns:
            Tuple[int, int]: 左右のモーターの出力（百分率）
        """
        if wall_direction == Obstacles.OBS_POS_LEFT:
            # 壁が左側にある場合、右に寄って進む
            self.motor_power_l = 50
            self.motor_power_r = 100
        elif wall_direction == Obstacles.OBS_POS_RIGHT:
            # 壁が右側にある場合、左に寄って進む
            self.motor_power_l = 100
            self.motor_power_r = 50

    def calculate_motor_power(self, target_x=0):
        """モーター出力を計算

        Args:
            target_x (int): 対象座標X
        """
        # 障害物が検出されている場合の処理
        if self.obs_detected:
            self._adjust_for_obstacles(target_x)
            return

        # 基礎速度の計算
        base_speed = self._calculate_base_speed()

        # 対象の位置に基づくモーター出力の調整
        self._adjust_motor_power_based_on_target_position(target_x, base_speed)

    def _adjust_for_obstacles(self, target_x):
        """障害物が検出された場合にモーター出力を調整する

        Args:
            target_x (int): 対象座標X
        """
        if self.obs_pos == Obstacles.OBS_POS_LEFT:
            # 左側に障害物がある場合、対象を右に移動
            target_x = 160 if target_x > 0 else -160
        elif self.obs_pos == Obstacles.OBS_POS_RIGHT:
            # 右側に障害物がある場合、対象を左に移動
            target_x = -160 if target_x < 0 else 160
        elif self.obs_pos == Obstacles.OBS_POS_CENTER:
            # 中央に障害物がある場合、左回転して停止
            self.send(Commands.ROTATE_LEFT)
            self.stop_exec_cmd = True

    def _calculate_base_speed(self):
        """占有率に基づいて基礎速度を計算する

        Returns:
            int: 基礎速度
        """
        # 占有率と基礎速度の対応ポイント
        occupancy_thresholds = [0.0, 0.15, 0.40, 0.60, 1.0]
        base_speeds = [100, 100, 80, 60, 0]
        occupancy = self.occupancy_ratio

        # 線形補間による基礎速度の計算
        base_speed = np.interp(occupancy, occupancy_thresholds, base_speeds)
        # 速度をクランプして 0 〜 100 に制限
        return max(0, min(100, base_speed))

    def _adjust_motor_power_based_on_target_position(self, target_x, base_speed):
        """対象の位置に基づいてモーター出力を調整する

        Args:
            target_x (int): 対象座標X
            base_speed (int): 基礎速度
        """
        # 中央15%以内の閾値
        central_threshold = 0.15 * self.frame_width

        if abs(target_x) <= central_threshold:
            # 中央15%以内に対象がある場合、左右のモーターに同じ基礎速度を設定
            self.motor_power_l = base_speed
            self.motor_power_r = base_speed
        else:
            # 中央15%を超える場合、対象の位置に基づき調整
            self._calculate_steering_power(target_x, base_speed)

    def _calculate_steering_power(self, target_x, base_speed):
        """左右のモーター出力を調整して旋回を行う

        Args:
            target_x (int): 対象座標X
            base_speed (int): 基礎速度
        """
        # フレームの半幅で対象の位置を正規化
        half_width = self.frame_width / 2
        normalized_error = target_x / (0.85 * half_width)  # 0.85は調整可能

        # 正規化されたエラーを -1.0 ~ 1.0 にクランプ
        normalized_error = max(-1.0, min(1.0, normalized_error))

        # 方向調整の強さ
        steer_strength = 50

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

        # モーター出力を整数値に丸める
        self.motor_power_l = int(self.motor_power_l)
        self.motor_power_r = int(self.motor_power_r)

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
        self.gui.root.after(
            0,
            self.gui.update_motor_values,
            int(self.motor_power_l),
            int(self.motor_power_r),
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
        self.target_x = -1
        self.obs_detected = False

    def capture_frame(self):
        if self.RFID_ONLY_MODE and self.RFID_ENABLED:
            self.frame = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255
        else:
            ret, frame = self.video_capture.read()
            if not ret:
                logger.warning("Failed to read frame from webcam.")
                self.frame = (
                    np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255
                )
            else:
                self.frame = cv2.flip(frame, 1)

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
        """障害物およびターゲットの処理を行う"""
        current_time = time.time()

        # ターゲット検出の処理
        detected_targets = self._detect_and_select_targets(current_time)

        # RFID処理
        self._process_rfid_logic(detected_targets)

        # 障害物処理
        self._process_obstacles(detected_targets)

        # モーター制御
        self._control_motor_for_target()

        # RFIDモードでのターゲット方向送信
        self._send_target_position_if_rfid_mode()

        # 速度変更がある場合の処理
        self._send_default_speed_if_changed()

    def _detect_and_select_targets(self, current_time):
        """ターゲットを検出して選択する処理"""
        detected_targets = self.target_processor.detect_targets(
            self.frame, self.RFID_ONLY_MODE, self.RFID_ENABLED
        )
        self.rfid_accessing = False
        self.stop_exec_cmd = False

        selected_target = None

        if not self.RFID_ONLY_MODE:
            if len(detected_targets) > 1:
                # 複数ターゲットの選択処理
                selected_target = self._select_and_process_multiple_targets(
                    detected_targets
                )
            elif len(detected_targets) == 1:
                # 単一ターゲットの処理
                selected_target = self._process_single_target(
                    detected_targets, current_time
                )
            elif len(detected_targets) == 0:
                # ターゲットなしの場合の処理
                self._handle_no_targets(current_time)

        return [selected_target] if selected_target else []

    def _select_and_process_multiple_targets(self, detected_targets):
        """複数のターゲットを検出した場合の処理"""
        selected_target = self.target_processor.select_target(detected_targets)
        if selected_target:
            result = self.target_processor.process_target([selected_target], self.frame)
            if result is not None:
                target_center_x, self.target_x, _ = result
                self.target_position_str = self.get_target_pos_str(target_center_x)
                self.target_processor.draw_all_targets(
                    detected_targets, selected_target, self.frame
                )

        if self.RFID_ENABLED:
            self.execute_rfid_direction()

        return selected_target

    def _process_single_target(self, detected_targets, current_time):
        """単一のターゲットを検出した場合の処理"""
        target = detected_targets[0]
        if self.target_detection_start_time is None:
            self.target_detection_start_time = current_time
        else:
            self.time_detected = current_time - self.target_detection_start_time
            if self.time_detected >= 0.3:
                self.target_last_seen_time = current_time
                self.stop_temp = False
                result = self.target_processor.process_target([target], self.frame)
                if result is not None:
                    target_center_x, self.target_x, _ = result
                    self.target_position_str = self.get_target_pos_str(target_center_x)

        return target

    def _handle_no_targets(self, current_time):
        """ターゲットが検出されなかった場合の処理"""
        if self.target_detection_start_time is not None:
            self.target_detection_start_time = None

        time_since_last_seen = current_time - self.target_last_seen_time
        if time_since_last_seen > 1.0 and not self.stop_temp:
            if self.RFID_ENABLED:
                self.execute_rfid_direction()
            else:
                if not self.stop:
                    self.send(self.lost_target_command)
                    self.stop_exec_cmd = True
                    self.motor_power_l = 0
                    self.motor_power_r = 0

    def _process_rfid_logic(self, detected_targets):
        """RFIDロジックの処理"""
        if len(detected_targets) == 0 and self.RFID_ENABLED:
            self.execute_rfid_direction()

    def _process_obstacles(self, detected_targets):
        """障害物処理を行う"""
        if not DEBUG_USE_ONLY_WEBCAM and DEBUG_DETECT_OBSTACLES:
            result = self.obstacles.process_obstacles(
                self.serial.depth_image, self.target_x, detected_targets
            )
            if len(result) == 2:
                if not self.auto_stop_obs:
                    self.stop_motor()
                    self.auto_stop_obs = True
                _, self.depth_frame = result
                return
            else:
                self.auto_stop_obs = False
                (
                    self.obs_pos,
                    self.obs_detected,
                    self.depth_frame,
                    wall,
                    self.avoid_wall,
                ) = result

        # 壁位置の設定
        self._update_wall_position(wall)

    def _update_wall_position(self, wall):
        """壁の位置に基づいて状態を更新"""
        if wall == Obstacles.OBS_POS_CENTER:
            self.wall = "BOTH"
        elif wall == Obstacles.OBS_POS_LEFT:
            self.wall = "LEFT"
        elif wall == Obstacles.OBS_POS_RIGHT:
            self.wall = "RIGHT"
        else:
            self.wall = "NONE"

    def _control_motor_for_target(self):
        """ターゲット位置に基づいてモーターを制御"""
        if self.target_x != -1:
            self.calculate_motor_power(self.target_x)

        # 壁を回避する場合
        if self.wall != Obstacles.OBS_POS_NONE and self.avoid_wall:
            self.apply_motor_wall(self.wall)

    def _send_target_position_if_rfid_mode(self):
        """RFIDモード時にターゲット方向を送信"""
        if self.RFID_ONLY_MODE and self.RFID_ENABLED:
            self.target_position_str = self.rfid_reader.get_direction()
            if not self.stop:
                self.send(self.target_position_str)

    def _send_default_speed_if_changed(self):
        """デフォルト速度に変更がある場合に送信"""
        if DEBUG_SLOW_MOTOR:
            self.default_speed = 100
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
            f"obstacle_detected: {self.obs_detected} obs_pos: {self.obs_pos} wall: {self.wall} stop_obs: {self.auto_stop_obs}\n"
            f"avoid: {self.avoid_wall} no_detection: {round(time.time() - self.target_last_seen_time, 2):.2f} "
            f"detecting: {round(self.time_detected, 2):.2f}\n"
            f"self.RFID_ENABLED: {self.RFID_ENABLED} RFID_only: {self.RFID_ONLY_MODE} def_speed: {self.default_speed}\n"
            f"auto_stop: {self.auto_stop} close: {self.is_close} occupancy: {self.occupancy_ratio:.2%}"
        )
        self.print_d(debug_text)
        self.print_key_binds()
        if self.stop:
            self.print_stop()

    def update_gui(self):
        if self.frame is not None:
            self.gui.update_frame(self.frame, self.depth_frame)

    def close(self):
        logger.info("Closing application")
        self.stop_motor()
        self.obstacles.close()
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

        self.frame_width = FRAME_WIDTH
        self.frame_height = FRAME_HEIGHT

        if not self.RFID_ONLY_MODE:
            self.video_capture = cv2.VideoCapture(CAM_NO)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            self.frame_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.frame_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.frame = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255
        self.depth_frame = np.ones((0, 0, 3), dtype=np.uint8) * 255

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
