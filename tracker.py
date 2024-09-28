import os
import threading

# カメラ起動高速化
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import time
import math
import tkinter as tk
import numpy as np
import asyncio

from scipy.cluster.hierarchy import fclusterdata

from constants import Commands
from constants import Cascades
from rfid import RFIDReader
import tracker_socket as ts
import gui
import logger as l

# シリアルポートの設定
SERIAL_PORT = "COM3"
SERIAL_BAUD = 19200
SERIAL_SEND_INTERVAL = 0.03

TCP_PORT = 8001

# キャプチャウィンドウサイズ
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CAM_NO = 0

DEBUG_SERIAL = True
DEBUG_USE_WEBCAM = False

logger = l.logger


class Tracker:
    """申し訳ないレベルのグローバル変数"""

    root = app = cascade = video_capture = None
    frame_width = frame_height = d_x = d_y = face_x = face_center_x = None
    max_motor_power_threshold = motor_power_l = motor_power_r = None
    interval_serial_send_start_time = seg = serial_sent = None

    serial = ts.TrackerSocket(
        SERIAL_PORT, SERIAL_BAUD, SERIAL_SEND_INTERVAL, DEBUG_SERIAL, TCP_PORT
    )

    stop = False
    target_last_seen_time = None  # 対象が最後に検出された時刻

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
            frame (MatLike): フレーム
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

    def detect_target(self):
        """画像から対象を検出

        Args:
            frame (MatLike): フレーム

        Returns:
            Sequence[Rect]: 座標情報
        """
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        target = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return target

    def send(self, data):
        if not Commands.contains(self.serial.get_command(data)):
            logger.warning(f"Invalid command: {data}")
            return False

        ret = self.serial.send_serial(data)
        if ret:
            self.print_serial_stat()
        self.exec_sent_command()

        if not Commands.is_ignore(data):
            self.app.queue.add("s", self.serial.serial_sent)

        return True

    def exec_sent_command(self):
        c = self.serial.command_sent.upper()
        if c == Commands.STOP:
            self.stop = True
            self.app.update_stop()
            logger.info("Motor stopped")
        if c == Commands.START:
            self.start_motor()
            logger.info("Motor started")

    def calculate_motor_power(self, target_x):
        """モーター出力を計算

        Args:
            target_x (int): 対象座標X
        """
        if target_x < 0:
            self.motor_power_l = math.floor(math.sqrt(-target_x))
            self.motor_power_r = math.floor(math.sqrt(-(target_x * 4)))
            self.motor_power_l -= self.motor_power_r
        else:
            self.motor_power_l = math.floor(math.sqrt(target_x * 4))
            self.motor_power_r = math.floor(math.sqrt(target_x))
            self.motor_power_r -= self.motor_power_l

        self.motor_power_l += 70
        self.motor_power_r += 70

        if self.motor_power_l > 100:
            self.motor_power_l = 100
        if self.motor_power_r > 100:
            self.motor_power_r = 100

    def get_target_pos_str(self, target_center_x):
        """対象のいる位置を文字列で取得

        Args:
            target_center_x (int): 対象のX座標

        Returns:
            str: 結果
        """
        if target_center_x < self.frame_width // 3:
            target_position_str = "LEFT"
        elif target_center_x < 2 * self.frame_width // 3:
            target_position_str = "CENTER"
        else:
            target_position_str = "RIGHT"

        return target_position_str

    def calculate_motor_power_with_distance(self, target_x, distance):
        """モーター出力を距離に基づいて計算します。

        Args:
            target_x (int): 対象のX座標
            distance (int): 対象までの距離 (mm)
        """
        # 距離に応じた速度係数を設定
        if distance is not None:
            if distance > 300:
                speed_factor = 1.0  # 最大速度
            elif distance > 200:
                speed_factor = 0.7  # 速度70%
            elif distance > 100:
                speed_factor = 0.4  # 速度40%
            else:
                speed_factor = 0.0  # 停止
        else:
            speed_factor = 1.0  # デフォルト

        # 既存のターゲットXに基づくモーター出力計算
        self.calculate_motor_power(target_x)

        # 速度係数を適用
        self.motor_power_l = int(self.motor_power_l * speed_factor)
        self.motor_power_r = int(self.motor_power_r * speed_factor)

        # モーター出力の下限を0に設定
        self.motor_power_l = max(self.motor_power_l, 0)
        self.motor_power_r = max(self.motor_power_r, 0)

        if speed_factor == 0.0:
            self.stop_motor(True)
        else:
            self.send_commands_if_needed()

    def process_target(self, target):
        """画像中の対象を囲み、中心座標と距離を取得

        Args:
            target (Sequence[Rect]): 座標情報

        Returns:
            target_center_x, target_x, distance: 顔座標原点X, 顔座標X, 距離
        """
        for x, y, w, h in target:
            self.d_x = x
            self.d_y = y
            self.target_detected = True

            target_center_x = x + w // 2
            target_x = math.floor(target_center_x - self.frame_width / 2)

            # 深度画像から対象の距離を取得
            if hasattr(self, "depth_frame") and self.depth_frame is not None:
                depth_pixels = self.depth_frame[y : y + h, x : x + w]
                if depth_pixels.size > 0:
                    # ノイズを減らすために中央値を使用
                    median_depth = int(np.median(depth_pixels))
                else:
                    median_depth = None
            else:
                median_depth = None

            # 対象を矩形で囲む
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text_org = (x, y - 10)
            if median_depth is not None:
                distance_text = f"{median_depth}mm"
            else:
                distance_text = ""
            cv2.putText(
                self.frame,
                f"TARGET {distance_text}",
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            return target_center_x, target_x, median_depth

        return None, None, None

    def stop_motor(self, no_target=False):
        if no_target:
            self.send(Commands.STOP_TEMP)
        else:
            self.send(Commands.STOP)

    def start_motor(self):
        if self.serial.ser_connected:
            self.stop = False
            self.app.update_stop()

    detected_obstacles = []

    def process_obstacles(self, frame, depth_image):
        """障害物を検知し、描画

        Args:
            frame (MatLike): フレーム
            depth_image: 深度フレーム

        Returns:
            frame, no_obstacle_detected, obstacle_free_center_x: 描画後フレーム, 障害物がないか, 安全なX座標
        """
        roi_depth, roi_color = self.get_roi(frame, depth_image)
        roi_depth_blurred = self.preprocess_depth(roi_depth)
        obstacle_image = self.create_obstacle_mask(roi_depth_blurred)
        obstacle_image_processed = self.apply_morphology(obstacle_image)
        contours = self.find_obstacle_contours(obstacle_image_processed)

        center_x = roi_color.shape[1] // 2

        no_obstacle_detected, obstacle_free_center_x = self.process_contours(
            contours, roi_color, center_x
        )
        return frame, no_obstacle_detected, obstacle_free_center_x

    def process_contours(self, contours, roi_color, center_x):
        """輪郭を処理して情報を取得する関数"""
        no_obstacle_detected = True
        detected_obstacles_info = []

        # 各輪郭のバウンディングボックスを取得
        bounding_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 1500:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append([x, y, x + w, y + h])

        # バウンディングボックスを近いもの同士で統合
        bounding_boxes = self.combine_close_bounding_boxes(bounding_boxes)

        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            obstacle_center_x = x1 + w // 2
            distance_to_center = obstacle_center_x - center_x

            detected_obstacles_info.append((x1, y1, w, h, distance_to_center))
            no_obstacle_detected = False

            # 障害物を描画
            cv2.rectangle(roi_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text_org = (x1, y1 - 10)
            cv2.putText(
                roi_color,
                "OBSTACLE X:{}".format(distance_to_center),
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        # 障害物のX座標を計算
        if detected_obstacles_info:
            all_obstacle_x_positions = [
                info[0] + info[2] // 2 for info in detected_obstacles_info
            ]
            min_obstacle_x = min(all_obstacle_x_positions)
            max_obstacle_x = max(all_obstacle_x_positions)
            obstacle_free_center_x = (min_obstacle_x + max_obstacle_x) // 2
        else:
            obstacle_free_center_x = center_x

        obstacle_free_center_x = math.floor(
            obstacle_free_center_x - self.frame_width / 2
        )

        return no_obstacle_detected, obstacle_free_center_x

    def combine_close_bounding_boxes(self, boxes, distance_threshold=50):
        """近接するバウンディングボックスを統合します。

        Args:
            boxes (List[List[int]]): バウンディングボックスのリスト
            distance_threshold (int): ボックス間の最大距離

        Returns:
            List[List[int]]: 統合後のバウンディングボックスのリスト
        """
        if not boxes:
            return []

        # ボックスの中心点を計算
        centers = np.array(
            [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes]
        )

        # クラスタリングによって近いボックスをグループ化
        if len(centers) == 1:
            clusters = np.array([1])
        else:
            clusters = fclusterdata(centers, t=distance_threshold, criterion="distance")

        combined_boxes = []
        for cluster_id in np.unique(clusters):
            cluster_boxes = [
                boxes[i] for i in range(len(boxes)) if clusters[i] == cluster_id
            ]
            x1 = min([box[0] for box in cluster_boxes])
            y1 = min([box[1] for box in cluster_boxes])
            x2 = max([box[2] for box in cluster_boxes])
            y2 = max([box[3] for box in cluster_boxes])
            combined_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        return combined_boxes

    def get_roi(self, frame, depth_image):
        """ROI（領域）を取得する関数"""
        roi_depth = depth_image[: int(depth_image.shape[0] * 4 / 5), :]
        roi_color = frame[: int(frame.shape[0] * 4 / 5), :]
        return roi_depth, roi_color

    def preprocess_depth(self, roi_depth):
        """深度ROIを前処理する関数"""
        roi_depth_blurred = cv2.GaussianBlur(roi_depth, (5, 5), 0)
        return roi_depth_blurred

    def create_obstacle_mask(self, roi_depth):
        """障害物マスクを作成する関数"""
        max_distance = 300  # 300mm
        min_distance = 100  # 100mm
        mask = np.logical_and(roi_depth > min_distance, roi_depth < max_distance)
        obstacle_image = np.zeros_like(roi_depth, dtype=np.uint8)
        obstacle_image[mask] = 255
        return obstacle_image

    def apply_morphology(self, obstacle_image):
        """モルフォロジー処理を適用する関数"""
        kernel = np.ones((5, 5), np.uint8)
        obstacle_image_processed = cv2.morphologyEx(
            obstacle_image, cv2.MORPH_CLOSE, kernel
        )
        obstacle_image_processed = cv2.morphologyEx(
            obstacle_image_processed, cv2.MORPH_OPEN, kernel
        )
        return obstacle_image_processed

    def find_obstacle_contours(self, obstacle_image):
        """障害物の輪郭を検出する関数"""
        contours, _ = cv2.findContours(
            obstacle_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def main_loop(self):
        """メインループを開始します。"""
        logger.info("Starting main loop")
        self.initialize_loop_variables()

        while True:
            self.update_seg()
            frame_start_time = time.time()
            self.reset_detection_flags()
            self.capture_frame()
            self.process_obstacles_and_targets()
            self.update_motor_power()
            self.update_rfid_power()
            self.send_commands_if_needed()
            self.receive_serial_data()
            self.update_fps(frame_start_time)
            self.handle_disconnection()
            self.update_debug_info()
            self.update_gui()

    def initialize_loop_variables(self):
        """ループ内で使用する変数を初期化します。"""
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

    def update_motor_power(self):
        self.app.root.after(
            0, self.app.update_motor_values, self.motor_power_l, self.motor_power_r
        )

    def update_rfid_power(self):
        power = self.rfid_reader.get_predicted_power()
        self.app.update_rfid_values([power, 0, 0, 0])

    def update_seg(self):
        self.seg += 1
        if self.seg > 9:
            self.seg = 0
        self.app.update_seg(self.seg)

    def reset_detection_flags(self):
        """検出に関するフラグをリセットします。"""
        self.target_detected = False
        self.target_position_str = "X"
        self.target_x = 0
        self.no_obs = True
        self.safe_x = 0

    def capture_frame(self):
        """フレームをキャプチャします。"""
        if DEBUG_USE_WEBCAM:
            _, frame = self.video_capture.read()
            self.frame = cv2.flip(frame, 1)
        else:
            self.frame = np.copy(self.serial.color_image)
            self.depth_frame = np.copy(self.serial.depth_image)

    def process_obstacles_and_targets(self):
        """障害物と対象物の検出および処理を行います。"""
        if not DEBUG_USE_WEBCAM:
            self.frame, self.no_obs, self.safe_x = self.process_obstacles(
                self.frame, self.depth_frame
            )

        targets = self.detect_target()
        self.bkl = self.motor_power_l
        self.bkr = self.motor_power_r

        if len(targets) > 0:
            # 対象が検出されたので、最後に検出された時刻を更新
            self.target_last_seen_time = time.time()
            self.stop_temp = False
            result = self.process_target(targets)
            if result != (None, None, None):
                target_center_x, self.target_x, target_distance = result
                self.target_position_str = self.get_target_pos_str(target_center_x)
                self.calculate_motor_power_with_distance(self.target_x, target_distance)
        else:
            # 対象が検出されなかった場合
            current_time = time.time()
            time_since_last_seen = current_time - self.target_last_seen_time
            if time_since_last_seen > 1.0:  # 1秒以上対象が検出されなかった場合
                if not self.stop_temp:
                    self.stop_motor(True)
                    self.stop_temp = True

        if not self.no_obs:
            self.calculate_motor_power(-self.safe_x)

    def send_commands_if_needed(self):
        """必要に応じてコマンドを送信します。"""
        current_time = time.time()
        if current_time - self.interval_serial_send_start_time > SERIAL_SEND_INTERVAL:
            self.sent_count += 1
            self.interval_serial_send_start_time = current_time

            if self.app.var_enable_tracking.get() and not self.stop:
                if self.bkl != self.motor_power_l:
                    self.send((Commands.SET_SPD_LEFT, self.motor_power_l))
                if self.bkr != self.motor_power_r:
                    self.send((Commands.SET_SPD_RIGHT, self.motor_power_r))

                if self.target_position_str == "X":
                    self.app.queue.add("g", "Target not found")
                else:
                    self.app.queue.add(
                        "g",
                        "GO:{} | L:{}% R:{}%".format(
                            self.target_position_str,
                            self.motor_power_l,
                            self.motor_power_r,
                        ),
                    )

            self.send(Commands.CHECK)

        if current_time - self.send_check_start_time > 1:
            self.sent_count = 0
            self.send_check_start_time = current_time

    def receive_serial_data(self):
        """シリアルデータを受信します。"""
        received = self.serial.get_received_queue()
        if received:
            self.app.queue.add_all("r", received)
            self.received_serial = received[-1]

    def update_fps(self, frame_start_time):
        """FPSを更新します。"""
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
        """接続状態を監視し、切断時の処理を行います。"""
        if not self.serial.ser_connected:
            self.stop = True
            self.print_stop()
            if self.serial.test_connect():
                asyncio.run(self.serial.connect_socket())
            if not self.disconnect:
                self.app.update_stop(connected=False)
                self.disconnect = True
                logger.warning("Disconnected from serial")
        else:
            if self.disconnect:
                self.app.update_stop(connected=True)
                logger.info("Reconnected to serial")
            self.disconnect = False

    def update_debug_info(self):
        """デバッグ情報を更新します。"""
        debug_text = (
            "{} {} x {} {} x: {} y: {} avr_fps: {} fps: {} target_position: {}\n"
            "target_x: {} motor_power_l: {} motor_power_r: {}\n"
            "connected: {} port: {} baud: {} data: {}\n"
            "STOP: {} serial_received: {}\n"
            "obstacle_detected: {} safe_x: {} no_detection: {}".format(
                self.seg,
                math.floor(self.frame_width),
                math.floor(self.frame_height),
                "O" if self.target_position_str != "X" else "X",
                self.d_x,
                self.d_y,
                math.floor(self.avr_fps),
                math.floor(self.fps),
                self.target_position_str,
                self.target_x,
                self.motor_power_l,
                self.motor_power_r,
                self.serial.ser_connected,
                SERIAL_PORT,
                SERIAL_BAUD,
                self.serial.serial_sent,
                self.stop,
                self.received_serial,
                not self.no_obs,
                self.safe_x,
                round(time.time() - self.target_last_seen_time, 2),
            )
        )
        self.print_d(debug_text)
        self.print_key_binds()
        if self.stop:
            self.print_stop()

    def update_gui(self):
        """GUIを更新します。"""
        self.app.update_frame(self.frame)

    def close(self):
        logger.info("Closing application")
        self.stop_motor()
        self.serial.close()

        if DEBUG_USE_WEBCAM:
            self.video_capture.release()

        cv2.destroyAllWindows()

    async def init(self):
        logger.info("Starting up...")
        time_startup = time.time()

        casc_path = cv2.data.haarcascades + Cascades.FACE
        self.cascade = cv2.CascadeClassifier(casc_path)
        if self.cascade.empty():
            logger.error("Failed to load cascade classifier")

        if DEBUG_USE_WEBCAM:
            self.video_capture = cv2.VideoCapture(CAM_NO)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            self.frame_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.frame_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(
                f"Webcam initialized: Width={self.frame_width}, Height={self.frame_height}"
            )
        else:
            self.frame_width = FRAME_WIDTH
            self.frame_height = FRAME_HEIGHT
            logger.info(
                f"Using predefined frame size: Width={self.frame_width}, Height={self.frame_height}"
            )

        self.frame = None

        logger.info("Initializing RFID Reader")
        self.rfid_reader = RFIDReader(port="COM4", baudrate=38400)
        logger.info(
            f"RFID Antenna Connected: {self.rfid_reader.check_antenna_status()}"
        )
        self.rfid_reader.start_reading()

        self.d_x, self.d_y = 0, 0
        self.face_x, self.face_center_x = 0, 0

        self.max_motor_power_threshold = self.frame_width / 4
        self.motor_power_l, self.motor_power_r = 0, 0

        self.interval_serial_send_start_time = 0
        self.send_check_start_time = 0

        self.seg = 0
        self.sent_count = 0

        logger.info(
            "Startup completed. Elapsed time: {}".format(self.elapsed_str(time_startup))
        )

        self.root = tk.Tk()
        self.app = gui.App(self, self.root)

        tracker_thread = threading.Thread(target=self.run_loop_serial)
        tracker_thread.daemon = True
        tracker_thread.start()

    def run_loop_serial(self):
        asyncio.run(self.serial.loop_serial())

    def start(self):
        tracker_thread = threading.Thread(target=self.main_loop)
        tracker_thread.daemon = True
        tracker_thread.start()

        self.root.mainloop()
        self.close()


async def main():
    tracker = Tracker()
    await tracker.init()
    tracker.start()


if __name__ == "__main__":
    asyncio.run(main())
