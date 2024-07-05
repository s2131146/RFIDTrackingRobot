import os
import threading

# カメラ起動高速化
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import time
import math
import tkinter as tk
import pyrealsense2 as rs
import numpy as np
import asyncio

from constants import Commands
from constants import Cascades
import tracker_socket as ts
import gui

# シリアルポートの設定
SERIAL_PORT = "COM3"
SERIAL_BAUD = 19200
SERIAL_SEND_INTERVAL = 0.1

TCP_PORT = 8001

# キャプチャウィンドウサイズ
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CAM_NO = 0

DEBUG_SERIAL = True
DEBUG_USE_WEBCAM = True


class Tracker:
    root = app = cascade = video_capture = None
    frame_width = frame_height = d_x = d_y = face_x = face_center_x = None
    max_motor_power_threshold = motor_power_l = motor_power_r = None
    interval_serial_send_start_time = seg = serial_sent = None

    serial = ts.TrackerSocket(
        SERIAL_PORT, SERIAL_BAUD, SERIAL_SEND_INTERVAL, DEBUG_SERIAL, TCP_PORT
    )

    stop = False

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
        text = "SERIAL SENT."
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

    def calculate_fps(self, fps_count, total_fps, avr_fps):
        """FPSを計算

        Args:
            fps_count (int): 平均FPSを算出するための呼び出し回数カウント
            total_fps (int): カウント中の合計FPS
            avr_fps (int): 平均FPS

        Returns:
            fps_count, total_fps, avr_fps
        """
        if fps_count >= 10:
            fps_count = 0
            avr_fps = total_fps / 10
            total_fps = 0
        return fps_count, total_fps, avr_fps

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
            return False

        ret = self.serial.send_serial(data)
        if ret:
            self.print_serial_stat()
        self.exec_sent_command()

        if data != Commands.CHECK:
            self.app.queue.add("s", self.serial.serial_sent)

        return True

    def exec_sent_command(self):
        c = self.serial.command_sent.upper()
        if c == Commands.STOP:
            self.stop = True
            self.app.update_stop()
        if c == Commands.START:
            self.start_motor()

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

    def process_target(self, target):
        """画像中の対象を囲む

        Args:
            frame (MatLike): フレーム
            target (Sequence[Rect]): 座標情報

        Returns:
            target_center_x, target_x: 顔座標原点X, 顔座標X
        """
        for x, y, w, h in target:
            self.d_x = x
            self.d_y = y
            self.target_detected = True

            target_center_x = x + w // 2
            target_x = math.floor(target_center_x - self.frame_width / 2)

            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text_org = (x, y - 10)
            cv2.putText(
                self.frame,
                "TARGET",
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            return target_center_x, target_x

    def stop_motor(self):
        self.send(Commands.STOP)

    def start_motor(self):
        if self.serial.ser_connected:
            self.stop = False
            self.app.update_stop()

    detected_obstacles = []

    def process_obstacles(self, frame, depth_frame):
        """障害物を検知し、描画

        Args:
            frame (MatLike): フレーム
            depth_frame: 深度フレーム

        Returns:
            frame, no_obs, x: 描画後フレーム, 障害物がないか, 安全なX座標
        """
        depth_image = np.asanyarray(depth_frame.get_data())
        roi_depth = depth_image[: int(depth_image.shape[0] * 4 / 5), :]
        roi_color = frame[: int(frame.shape[0] * 4 / 5), :]

        roi_depth = cv2.GaussianBlur(roi_depth, (5, 5), 0)

        max_distance = 300  # 300mm
        min_distance = 100  # 100mm

        mask = np.logical_and(roi_depth > min_distance, roi_depth < max_distance)
        obstacle_image = np.zeros_like(roi_depth)
        obstacle_image[mask] = 255

        kernel = np.ones((5, 5), np.uint8)
        obstacle_image = cv2.morphologyEx(obstacle_image, cv2.MORPH_CLOSE, kernel)
        obstacle_image = cv2.morphologyEx(obstacle_image, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            obstacle_image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        center_x = roi_color.shape[1] // 2

        max_distance_to_center = 0
        obstacle_free_center_x = center_x
        no_obstacle_detected = True
        detected_obstacles_info = []

        for contour in contours:
            if cv2.contourArea(contour) > 1500:
                no_obstacle_detected = False

                x, y, w, h = cv2.boundingRect(contour)
                obstacle_center_x = x + w // 2

                # 画像の中央から障害物までの距離を計算
                distance_to_center = obstacle_center_x - center_x

                # 最も遠い障害物までの距離を更新
                if abs(distance_to_center) > abs(max_distance_to_center):
                    max_distance_to_center = distance_to_center

                detected_obstacles_info.append((x, y, w, h, cv2.contourArea(contour)))
                all_obstacle_x_positions = [
                    info[0] + info[2] // 2 for info in detected_obstacles_info
                ]

                # 安全なX座標を計算
                if detected_obstacles_info:
                    min_obstacle_x = min(all_obstacle_x_positions)
                    max_obstacle_x = max(all_obstacle_x_positions)
                    obstacle_free_center_x = (min_obstacle_x + max_obstacle_x) // 2
                else:
                    obstacle_free_center_x = center_x

                obstacle_free_center_x = math.floor(
                    obstacle_center_x - self.frame_width / 2
                )

                cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
                text_org = (x, y - 10)
                cv2.putText(
                    frame,
                    "OBSTACLE {}mm".format(distance_to_center),
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        return frame, no_obstacle_detected, obstacle_free_center_x

    def main_loop(self):
        fps, total_fps, avr_fps, fps_count, bkl, bkr = 0, 0, 0, 0, 0, 0
        received_serial = None
        disconnect = False

        while True:
            self.seg += 1
            frame_start_time = time.time()
            self.target_detected = False
            target_position_str = "X"
            target_x = 0
            no_obs, safe_x = True, 0

            # 画面をキャプチャ
            if DEBUG_USE_WEBCAM:
                _, frame = self.video_capture.read()
                self.frame = cv2.flip(frame, 1)
            else:
                frames = self.rs_pipeline.wait_for_frames()
                frame = frames.get_color_frame()
                frame = np.asanyarray(frame.get_data())
                depth_frame = frames.get_depth_frame()
                self.frame, no_obs, safe_x = self.process_obstacles(frame, depth_frame)

            # 対象が画面内にいれば処理
            targets = self.detect_target()
            bkl = self.motor_power_l
            bkr = self.motor_power_r

            if len(targets) > 0:
                target_center_x, target_x = self.process_target(targets)
                target_position_str = self.get_target_pos_str(target_center_x)
                self.calculate_motor_power(target_x)

            # 障害物がある場合、避ける
            if not no_obs:
                self.calculate_motor_power(-safe_x)

            # 一定間隔でシリアル通信を行う
            if (
                time.time() - self.interval_serial_send_start_time
                > SERIAL_SEND_INTERVAL
            ):
                self.interval_serial_send_start_time = time.time()

                if self.app.var_enable_tracking.get():
                    if not self.stop:
                        if bkl != self.motor_power_l:
                            self.send((Commands.SET_SPD_LEFT, self.motor_power_l))
                        if bkr != self.motor_power_r:
                            self.send((Commands.SET_SPD_RIGHT, self.motor_power_r))

                        if target_position_str == "X":
                            self.app.queue.add("g", "Target not found")
                        else:
                            self.app.queue.add(
                                "g",
                                "GO:{} | L:{}% R:{}%".format(
                                    target_position_str,
                                    self.motor_power_l,
                                    self.motor_power_r,
                                ),
                            )

                self.send(Commands.CHECK)

            received = self.serial.get_received_queue()
            if received:
                self.app.queue.add_all("r", received)
                received_serial = received[-1]

            # 処理にかかった時間
            fps_end_time = time.time()
            time_taken = fps_end_time - frame_start_time
            if time_taken > 0:
                fps = 1 / time_taken

            # FPS計算
            fps_count += 1
            total_fps += fps
            fps_count, total_fps, avr_fps = self.calculate_fps(
                fps_count, total_fps, avr_fps
            )

            if self.seg > 9:
                self.seg = 0

            if not self.serial.ser_connected:
                self.stop = True
                self.print_stop()
                if self.serial.test_connect():
                    asyncio.run(self.serial.connect_socket())
                if not disconnect:
                    self.app.update_stop(connected=False)
                    disconnect = True
            else:
                if disconnect:
                    self.app.update_stop(connected=True)
                disconnect = False

            # Debug
            debug_text = (
                "{} {} x {} {} x: {} y: {} avr_fps: {} fps: {} target_position: {}\n"
                "target_x: {} self.motor_power_l: {} self.motor_power_r: {}\n"
                "socket_connected: {} port: {} baud: {} data: {}\n"
                "STOP: {} serial_received: {}\n"
                "obstacle_detected: {} safe_x: {}".format(
                    self.seg,
                    math.floor(self.frame_width),
                    math.floor(self.frame_height),
                    "O" if target_position_str != "X" else "X",
                    self.d_x,
                    self.d_y,
                    math.floor(avr_fps),
                    math.floor(fps),
                    target_position_str,
                    target_x,
                    self.motor_power_l,
                    self.motor_power_r,
                    self.serial.ser_connected,
                    SERIAL_PORT,
                    SERIAL_BAUD,
                    self.serial.serial_sent,
                    self.stop,
                    received_serial,
                    not no_obs,
                    safe_x,
                )
            )
            self.app.update_seg(self.seg)
            self.print_d(debug_text)
            self.print_key_binds()

            if self.stop:
                self.print_stop()

            self.app.update_frame(self.frame)

    def close(self):
        self.stop_motor()
        self.serial.close()

        if DEBUG_USE_WEBCAM:
            self.video_capture.release()
        else:
            self.rs_pipeline.stop()

        cv2.destroyAllWindows()

    async def init(self):
        # 起動時間計測
        print("[System] Starting up...")
        time_startup = time.time()

        # OpenCV Haar分類器の指定
        cascPath = cv2.data.haarcascades + Cascades.FACE
        self.cascade = cv2.CascadeClassifier(cascPath)

        if DEBUG_USE_WEBCAM:
            self.video_capture = cv2.VideoCapture(CAM_NO)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            self.frame_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.frame_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.rs_config.enable_stream(
                rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 60
            )
            self.rs_config.enable_stream(
                rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 60
            )
            self.rs_pipeline.start(self.rs_config)

            profile = self.rs_pipeline.get_active_profile()
            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            color_intrinsics = color_profile.get_intrinsics()
            self.frame_width = color_intrinsics.width
            self.frame_height = color_intrinsics.height

        self.frame = None

        # 顔座標
        self.d_x, self.d_y = 0, 0
        self.face_x, self.face_center_x = 0, 0

        # モーター出力
        self.max_motor_power_threshold = self.frame_width / 4
        self.motor_power_l, self.motor_power_r = 0, 0

        # シリアル通信送信時の間隔測定用
        self.interval_serial_send_start_time = 0

        self.seg = 0

        await self.serial.connect_socket()
        asyncio.create_task(self.serial.loop_serial())

        print(
            "[System] Startup completed. Elapsed time:", self.elapsed_str(time_startup)
        )

        self.root = tk.Tk()
        self.app = gui.App(self, self.root)

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
