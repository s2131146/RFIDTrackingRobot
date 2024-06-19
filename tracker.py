import os

# カメラ起動高速化
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import time
import math
import serial
import gui
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

# 起動時間計測
print("[System] Starting up...")
time_startup = time.time()

# シリアルポートの設定
SERIAL_PORT = "COM3"
SERIAL_BAUD = 19200
SERIAL_SEND_INTERVAL = 0.3

# キャプチャウィンドウサイズ
FRAME_WIDTH = 512
FRAME_HEIGHT = 512

com: serial.Serial = None
com_connected = False

# 送信したコマンド
serial_sent = ""


def connect_socket():
    """シリアル通信を開始

    Returns:
        bool: 接続結果
    """
    global com_connected, com
    print("[Serial] Connecting to port")
    com = serial.Serial()
    com.port = SERIAL_PORT
    com.baudrate = SERIAL_BAUD
    com.timeout = None
    com.dtr = False

    try:
        com.open()
        com_connected = True
        print("[Serial] Serial port connected")
    except serial.SerialException:
        print("[Serial] Serial port is not open")
    finally:
        return com_connected


def elapsed_str(start_time):
    """経過時間の文字列を取得

    Args:
        start_time (float): 計測開始時刻

    Returns:
        str: 経過時間の文字列
    """
    return "{}ms".format(math.floor((time.time() - start_time) * 1000))


def print_d(frame, text):
    """デバッグ用画面出力

    Args:
        frame (MatLike): フレーム
        text (str): 出力文字列
    """
    text_lines = text.split("\n")
    for index, line in enumerate(text_lines, start=1):
        cv2.putText(
            frame, line, (5, 15 * index), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0)
        )


def print_key_binds(frame):
    """キーバインドを出力

    Args:
        frame (MatLike): フレーム
    """
    text = "PRESS X TO STOP.\nPRESS Q TO EXIT.\n{}".format(seg)
    text_lines = text.split("\n")
    text_lines.reverse()
    for index, line in enumerate(text_lines, start=1):
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = math.floor(frame_width - text_size[0] - 15)
        text_y = math.floor(frame_height - (15 * index)) + 5
        cv2.putText(
            frame, line, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0)
        )


def print_serial_stat(frame):
    """シリアル通信状況の出力

    Args:
        frame (MatLike): フレーム
    """
    text = "SERIAL SENT."
    text_y = math.floor(frame_height - 10)
    cv2.putText(frame, text, (5, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))


def send_serial(data, frame=None):
    """シリアル通信でコマンドを送信

    Args:
        data (str|tuple): 送信データ (コマンド|(コマンド, 値))
        frame (MatLike, optional): フレーム. Defaults to None.
    """
    global com_connected, serial_sent
    if not com_connected:
        if not connect_socket():
            return

    if isinstance(data, tuple):
        data = "{}:{}".format(data[0], data[1])
    elif not isinstance(data, str):
        print("[Serial] data is unsupported type")
        return

    serial_sent = data
    data += "\n"

    try:
        com.write(data.encode())
        if frame is not None:
            print_serial_stat(frame)
    except serial.SerialException:
        com_connected = False
        print("[Serial] Serial port is closed")


def print_serial(received):
    """Arduinoからのシリアル通信の内容を出力"""
    try:
        if com.in_waiting > 0:
            res = com.readline().decode("utf-8", "ignore").strip()
            print("[Serial] Received: {} [{}]".format(res, seg))
            return res
        else:
            return received
    except serial.SerialException:
        print("[Serial] Serial port is closed")


def calculate_fps(fps_count, total_fps, avr_fps):
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


def detect_target(frame):
    """画像から対象を検出

    Args:
        frame (MatLike): フレーム

    Returns:
        Sequence[Rect]: 座標情報
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    target = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return target


def calculate_motor_power(target_x):
    """モーター出力を計算

    Args:
        target_x (int): 対象座標X
    """
    global motor_power_l, motor_power_r
    if target_x < 0:
        motor_power_l = math.floor(math.sqrt(-target_x))
        motor_power_r = math.floor(math.sqrt(-(target_x * 4)))
        motor_power_l -= motor_power_r
    else:
        motor_power_l = math.floor(math.sqrt(target_x * 4))
        motor_power_r = math.floor(math.sqrt(target_x))
        motor_power_r -= motor_power_l

    motor_power_l += 70
    motor_power_r += 70

    if motor_power_l > 100:
        motor_power_l = 100
    if motor_power_r > 100:
        motor_power_r = 100


def get_target_pos_str(target_center_x):
    if target_center_x < frame_width // 3:
        target_position_str = "L"
    elif target_center_x < 2 * frame_width // 3:
        target_position_str = "C"
    else:
        target_position_str = "R"

    return target_position_str


def process_target(frame, target):
    """画像中の対象を囲む

    Args:
        frame (MatLike): フレーム
        target (Sequence[Rect]): 座標情報

    Returns:
        target_center_x, target_x: 顔座標原点X, 顔座標X
    """
    global d_x, d_y, target_detected
    for x, y, w, h in target:
        d_x = x
        d_y = y
        target_detected = True

        target_center_x = x + w // 2
        target_x = math.floor(target_center_x - frame_width / 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return target_center_x, target_x


def main_loop():
    """メインループ関数"""
    global seg, target_detected, interval_serial_send_start_time, d_x, d_y
    fps, total_fps, avr_fps, fps_count = 0, 0, 0, 0
    received_serial = None
    stop = False

    while True:
        seg += 1
        frame_start_time = time.time()
        target_detected = False
        target_position_str = "X"
        target_x = 0

        # 画面をキャプチャ
        _, frame = video_capture.read()
        frame = cv2.flip(frame, 1)

        # 対象が画面内にいれば処理
        targets = detect_target(frame)
        if len(targets) > 0:
            target_center_x, target_x = process_target(frame, targets)
            target_position_str = get_target_pos_str(target_center_x)
            calculate_motor_power(target_x)

        if cv2.waitKey(1) & 0xFF == ord("x"):
            stop = not stop

        if stop:
            send_serial("STOP", frame)
        else:
            # 一定間隔でシリアル通信を行う
            if time.time() - interval_serial_send_start_time > SERIAL_SEND_INTERVAL:
                interval_serial_send_start_time = time.time()
                if cv2.waitKey(1) & 0xFF == ord("a"):
                    send_serial(("L", 100), frame)
                    send_serial(("R", 60), frame)
                if cv2.waitKey(1) & 0xFF == ord("d"):
                    send_serial(("L", 60), frame)
                    send_serial(("R", 100), frame)
                send_serial(("L", motor_power_l), frame)
                send_serial(("R", motor_power_r), frame)

        received_serial = print_serial(received_serial)

        # 処理にかかった時間
        fps_end_time = time.time()
        time_taken = fps_end_time - frame_start_time
        if time_taken > 0:
            fps = 1 / time_taken

        # FPS計算
        fps_count += 1
        total_fps += fps
        fps_count, total_fps, avr_fps = calculate_fps(fps_count, total_fps, avr_fps)

        if seg > 9:
            seg = 0

        # Debug
        debug_text = (
            "{} {} x {} {} x: {} y: {} avr_fps: {} fps: {} target_position: {}\n"
            "target_x: {} motor_power_l: {} motor_power_r: {}\n"
            "socket_connected: {} port: {} baud: {} data: {}\n"
            "STOP: {} serial_received: {}".format(
                seg,
                math.floor(frame_width),
                math.floor(frame_height),
                "O" if target_position_str != "X" else "X",
                d_x,
                d_y,
                math.floor(avr_fps),
                math.floor(fps),
                target_position_str,
                target_x,
                motor_power_l,
                motor_power_r,
                com_connected,
                SERIAL_PORT,
                SERIAL_BAUD,
                serial_sent,
                stop,
                received_serial,
            )
        )
        print_d(frame, debug_text)
        print_key_binds(frame)

        cv2.imshow("Metoki - Human Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# OpenCV Haar分類器の指定
cascPath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascPath)

# フレームの設定
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

frame_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

# 顔座標
d_x, d_y = 0, 0
face_x, face_center_x = 0, 0

# モーター出力
max_motor_power_threshold = frame_width / 4
motor_power_l, motor_power_r = 0, 0

# シリアル通信送信時の間隔測定用
interval_serial_send_start_time = 0

seg = 0

connect_socket()

print("[System] Startup completed. Elapsed time:", elapsed_str(time_startup))

root = tk.Tk()
app = gui.App(root)
main_loop()

send_serial("STOP")
com.close()
video_capture.release()
cv2.destroyAllWindows()
