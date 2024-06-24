#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import time
import math
import sys
import io
import numpy as np
import serial
# 標準出力のエンコーディングをUTF-8に設定
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
SERIAL_PORT = "COM3"
SERIAL_BAUD = 19200
SERIAL_SEND_INTERVAL = 0.3
def printD(frame, text):
    cv2.putText(frame, text, (5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

def printKeybinds(frame, seg):
    text = "PRESS Q TO EXIT. {}".format(seg)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = math.floor(frame_width - text_size[0] - 15)
    text_y = math.floor(frame_height - 10)
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

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
        print("[Serial] Unsupported data type for serial communication.")
        return

    serial_sent = data
    data += "\n"

    try:
        com.write(data.encode())
        if frame is not None:
            print_serial_stat(frame)
    except serial.SerialException:
        com_connected = False
        print("[Serial] Serial port is closed.")


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
        print("[Serial] Serial port is closed.")

# Webカメラのキャプチャ
video_capture = cv2.VideoCapture(1)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

frame_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

# 物体座標
dX, dY = 0, 0

# FPS計算
fps_start_time = 0
fps, total_fps, avr_fps, fps_count = 0, 0, 0, 0

seg = 0

miss_count = 0

# #FF734BのHSV範囲
lower_orange = np.array([5, 150, 200])
upper_orange = np.array([15, 255, 255])

connect_socket()

start = None

while True:
    seg += 1
    detected = False
    frame_start_time = time.time()

    # フレームのキャプチャ
    ret, frame = video_capture.read()

    # オレンジ色（#FF734B）の検出
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        detected = True
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        dX, dY = x + w // 2, y + h // 2  # オブジェクトの中心座標

        # オレンジ色の矩形を描画
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # モーターを動かす
        if not (w * h > frame_width * frame_height / 2):
            send_serial("L_DIRECTION", 0)
            send_serial("R_DIRECTION", 0)
            send_serial(("L", 100))
            send_serial(("R", 100))
            miss_count = 0

            # オブジェクトの中心が画面の中央から左右にずれているか判定
            frame_center_x = frame_width // 2
            if dX < frame_center_x - 10:  # 10ピクセル以上左にずれている場合
                direction = "Left"
                send_serial(("L", 0))
                send_serial(("R", 100))
            elif dX > frame_center_x + 10:  # 10ピクセル以上右にずれている場合
                direction = "Right"
                send_serial(("L", 100))
                send_serial(("R", 0))
            else:
                direction = "Center"

        # オブジェクトの面積が画角の2分の1以上を占めたら「STOP」テキストを表示
        if w * h > frame_width * frame_height / 2:
            if start is None:
                start = time.time()
            else:
                if time.time() - start > 3:
                    send_serial("L_DIRECTION", 1)
                    send_serial("R_DIRECTION", 1)
                    start = None
            send_serial(("STOP"))
            cv2.putText(frame, "STOP", (int(frame_width/2) - 50, int(frame_height/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    else:
        miss_count += 1
        direction = ""

    # 方向を画面に表示
    cv2.putText(frame, direction, (5, 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    # FPS計算
    frame_end_time = time.time()
    time_taken = frame_end_time - frame_start_time

    if time_taken > 0:
        fps = 1 / time_taken

    fps_count += 1
    total_fps += fps

    if fps_count >= 10:
        fps_count = 0
        avr_fps = total_fps / 10
        total_fps = 0

    if seg > 9:
        seg = 0

    # Debug
    printD(frame, "{} {} x {} {} x: {} y: {} ave_fps: {} fps: {}".format(
               seg,
               math.floor(frame_width), math.floor(frame_height),
               "O" if detected else "X",
               dX, dY,
               math.floor(avr_fps),
               math.floor(fps)))

    printKeybinds(frame, seg)

    # 結果を表示
    cv2.imshow('Metoki - Orange Color Tracker', frame)

    # 'q'キーを押すとループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャを解放し、ウィンドウを閉じる
video_capture.release()
cv2.destroyAllWindows()
