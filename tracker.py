import os

# カメラ起動高速化
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import time
import math
import serial

# 起動時間計測
print("[System] Starting up...")
time_startup = time.time()

# シリアルポートの設定
SERIAL_PORT = "COM3"
SERIAL_BAUD = 9600
SERIAL_SEND_INTERVAL = 0.1

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

def print_d(frame, text, line):
    """デバッグ用画面出力

    Args:
        frame (MatLike): フレーム
        text (str): 出力文字列
        line (int): 出力行
    """
    cv2.putText(frame, text, (5, 15 * line), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

def print_key_binds(frame):
    """キーバインドを出力

    Args:
        frame (MatLike): フレーム
    """
    text = "PRESS Q TO EXIT. {}".format(seg)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = math.floor(frame_width - text_size[0] - 15)
    text_y = math.floor(frame_height - 10)
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

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

def print_serial():
    """Arduinoからのシリアル通信の内容を出力
    """
    if com.in_waiting > 0:
        res = com.readline().decode('utf-8', "ignore").strip()
        print(res)

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
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return target

def process_target(frame, target):
    """画像中の対象を囲む

    Args:
        frame (MatLike): フレーム
        target (Sequence[Rect]): 座標情報

    Returns:
        target_position, target_x: 顔の位置情報(左中右), 顔座標X
    """
    global motor_power_l, motor_power_r, d_x, d_y, target_detected
    for (x, y, w, h) in target:
        d_x = x
        d_y = y
        target_detected = True

        target_center_x = x + w // 2
        target_x = math.floor(target_center_x - frame_width / 2)

        if target_x < 0:
            motor_power_l = math.floor(math.sqrt(-target_x))
            motor_power_r = math.floor(math.sqrt(-(target_x * 2)))
        else:
            motor_power_l = math.floor(math.sqrt(target_x * 2))
            motor_power_r = math.floor(math.sqrt(target_x))

        # 左右の判断
        if target_center_x < frame_width // 3:
            target_position_str = "L"
        elif target_center_x < 2 * frame_width // 3:
            target_position_str = "C"
        else:
            target_position_str = "R"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return target_position_str, target_x

def main_loop():
    """メインループ関数"""
    global seg, target_detected, interval_serial_send_start_time, d_x, d_y
    fps, total_fps, avr_fps, fps_count = 0, 0, 0, 0
    
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
            target_position_str, target_x = process_target(frame, targets)

        # 一定間隔でシリアル通信を行う
        if time.time() - interval_serial_send_start_time > SERIAL_SEND_INTERVAL:
            interval_serial_send_start_time = time.time()
            send_serial((target_position_str, 98.233), frame)

        print_serial()

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
        print_d(frame, "{} {} x {} {} x: {} y: {} ave_fps: {} fps: {} face_position: {}".format(
                seg,
                math.floor(frame_width), math.floor(frame_height),
                "O" if target_position_str != "X" else "X",
                d_x, d_y,
                math.floor(avr_fps),
                math.floor(fps),
                target_position_str), 1)
        print_d(frame, "face_x: {} motor_power_l: {} motor_power_r: {}".format(
                target_x, motor_power_l, motor_power_r), 2)
        print_d(frame, "socket_connected: {} port: {} baud: {} data: {}".format(
                com_connected, SERIAL_PORT, SERIAL_BAUD, serial_sent), 3)

        print_key_binds(frame)

        cv2.imshow('Metoki - Human Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# OpenCV Haar分類器の指定
cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
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

main_loop()

com.close()
video_capture.release()
cv2.destroyAllWindows()
