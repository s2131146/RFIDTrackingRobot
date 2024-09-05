import socket
import traceback
import serial
import string
import datetime
import pyrealsense2 as rs
import numpy as np
import math
import time
import cv2


def pr(txt, ow=False):
    txt = txt.rstrip("\n")
    txt = txt.replace("\n", "")
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, "JST")
    now = datetime.datetime.now(JST)
    if not ow:
        print("[Socket {}] {}".format(now.strftime("%H:%M:%S.%f")[:-3], txt))


class TrackerSocketRobot:
    com: serial.Serial = None
    com_connected = False

    def __init__(self, p, b, i, ip, port):
        self.ser_port = p
        self.baud = b
        self.interval = i
        self.tcp_ip = ip
        self.tcp_port = port

    def on_receive(self, received):
        parts = str(received).split(":")
        cmd = parts[0]
        cmd_id = parts[1]
        data = (parts[2] if len(parts) > 1 else "").rstrip("\n")

        if cmd == "connect":
            self.connect_socket()
        elif cmd == "close":
            self.close()
        elif cmd == "send":
            ret = self.send_serial(data)
            self.send_to_host(ret, cmd_id, "send", True if data == "YO" else False)
        elif cmd == "print":
            res, ret = self.print_serial()
            response = ret if res else "NULL"
            self.send_to_host(response, cmd_id, "print")

    def send_to_host(self, data, id=-1, cmd=None, wo=False):
        send_str = "{}:{}:{}".format(cmd if cmd is not None else "force", id, data)
        pr("Sending: {}".format(send_str), wo)
        try:
            send_data(send_str.encode())
        except socket.timeout:
            self.send_to_host(data, id, cmd)
            pr("Send timed out")

    def connect_socket(self):
        """シリアル通信を開始

        Returns:
            bool: 接続結果
        """
        self.send_to_host("Connecting to Arduino...")
        self.com = serial.Serial()
        self.com.port = self.ser_port
        self.com.baudrate = self.baud
        self.com.timeout = 0.5
        self.com.dtr = False

        try:
            self.com.open()
            self.com_connected = True
            self.send_to_host("Serial port connected")
        except serial.SerialException:
            self.send_to_host("Serial port is not open")
        return self.com_connected

    def get_command(self, data) -> string:
        if isinstance(data, tuple):
            data = data[0].split(":")[0].upper()
        else:
            data = data.split(":")[0].upper()

        return data

    def send_serial(self, data) -> bool:
        """シリアル通信でコマンドを送信

        Args:
            data (str|tuple): 送信データ (コマンド|(コマンド, 値))

        Return:
            bool: 送信結果
        """
        if not self.com_connected and not self.connect_socket():
            return False

        if isinstance(data, tuple):
            data = "{}:{}".format(data[0], data[1])
        else:
            data = data.upper()

        self.command_sent = self.get_command(data).upper()
        self.serial_sent = data
        data += "\n"

        try:
            self.com.write(data.encode())
            return True
        except serial.SerialException:
            self.com_connected = False
            self.send_to_host("Failed to connect to Arduino.")
            return False

    def print_serial(self):
        """Arduinoからのシリアル通信の内容を出力"""
        try:
            if self.com.in_waiting > 0:
                res = self.com.readline().decode("utf-8", "ignore").strip()
                self.send_to_host("Received: {}".format(res))
                return True, res
            else:
                return False, ""
        except serial.SerialException:
            self.send_to_host("Failed to connect to Arduino.")
            return False, ""

    def close(self):
        client_socket.close()
        self.com.close()


client_socket = None
server_socket = None
tracker = None
pipeline = None
cam_connected = False


def init():
    global client_socket, server_socket
    server_ip = "0.0.0.0"
    server_port = 8001
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(1)
    server_socket.settimeout(0.1)

    pr("Waiting for connection...")

    while True:
        try:
            client_socket, client_address = server_socket.accept()
            break
        except socket.timeout:
            continue
    pr(f"Device connected: {client_address}")


def stream_cam():
    global pipeline, cam_connected
    if not cam_connected:
        return

    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        _, color_encoded = cv2.imencode(".jpg", color_image, encode_param)

        color_data = color_encoded.tobytes()
        depth_data = depth_image.astype(np.uint16).tobytes()

        send_data(b"color" + color_data)
        send_data(b"depth" + depth_data)
    except Exception:
        pipeline = None
        cam_connected = False


def send_data(data):
    if isinstance(data, str):
        client_socket.sendall(b"START" + data.encode() + b"END")
    else:
        client_socket.sendall(b"START" + data + b"END")


def receive_data():
    data_buffer = b""
    while True:
        chunk = client_socket.recv(4096)
        if not chunk:
            break
        data_buffer += chunk
        start_idx = data_buffer.find(b"START")
        end_idx = data_buffer.find(b"END")
        if start_idx != -1 and end_idx != -1:
            data = data_buffer[start_idx + 5 : end_idx]
            data_buffer = data_buffer[end_idx + 3 :]
            return data.decode()
    return None


def loop():
    global tracker, cam_connected
    try:
        while True:
            if not cam_connected:
                cam_connected = init_cam()
            stream_cam()
            data = receive_data()
            if not data:
                continue
            if data.startswith("setup:"):
                data = data[len("setup:") :].split(":")
                tracker = TrackerSocketRobot(*data)
            else:
                if tracker is not None:
                    tracker.on_receive(data)
    except ConnectionResetError:
        pr("Client disconnected")
    except Exception as e:
        pr("An error occurred: {}\n{}".format(e, traceback.format_exc()))
    finally:
        pr("Closing sockets...")
        client_socket.close()
        server_socket.close()


def main():
    while True:
        try:
            init()
            pr("Loop started")
            loop()
        except KeyboardInterrupt:
            pr("Interrupted by user")
            break
        except Exception as e:
            pr(
                "An error occurred in main loop: {}\n{}".format(
                    e, traceback.format_exc()
                )
            )
            pr("Restarting...")


def init_cam():
    global pipeline
    try:
        pr("Initializing RealSense...")
        pipeline = rs.pipeline()
        pr(f"Created an Object: {pipeline}")
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        pr("Started streaming of RealSense.")

        profile = pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        frame_width = color_intrinsics.width
        frame_height = color_intrinsics.height
        pr(f"Frame WIDTH: {frame_width}, HEIGHT: {frame_height}")

    except RuntimeError:
        return False

    context = rs.context()
    devices = context.query_devices()

    pr("Finding Activated RealSenses...")
    if len(devices) > 0:
        pr(f"Active RealSenses ({len(devices)}):")
        for device in devices:
            pr(
                f"- {device.get_info(rs.camera_info.name)} [#{device.get_info(rs.camera_info.serial_number)}]"
            )
        return True
    else:
        return False


if __name__ == "__main__":
    pr("RFID Human Tracker [Server side] 2024 Metoki.")
    pr("Starting...")
    s = time.time()
    cam_connected = init_cam()
    if not cam_connected:
        pr("RealSense is not connected.")
    pr("Startup completed. Elapsed: {}ms".format(math.floor((time.time() - s) * 1000)))
    main()
