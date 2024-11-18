import socket
import sys
import traceback
import serial
import string
import pyrealsense2 as rs
import numpy as np
import math
import time
import cv2
import os
import logging

from constants import Commands

LOG_FILENAME = "robot.log"

if os.path.exists(LOG_FILENAME):
    os.remove(LOG_FILENAME)

logging.basicConfig(
    level=logging.INFO,
    format="[Robot   %(asctime)s.%(msecs)03d] INFO: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode="w"),
        logging.StreamHandler(),
    ],
)


def pr(txt, ow=False):
    txt = txt.rstrip("\n").replace("\n", "")
    if not ow:
        logging.info(txt)


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
        cmd_id = parts[1] if len(parts) > 1 else 0
        remaining_parts = parts[2:] if len(parts) > 2 else ""

        if len(remaining_parts) == 0:
            return

        if "\n" in remaining_parts[-1]:
            last_part = remaining_parts[-1].strip()
            remaining_parts[-1] = last_part

        if len(remaining_parts) > 1:
            data = tuple(remaining_parts)
        else:
            data = remaining_parts[0]

        if data == Commands.DETACH_MOTOR:
            self.com.dtr = False
            time.sleep(0.1)
            self.com.dtr = True

        if cmd == "connect":
            self.connect_socket()
        elif cmd == "close":
            self.close()
        elif cmd == "send":
            ret = self.send_serial(data)
            self.send_to_host(
                ret,
                cmd_id,
                "send",
                True if data == Commands.CHECK or data == Commands.STOP_TEMP else False,
            )
        elif cmd == "print":
            res, ret = self.print_serial()
            response = ret if res else "NULL"
            self.send_to_host(response, cmd_id, "print")

    def send_to_host(self, data, id=-1, cmd=None, wo=False):
        send_str = "{}:{}:{}".format(cmd if cmd is not None else "force", id, data)
        if data is not True:
            pr("Send to client: {}".format(send_str), wo)
        try:
            send_data(send_str.encode())
        except socket.timeout:
            self.send_to_host(data, id, cmd)
            pr("Send timed out")

    def connect_socket(self):
        """Start serial communication.

        Returns:
            bool: Connection result
        """
        self.com = serial.Serial()
        self.com.port = self.ser_port
        self.com.baudrate = self.baud
        self.com.timeout = 0.5
        self.com.dtr = False

        try:
            self.com.open()
            self.com_connected = True
        except serial.SerialException:
            pass
        return self.com_connected

    def get_command(self, data) -> string:
        if isinstance(data, tuple):
            data = data[0].split(":")[0].upper()
        else:
            data = data.split(":")[0].upper()

        return data

    def stop(self):
        self.send_serial("stop")

    last_send_check = time.time()

    def send_serial(self, data) -> bool:
        """Send a command via serial communication.

        Args:
            data (str|tuple): Data to send (command|(command, value))

        Return:
            bool: Send result
        """
        if not self.com_connected and not self.connect_socket():
            return False

        if isinstance(data, tuple):
            data = "{}:{}".format(data[0], data[1])
        else:
            data = data.upper()

        self.command_sent = self.get_command(data).upper()
        self.serial_sent = data
        if data == Commands.CHECK:
            if time.time() - self.last_send_check < 0.5:
                return True
            else:
                self.last_send_check = time.time()
        data += "\n"

        try:
            self.com.write(data.encode())
            return True
        except serial.SerialException:
            self.com_connected = False
            return False

    def print_serial(self):
        """Output the content received from Arduino via serial communication."""
        try:
            if isinstance(self.com, serial.Serial) and self.com.in_waiting > 0:
                res = self.com.readline().decode("utf-8", "ignore").strip()
                return True, res
            else:
                return False, ""
        except serial.SerialException:
            return False, ""

    def close(self):
        if client_socket:
            client_socket.close()
        if self.com:
            self.com.close()


client_socket = None
server_socket = None
tracker: TrackerSocketRobot
pipeline = None
cam_connected = False


def get_ipv4_address():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


def init():
    global client_socket, server_socket
    server_ip = "0.0.0.0"
    server_port = 8001
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(1)
    server_socket.settimeout(0.1)

    pr(f"Listening on IP: {get_ipv4_address()} PORT: {server_port}")
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
    except Exception as e:
        pr(f"Camera streaming error: {e}")
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
        chunk = client_socket.recv(32)
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
    global tracker, cam_connected, reconnect
    try:
        while True:
            if not cam_connected:
                cam_connected = init_cam()
            # stream_cam()
            data = receive_data()
            if not data:
                continue
            if data.startswith(Commands.DISCONNECT):
                reconnect = True
                tracker.send_to_host("Waiting for reconnect...")
                return
            if data.startswith("setup:"):
                data = data[len("setup:") :].split(":")
                tracker = TrackerSocketRobot(*data)
            else:
                if tracker:
                    tracker.on_receive(data)
            if tracker:
                res, ret = tracker.print_serial()
                if res:
                    pr(f"Arduino: {ret}")
                    response = ret if res else "NULL"
                    tracker.send_to_host(response, -1, "print", True)
    except ConnectionResetError:
        if tracker:
            tracker.stop()
        pr("Client disconnected")
    except Exception as e:
        pr("An error occurred: {}\n{}".format(e, traceback.format_exc()))
    finally:
        pr("Closing sockets...")
        if client_socket:
            client_socket.close()
        if server_socket:
            server_socket.close()


def main():
    try:
        init()
        pr("Loop starting")
        loop()
        if tracker:
            tracker.close()
    except KeyboardInterrupt:
        if tracker:
            tracker.stop()
        pr("Interrupted by user")
    except Exception as e:
        if tracker:
            tracker.stop()
        pr("An error occurred in main loop: {}\n{}".format(e, traceback.format_exc()))


def init_cam():
    return True
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

    except RuntimeError as e:
        pr(f"RealSense initialization failed: {e}")
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
        pr("No RealSense devices found.")
        return False


reconnect = False

if __name__ == "__main__":
    pr("RFID Human Tracker [Robot] 2024 Metoki.")
    pr("Starting...")
    s = time.time()
    cam_connected = init_cam()
    if not cam_connected:
        pr("RealSense is not connected.")
    pr("Startup completed. Elapsed: {}ms".format(math.floor((time.time() - s) * 1000)))
    while True:
        main()
        if not (len(sys.argv) > 1 and sys.argv[1] == "--no_abort") and not reconnect:
            break
        reconnect = False
