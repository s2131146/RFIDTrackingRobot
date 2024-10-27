import math
import time
import socket
import traceback
import tqueue
import asyncio
import numpy as np
import cv2
import logger as l
import pyrealsense2 as rs
from utils import Utils
from constants import Commands

logger = l.logger


class TrackerSocket:
    serial_sent = None
    command_sent = ""
    queue = tqueue.TQueue()
    data_id = 0
    pipeline = None

    def __init__(self, p, b, i, d, port, webcam):
        self.ser_port = p
        self.baud = b
        self.interval = i
        self.debug = d
        self.tcp_ip = self.get_local_ip()
        self.tcp_port = port

        if not webcam:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            try:
                self.pipeline.start(config)
            except RuntimeError:
                logger.info("No RealSense connected.")

    @classmethod
    def get_local_ip(cls):
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            return ip_address
        except socket.error as e:
            logger.error(
                "[Socket] An error occured during getting local IP: {}".format(e)
            )
            return None

    def test_connect(self):
        try:
            test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test.settimeout(0.5)
            test.connect((self.tcp_ip, self.tcp_port))
            test.close()
            return True
        except socket.error:
            return False

    ser_connected = False

    async def connect_socket(self):
        """シリアル通信を開始

        Returns:
            bool: 接続結果
        """
        if self.tcp_ip is None:
            logger.error("[Socket] IP Address is not defined.")
            self.ser_connected = False
            return
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(0.5)
            self.client_socket.setblocking(True)
            self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
            await asyncio.get_event_loop().sock_connect(
                self.client_socket, (self.tcp_ip, self.tcp_port)
            )
            self.ser_connected = True
            logger.info("[Socket] Socket connected.")
            self.send_startup()
        except socket.error as e:
            logger.error("[Socket] An error occured at connect_socket: {}".format(e))
            self.ser_connected = False
        return self.ser_connected

    async def receive_data(self):
        data_buffer = b""
        while True:
            chunk = await asyncio.get_event_loop().sock_recv(self.client_socket, 8192)
            if not chunk:
                break
            data_buffer += chunk
            start_idx = data_buffer.find(b"START")
            end_idx = data_buffer.find(b"END")
            if start_idx != -1 and end_idx != -1:
                data = data_buffer[start_idx + 5 : end_idx]
                data_buffer = data_buffer[end_idx + 3 :]
                return data
        return None

    def process_data(self, data):
        """header = data.decode("ISO-8859-1")[:5]
        if header == "color" or header == "depth":
            self.make_images(header, data[5:])
            return"""

        if self.pipeline is not None:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame and depth_frame:
                self.color_image = np.asanyarray(color_frame.get_data())
                self.depth_image = np.asanyarray(depth_frame.get_data())

        self.exec_data(data)

    def exec_data(self, data):
        data = data.decode()
        parts = data.split(":", 2)
        cmd = parts[0]
        if len(parts) > 1:
            cid = parts[1]
            value = parts[2]
            value_bool = Utils.try_to_bool(value)
            if not isinstance(value_bool, bool):
                self.queue.add("r", value)
            if cmd == "force":
                logger.info(value)
            if cmd == "logger.info":
                self.logger.info_serial(value)
            if cid != -1 and self.queue.get("w:{}".format(cid)) is not None:
                self.queue.add(cid, value_bool)

    async def loop_serial(self):
        while True:
            try:
                if not self.ser_connected and not await self.connect_socket():
                    continue

                data = await self.receive_data()
                if not data:
                    continue

                try:
                    self.process_data(data)
                except RuntimeError:
                    pass
            except Exception:
                logger.error(
                    "[Socket] An error occured at loop_serial: {}".format(
                        traceback.format_exc()
                    )
                )

    def send_startup(self):
        self.send_data(
            "setup:{}:{}:{}:{}:{}\n".format(
                self.ser_port, self.baud, self.interval, self.tcp_ip, self.tcp_port
            )
        )

    def wait_for_result(self, data, id):
        if not self.ser_connected:
            logger.error("[Socket] Not connected. Cannot start loop_serial.")
            return
        fid = "w:{}".format(id)
        self.queue.add(fid)
        self.send_data(data)
        s = time.time()
        while not self.queue.has(fid):
            if math.floor((time.time() - s) * 1000) > 5000:
                self.connect_socket()
                return None
        return self.queue.get(fid)

    def send_serial(self, data) -> bool:
        """シリアル通信でコマンドを送信

        Args:
            data (str|tuple): 送信データ (コマンド|(コマンド, 値))

        Return:
            bool: 送信結果
        """
        if not self.ser_connected:
            if not self.test_connect():
                return False
            else:
                asyncio.run(self.connect_socket())
                return False

        if self.data_id > 99999:
            self.data_id = 0

        self.data_id += 1
        skip_log = True if Commands.is_ignore(data) else False
        if isinstance(data, tuple):
            cmd = data[0].upper()
            data = "send:{}:{}:{}".format(self.data_id, data[0], data[1])
        else:
            cmd = data.upper()
            data = "send:{}:{}".format(self.data_id, data.upper())

        self.command_sent = self.get_command(cmd).upper()
        if not skip_log:
            self.serial_sent = data
        logger.debug(f"Send: {data}")
        data += "\n"

        try:
            if skip_log:
                self.send_data(data)
                return True
            else:
                res = self.wait_for_result(data, self.data_id)
                if data == "STOP":
                    logger.info(res)
            if res is None:
                logger.error("[Socket] No connection now")
                self.ser_connected = False
            return True
        except socket.error as e:
            self.ser_connected = False
            if self.debug:
                logger.error("[Socket] Failed to send to client: {}".format(e))
            return False

    def get_command(self, data):
        if isinstance(data, tuple):
            data = data[0].split(":")[0].upper()
        else:
            data = data.split(":")[0].upper()

        return data

    def send_data(self, data):
        message = f"START{data}END"
        self.client_socket.sendall(message.encode())

    def add_received_queue(self, received):
        self.queue.add("r", received)

    def get_received_queue(self):
        return self.queue.get_all("r")

    def print_serial(self, received):
        """Arduinoからのシリアル通信の内容を出力"""
        if self.debug:
            logger.info("[Serial] {}".format(received))

    color_image, depth_image = (
        np.ones((480, 640, 3), dtype=np.uint8) * 255,
        np.ones((480, 640), dtype=np.uint16) * 65535,
    )

    def make_images(self, header, data):
        if (
            not hasattr(self, "client_socket")
            or self.client_socket is None
            or not self.ser_connected
        ):
            logger.error("[Socket] client_socket is not initialized.")
            return

        try:
            if header == "color":
                nparr = np.frombuffer(data, np.uint8)
                self.color_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif header == "depth":
                nparr = np.frombuffer(data, np.uint16)
                self.depth_image = nparr.reshape((480, 640))
        except Exception as e:
            logger.error(f"[Socket] Error in make_images: {e}")

    def close(self):
        self.send_data("close:")
        self.client_socket.close()
