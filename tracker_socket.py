import threading
import socket
import traceback
import tqueue
import asyncio
from utils import Utils


class TrackerSocket:
    serial_sent = None
    command_sent = ""
    queue = tqueue.TQueue()
    data_id = 0

    def __init__(self, p, b, i, d, port):
        self.ser_port = p
        self.baud = b
        self.interval = i
        self.debug = d
        self.tcp_ip = TrackerSocket.get_local_ip()
        self.tcp_port = port

    @classmethod
    def get_local_ip(self):
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            return ip_address
        except socket.error as e:
            print("[Socket] An error occured during getting local IP: {}".format(e))
            return None
        
    def test_connect(self):
        try:
            test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test.settimeout(0.5)
            test.connect((self.tcp_ip, self.tcp_port))
            test.close()
            return True
        except socket.error as e:
            print("[Socket] An error occured at connect_socket: {}".format(traceback.format_exc()))
            return False

    async def connect_socket(self):
        """シリアル通信を開始

        Returns:
            bool: 接続結果
        """
        if self.tcp_ip is None:
            print("[Socket] IP Address is not defined.")
            self.ser_connected = False
            return
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(0.5)
            self.client_socket.setblocking(False)
            self.socket_loop = asyncio.get_event_loop()
            await self.socket_loop.sock_connect(self.client_socket, (self.tcp_ip, self.tcp_port))
            self.ser_connected = True
            print("[Socket] Socket connected.")
        except socket.error as e:
            print("[Socket] An error occured at connect_socket: {}".format(traceback.format_exc()))
            self.ser_connected = False
        finally:
            return self.ser_connected

    async def loop_serial(self):
        try:
            while True:
                data = await self.socket_loop.sock_recv(self.client_socket, 1024)
                if not data:
                    continue
                data = str(data.decode())
                parts = data.split(":")
                cmd = parts[0]
                id = parts[1]
                value = parts[2]
                value_bool = Utils.try_to_bool(value)
                if cmd == "force":
                    print(value)
                if cmd == "print":
                    self.print_serial(value)
                if id != -1:
                    if self.queue.get("w:{}".format(id)) is not None:
                        self.queue.add(id, value_bool)
        except Exception as e:
            print("[Socket] An error occured at loop_serial: {}".format(traceback.format_exc()))
        finally:
            self.client_socket.close()

    def send_startup(self):
        self.client_socket.sendall(
            "setup:{}:{}:{}:{}:{}",
            self.ser_port,
            self.baud,
            self.interval,
            self.tcp_ip,
            self.tcp_port,
        )

    def wait_for_result(self, data, id):
        self.queue.add("w:{}".format(id))
        self.client_socket.sendall(data.encode())
        while not self.queue.has(id):
            pass
        return self.queue.get(id)

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

        self.data_id += 1
        if isinstance(data, tuple):
            data = "send:{}:{}:{}".format(self.data_id, data[0], data[1])
        else:
            data = "send:{}:{}".format(self.data_id, data.upper())

        self.command_sent = self.get_command(data).upper()
        self.serial_sent = data
        data += "\n"

        try:
            self.wait_for_result(data, self.data_id)
            return True
        except socket.error as e:
            self.ser_connected = False
            if self.debug:
                print("[Socket] Failed to send to client: {}".format(traceback.format_exc()))
            return False

    def get_command(self, data):
        if isinstance(data, tuple):
            data = data[0].split(":")[0].upper()
        else:
            data = data.split(":")[0].upper()

        return data

    def add_received_queue(self, received):
        self.queue.add("r", received)

    def get_received_queue(self):
        return self.queue.get_all("r")

    def print_serial(self, received):
        """Arduinoからのシリアル通信の内容を出力"""
        if self.debug:
            print("[Serial] {}".format(received))

    def close(self):
        self.client_socket.sendall("close:")
        self.client_socket.close()
