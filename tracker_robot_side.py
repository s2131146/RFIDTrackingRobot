import socket
import traceback
import serial
import string


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
        id = parts[1]
        data = parts[2] if len(parts) > 1 else ""

        if cmd == "connect":
            self.connect_socket()
        elif cmd == "close":
            self.close()
        elif cmd == "send":
            ret = self.send_serial(data)
            self.send_to_host(ret, id, "send")
        elif cmd == "print":
            res, ret = self.print_serial()
            response = ret if res else "NULL"
            self.send_to_host(response, id, "print")

    def send_to_host(self, data, id=-1, cmd=None):
        str = "{}:{}:{}".format(cmd if cmd is not None else "force", id, data)
        print("[Socket] Sending to host: {}".format(str))
        try:
            client_socket.sendall(str.encode())
        except socket.timeout:
            self.send_to_host(data, id, cmd)
            print("[Socket] Send timed out")
            
    def connect_socket(self):
        """シリアル通信を開始

        Returns:
            bool: 接続結果
        """
        self.send_to_host("[Serial] Connecting to Arduino...")
        self.com = serial.Serial()
        self.com.port = self.ser_port
        self.com.baudrate = self.baud
        self.com.timeout = 0.5
        self.com.dtr = False

        try:
            self.com.open()
            self.com_connected = True
            self.send_to_host("[Serial] Serial port connected")
        except serial.SerialException:
            self.send_to_host("[Serial] Serial port is not open")
        finally:
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
        if not self.com_connected:
            if not self.connect_socket():
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
            self.send_to_host("[Serial] Failed to connect to Arduino.")
            return False

    def print_serial(self):
        """Arduinoからのシリアル通信の内容を出力"""
        try:
            if self.com.in_waiting > 0:
                res = self.com.readline().decode("utf-8", "ignore").strip()
                self.send_to_host("[Serial] Received: {}".format(res))
                return True, res
            else:
                return False, ""
        except serial.SerialException:
            self.send_to_host("[Serial] Failed to connect to Arduino.")
            return False, ""

    def close(self):
        client_socket.close()
        self.com.close()

client_socket = None
server_socket = None
tracker = None

def init():
    global client_socket, server_socket
    server_ip = "0.0.0.0"
    server_port = 8001
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(1)
    server_socket.settimeout(0.1)

    print("[Socket] Waiting for connection...")

    while True:
        try:
            client_socket, client_address = server_socket.accept()
            break
        except socket.timeout:
            continue
    print(f"[Socket] Device connected: {client_address}")

def loop():
    global tracker
    try:
        while True:
            data = client_socket.recv(1024).decode()
            if not data:
                continue
            print("[Socket] {}".format(data))
            if data.startswith("setup:"):
                data = data[len("setup:"):].split(":")
                tracker = TrackerSocketRobot(*data)
            else:
                if tracker is not None:
                    tracker.on_receive(data)
    except ConnectionResetError:
        print("[Socket] Client disconnected")
    except Exception as e:
        print("[Socket] An error occurred: {}\n{}".format(e, traceback.format_exc()))
    finally:
        print("[Socket] Closing sockets...")
        client_socket.close()
        server_socket.close()

def main():
    while True:
        try:
            init()
            loop()
        except KeyboardInterrupt:
            print("[Socket] Interrupted by user")
            break
        except Exception as e:
            print("[Socket] An error occurred in main loop: {}\n{}".format(e, traceback.format_exc()))
            print("[Socket] Restarting...")

if __name__ == "__main__":
    main()