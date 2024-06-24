import serial
import tqueue
from constants import Commands


class TrackerSocket:
    com: serial.Serial = None
    com_connected = False

    serial_sent = None
    command_sent = ""

    def __init__(self, p, b, i, d):
        self.port = p
        self.baud = b
        self.interval = i
        self.debug = d

    def connect_socket(self):
        """シリアル通信を開始

        Returns:
            bool: 接続結果
        """
        if self.debug:
            print("[Serial] Connecting to port")
        self.com = serial.Serial()
        self.com.port = self.port
        self.com.baudrate = self.baud
        self.com.timeout = None
        self.com.dtr = False

        try:
            self.com.open()
            self.com_connected = True
            if self.debug:
                print("[Serial] Serial port connected")
        except serial.SerialException:
            if self.debug:
                print("[Serial] Serial port is not open")
        finally:
            return self.com_connected

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
            if self.debug:
                print("[Serial] Serial port is closed.")
            return False
        
    def get_command(self, data):
        if isinstance(data, tuple):
            data = data[0].split(":")[0].upper()
        else:
            data = data.split(":")[0].upper()
            
        return data

    def print_serial(self, received):
        """Arduinoからのシリアル通信の内容を出力"""
        try:
            if self.com.in_waiting > 0:
                res = self.com.readline().decode("utf-8", "ignore").strip()
                if self.debug:
                    print("[Serial] Received: {}".format(res))
                return True, res
            else:
                return False, received
        except serial.SerialException:
            if self.debug:
                print("[Serial] Serial port is closed.")
            return False, ""

    def close(self):
        self.com.close()
