import serial
import time
import re


def send_command(ser, command, skip_tag=False):
    """シリアル通信でコマンドを送信し、応答を受信"""
    ser.reset_input_buffer()
    if not skip_tag:
        full_command = f"\x0A{command}\x0D"
    else:
        full_command = command
    ser.write(full_command.encode())
    time.sleep(0.01)
    response = ser.read_all().decode(errors="ignore")
    response = re.sub(r"[\n\t\r]", "", response)
    return response


def switch_antenna(ser, antenna_id):
    """アンテナを切り替えるためのコマンドを送信"""
    base_command = "\x0AN7,22\x0D\x0AN7,11\x0D"
    send_command(ser, base_command, True)

    if antenna_id == 1 or antenna_id == 2:
        command = "N9,20"
    elif antenna_id == 3 or antenna_id == 4:
        command = "N9,22"

    send_command(ser, command)

    if antenna_id == 1 or antenna_id == 4:
        command2 = "N9,10"
    elif antenna_id == 2 or antenna_id == 3:
        command2 = "N9,11"

    send_command(ser, command2)
    print(f"Antenna {antenna_id} switched.")


def read_epc(ser):
    """EPCを読み取る"""
    epc_command = "Q"  # EPC読み取りコマンド
    response = send_command(ser, epc_command)
    print(f"EPC data: {response}")


def main():
    try:
        ser = serial.Serial(port="COM4", baudrate=38400, timeout=1)

        while True:
            for antenna_id in range(1, 5):  # アンテナ1～4を順番に処理
                # switch_antenna(ser, antenna_id)
                read_epc(ser)

    except serial.SerialException as e:
        print(f"Serial connection error: {e}")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
