import serial
import time
import re

ser = serial.Serial(
    port="COM4",
    baudrate=38400,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1,
)


def send_command(command):
    ser.write(command.encode())
    time.sleep(0.01)
    response = ser.read(ser.in_waiting)
    return response.decode("ascii")


def read_epc():
    command = "\x0AQ\x0D"  # EPC読み取りコマンド: <LF>Q<CR>
    response = send_command(command)
    response = re.sub(r"[\n\t\r]", "", response).lstrip("Q")
    if len(response) > 1:
        return response
    return None


def read_multiple_epc():
    command = "\x0AU\x0D"  # マルチタグ読み取りコマンド: <LF>U<CR>
    response = send_command(command)
    if response:
        print(f"Multiple EPCs: {response}")


def predict_rf_power(detection_count):
    # 検出頻度を使ってdBm値を推定
    max_detections = 17  # 1秒あたりの最大検出回数
    detection_ratio = min(detection_count / max_detections, 1.0)

    # 電波強度の範囲を仮定 (-100dBmから0dBm)
    min_power = -100
    max_power = 0

    predicted_power = min_power + detection_ratio * (max_power - min_power)

    # 電波強度を百分率に変換
    percentage = (predicted_power - min_power) / (max_power - min_power) * 100

    return percentage


def check_antenna_status():
    command = "\x0AN7,22\x0D"  # Command to check if antenna is open
    response = send_command(command)
    if response.strip():
        return True
    else:
        print("Antenna not found.")
        return False


def monitor_rfid():
    detection_count = 0
    start_time = time.time()

    try:
        while True:
            epc = read_epc()
            if epc:
                detection_count += 1

                # 1秒ごとに検出頻度に基づいて電波強度を予測
                current_time = time.time()
                if current_time - start_time >= 0.5:
                    predicted_power = predict_rf_power(detection_count)
                    print(
                        f"EPC: {epc} Power level: {predicted_power:.2f} Count: {detection_count}"
                    )

                    detection_count = 0
                    start_time = current_time
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if check_antenna_status():
    print("Waiting for RFID...")
    monitor_rfid()
