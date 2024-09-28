#include <MotorWheel.h>
#include <Omni3WD.h>
#include <PID_Beta6.h>
#include <PinChangeInt.h>
#include <PinChangeIntConfig.h>
#include <SONAR.h>
#include <fuzzy_table.h>

#include "Command.hpp"

/**
 * @brief RFID Tracking Robot
 */
namespace RFIDTR {
/**
 * @brief シリアル通信用ポート番号
 */
const int SERIAL_PORT = 19200;

/**
 * @brief モーター最大出力
 */
const int MOTOR_MAX_POWER = 400;

SONAR sonar11(0x11), sonar12(0x12), sonar13(0x13);

irqISR(irq1, isr1);
/// @brief 中央モーター
MotorWheel wheel2(9, 8, 6, 7, &irq1);

irqISR(irq2, isr2);
/// @brief 左モーター
MotorWheel wheel3(10, 11, 14, 15, &irq2);

irqISR(irq3, isr3);
/// @brief 右モーター
MotorWheel wheel1(3, 2, 4, 5, &irq3);

/**
 * @brief オムニホイールオブジェクト
 *
 * @return Omni3WD
 */
Omni3WD Omni(&wheel1, &wheel2, &wheel3);

int speedL = 0, speedR = 0, dirL = 0, dirR = 0;

/**
 * @brief モーター出力を変更
 *
 * @param left 左モーター出力% (-1で変更なし)
 * @param right 右モーター出力% (-1で変更なし)
 */
void setWheelSpeed(int left = -1, int right = -1, bool force = false) {
    int lp, rp;

    if (left == -1) {
        lp = speedL;
    } else {
        lp = floorf(MOTOR_MAX_POWER * (left / 100.0));
    }
    if (right == -1) {
        rp = speedR;
    } else {
        rp = floorf(MOTOR_MAX_POWER * (right / 100.0));
    }

    if (left == 0 && right == 0) {
        if (speedL != 0 || speedR != 0 || force) {
            Serial.println("Stopped");
            if (dirL == 0) {
                Omni.wheelLeftSetSpeedMMPS(lp);
            } else {
                Omni.wheelLeftSetSpeedMMPS(lp, DIR_BACKOFF);
            }
            if (dirR == 0) {
                Omni.wheelRightSetSpeedMMPS(rp, DIR_BACKOFF);
            } else {
                Omni.wheelRightSetSpeedMMPS(rp);
            }
            speedL = 0;
            speedR = 0;
        }
    } else {
        if (left != -1 || force) {
            if (lp != speedL || force) {
                if (!force) {
                    Serial.println("Set left speed: " + String(lp));
                }
                if (dirL == 0) {
                    Omni.wheelLeftSetSpeedMMPS(lp);
                } else {
                    Omni.wheelLeftSetSpeedMMPS(lp, DIR_BACKOFF);
                }
                speedL = lp;
            }
        }
        if (right != -1 || force) {
            if (rp != speedR || force) {
                if (!force) {
                    Serial.println("Set right speed: " + String(rp));
                }
                if (dirR == 0) {
                    Omni.wheelRightSetSpeedMMPS(rp, DIR_BACKOFF);
                } else {
                    Omni.wheelRightSetSpeedMMPS(rp);
                }
            }
            speedR = rp;
        }
    }

    Omni.PIDRegulate();
}

/**
 * @brief ソナーを初期化
 */
void initSonar() {
    TCCR1B = TCCR1B & 0xf8 | 0x01;  // Pin9,Pin10 PWM 31250Hz
    TCCR2B = TCCR2B & 0xf8 | 0x01;  // Pin3,Pin11 PWM 31250Hz

    SONAR::init(13);

    Omni.PIDEnable(0.26, 0.02, 0, 10);
}

/**
 * @brief 受信したシリアル通信の内容
 */
String receivedStr = "";

/**
 * @brief シリアル通信で文字列を受信 (EOFまで)
 *
 * @return String 受信した文字列
 */
String getSerialStr() {
    receivedStr = "";
    while (true) {
        while (!Serial.available());
        char c = Serial.read();
        if (c != '\n') {
            receivedStr += c;
        } else {
            break;
        }
    }

    return receivedStr;
}

unsigned long startMillis;
String prevCommand;

/**
 * @brief コマンドを実行
 *
 * @param cmdStr コマンド文字列
 */
void executeCommand(String cmdStr) {
    model::Command cmd(cmdStr);
    String command = cmd.getCommand();
    double value = cmd.getValue();
    int i_value = (int)value;

    if (cmdStr != "CONNECT_WHEEL") {
        prevCommand = cmdStr;
    }

    if (command == "YO") {
        startMillis = millis();
        return;
    }
    if (command == "L") {
        setWheelSpeed(i_value, -1);
        digitalWrite(LED_BUILTIN, HIGH);
    }
    if (command == "R") {
        setWheelSpeed(-1, i_value);
    }
    if (command == "L_DIRECTION") {
        dirL = i_value;
        setWheelSpeed(-1, -1, true);
    }
    if (command == "R_DIRECTION") {
        dirR = i_value;
        setWheelSpeed(-1, -1, true);
    }
    if (command == "STOP" || command == "STMP") {
        setWheelSpeed(0, 0);
    }
    if (command == "CONNECT_WHEEL") {
        setWheelSpeed(-1, -1, true);
    }
}
}  // namespace RFIDTR

/**
 * @brief Setup function of Arduino
 */
void setup() {
    Serial.begin(RFIDTR::SERIAL_PORT);
    pinMode(LED_BUILTIN, OUTPUT);
    RFIDTR::initSonar();
    Serial.println("Setup completed");
}

void (*reset)(void) = 0;

/**
 * @brief Loop function of Arduino
 */
void loop() {
    if (!Serial) {
        delay(500);
        reset();
        return;
    }

    if (Serial.available()) {
        String data = RFIDTR::getSerialStr();
        RFIDTR::executeCommand(data);
    } else {
        if (millis() - RFIDTR::startMillis > 1000) {
            RFIDTR::executeCommand("STOP");
        } else {
            RFIDTR::executeCommand(RFIDTR::prevCommand);
            RFIDTR::executeCommand("CONNECT_WHEEL");
        }
    }
}
