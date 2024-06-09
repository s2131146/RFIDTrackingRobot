#include <MotorWheel.h>
#include <Omni3WD.h>
#include <PID_Beta6.h>
#include <PinChangeInt.h>
#include <PinChangeIntConfig.h>
#include <SONAR.h>
#include <fuzzy_table.h>

#include "Command.hpp"

/**
 * @brief シリアル通信用ポート番号
 */
const int SERIAL_PORT = 9600;

/**
 * @brief モーター最大出力
 */
const int MOTOR_MAX_POWER = 400;

SONAR sonar11(0x11), sonar12(0x12), sonar13(0x13);

irqISR(irq1, isr1);
MotorWheel wheel1(9, 8, 6, 7,
                  &irq1);  // Pin9:PWM, Pin8:DIR, Pin6:PhaseA, Pin7:PhaseB

irqISR(irq2, isr2);
MotorWheel wheel2(10, 11, 14, 15,
                  &irq2);  // Pin10:PWM, Pin11:DIR, Pin14:PhaseA, Pin15:PhaseB

irqISR(irq3, isr3);
MotorWheel wheel3(3, 2, 4, 5,
                  &irq3);  // Pin3:PWM, Pin2:DIR, Pin4:PhaseA, Pin5:PhaseB

/**
 * @brief オムニホイールオブジェクト
 * 
 * @return Omni3WD 
 */
Omni3WD Omni(&wheel1, &wheel2, &wheel3);

/**
 * @brief ホイールの速度を設定
 * 
 * @param left 
 * @param right 
 */
void setWheelSpeed(int left, int right) {
    /**
     * @brief 開発環境では左右反転
     */
    Omni.wheelLeftSetSpeedMMPS(right);
    Omni.wheelRightSetSpeedMMPS(left);
}

void move(int left, int right) {
    int lp = MOTOR_MAX_POWER * (left / 100);
    int rp = MOTOR_MAX_POWER * (right / 100);

    Omni.wheelRightSetSpeedMMPS(500);
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
 * @brief Setup function of Arduino
 */
void setup() {
    Serial.begin(SERIAL_PORT);
    pinMode(LED_BUILTIN, OUTPUT);
    initSonar();
    move(50, 100);
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

/**
 * @brief コマンドを実行
 * 
 * @param cmdStr コマンド文字列
 */
void execute(String cmdStr) {
    model::Command cmd(cmdStr);
    String command = cmd.getCommand();
    double value = cmd.getValue();

    if (command == "L") {
        digitalWrite(LED_BUILTIN, LOW);
    }
    if (command == "R") {
        digitalWrite(LED_BUILTIN, HIGH);
    }
}

/**
 * @brief Loop function of Arduino
 */
void loop() {
    if (Serial.available()) {
        String data = getSerialStr();
        execute(data);
    }
}
