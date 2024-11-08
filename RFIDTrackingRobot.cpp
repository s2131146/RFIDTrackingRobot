#include "Arduino.h"
#include <Servo.h>

#include <MotorWheel.h>
#include <Omni3WD.h>
#include <Servo.h>

#include "Commands.hpp"

// 各モーターのPIN
#define PIN_MOTOR_LEFT 10
#define PIN_MOTOR_RIGHT 3

enum Motors {
    MOTOR_LEFT,
    MOTOR_RIGHT
};

// 回転角
const int FORWARD = 90;
const int REVERSE = 180;
const int STOP = 0;

// 動作モード
enum OperationMode {
    PID,    // PID制御    *モーターが加速するバグあり
    SERVO   // サーボ制御  * 速度調整不可 
};

Servo wheels[2];

irqISR(irq1, isr1);
MotorWheel wheel1(9, 8, 6, 7, &irq1);        // 中央モーター: Pin9:PWM, Pin8:DIR, Pin6:PhaseA, Pin7:PhaseB

irqISR(irq2, isr2);
MotorWheel wheel2(10, 11, 14, 15, &irq2);    // 左モーター: Pin10:PWM, Pin11:DIR, Pin14:PhaseA, Pin15:PhaseB

irqISR(irq3, isr3);
MotorWheel wheel3(3, 2, 4, 5, &irq3);        // 右モーター: Pin3:PWM, Pin2:DIR, Pin4:PhaseA, Pin5:PhaseB

Omni3WD Omni(&wheel1, &wheel2, &wheel3);

const OperationMode operationMode = PID;

// 速度定数
// MAX: 500
const int MAX_SPEED = 500;
const int MIN_SPEED = 150;

int DEFAULT_SPEED = 300;

int currentSpeed = DEFAULT_SPEED;

const int TIME_APPLY_SPEED = 1500;

namespace RFIDTR {
    // 前回実行したコマンド
    String prevCommand = Commands::STOP;
    int rightSpeed;
    int leftSpeed;

    // 目標速度
    int targetLeftSpeed = DEFAULT_SPEED;
    int targetRightSpeed = DEFAULT_SPEED;

    // 現在の速度
    int currentLeftSpeed = 0;
    int currentRightSpeed = 0;

    // 速度変更の開始時刻
    unsigned long leftSpeedChangeStartTime = 0;
    unsigned long rightSpeedChangeStartTime = 0;

    // 速度変更の状態
    bool leftSpeedChanging = false;
    bool rightSpeedChanging = false;

    // 速度変更のステップ数
    const int SPEED_CHANGE_STEPS = 10;
    // 速度変更の間隔（ms）
    const unsigned long SPEED_CHANGE_INTERVAL = TIME_APPLY_SPEED / SPEED_CHANGE_STEPS;

    // 速度変更のためのカウンタ
    int leftSpeedStep = 0;
    int rightSpeedStep = 0;

    void attachAll() {
        wheels[MOTOR_RIGHT].attach(PIN_MOTOR_RIGHT);
        wheels[MOTOR_LEFT].attach(PIN_MOTOR_LEFT);
    }

    void detachAll() {
        for (int i = 0; i < 2; i++) wheels[i].detach();
    }

    /**
    * @brief 左モーター速度を設定
    *
    * @param percent 左モーターの速度パーセンテージ
    */
    void setLeftMotorSpeed(int percent, bool invert = false, bool fixed = false) {
        int speed = fixed ? percent : ((float)(percent) / 100.0) * DEFAULT_SPEED;
        targetLeftSpeed = speed;

        if (currentLeftSpeed == 0 || targetLeftSpeed == 0) {
            leftSpeedChanging = true;
            leftSpeedChangeStartTime = millis();
            leftSpeedStep = 0;
        }
        else {
            currentLeftSpeed = speed;
            Omni.wheelLeftSetSpeedMMPS(speed, invert ? DIR_BACKOFF : DIR_ADVANCE);
        }

        Omni.PIDRegulate();
    }

    /**
    * @brief 右モーター速度を設定
    *
    * @param percent 右モーターの速度パーセンテージ
    */
    void setRightMotorSpeed(int percent, bool invert = false, bool fixed = false) {
        int speed = fixed ? percent : ((float)(percent) / 100.0) * DEFAULT_SPEED;
        targetRightSpeed = speed;

        if (currentRightSpeed == 0 || targetRightSpeed == 0) {
            rightSpeedChanging = true;
            rightSpeedChangeStartTime = millis();
            rightSpeedStep = 0;
        }
        else {
            currentRightSpeed = speed;
            Omni.wheelRightSetSpeedMMPS(speed, invert ? DIR_ADVANCE : DIR_BACKOFF);
        }

        Omni.PIDRegulate();
    }

    void goLeft() {
        if (operationMode == PID) {
            setLeftMotorSpeed(0);
            setRightMotorSpeed(100);
            Serial.println("Action: Rotate Left");
        } else {
            attachAll();
            wheels[MOTOR_RIGHT].write(FORWARD);
            wheels[MOTOR_LEFT].detach();
        }
    }

    void goRight() {
        if (operationMode == PID) {
            setRightMotorSpeed(0);
            setLeftMotorSpeed(100);
            Serial.println("Action: Rotate Right");
        } else {
            attachAll();
            wheels[MOTOR_LEFT].write(FORWARD);
            wheels[MOTOR_RIGHT].detach();
        }
    }

    void goFoward() {
        if (operationMode == PID) {
            setLeftMotorSpeed(100);
            setRightMotorSpeed(100);
            Serial.println("Action: Forward");
        } else {
            attachAll();
            wheels[MOTOR_LEFT].write(REVERSE);
            wheels[MOTOR_RIGHT].write(REVERSE);
        }
    }

    void goBack() {
        if (operationMode == PID) {
            Omni.setCarBackoff(currentSpeed);
            Serial.println("Action: Go Back");
        } else {
            attachAll();
            wheels[MOTOR_LEFT].write(FORWARD);
            wheels[MOTOR_RIGHT].write(FORWARD);
        }
    }

    void stop() {
        if (operationMode == PID) {
            setLeftMotorSpeed(0);
            setRightMotorSpeed(0);
            Serial.println("All motors stopped.");
        } else {
            detachAll();
        }
    }

    void rotateRight() {
        if (operationMode == PID) {
            leftSpeedChanging = false;
            rightSpeedChanging = false;
            setRightMotorSpeed(150, true, true);
            setLeftMotorSpeed(150, false, true);
        } else {
            attachAll();
            wheels[MOTOR_LEFT].write(FORWARD);
            wheels[MOTOR_RIGHT].write(REVERSE);
        }
    }

    void rotateLeft() {
        if (operationMode == PID) {
            leftSpeedChanging = false;
            rightSpeedChanging = false;
            setRightMotorSpeed(150, false, true);
            setLeftMotorSpeed(150, true, true);
        } else {
            attachAll();
            wheels[MOTOR_LEFT].write(REVERSE);
            wheels[MOTOR_RIGHT].write(FORWARD);
        }
    }

    // コマンドを処理
    void executeCommand(String cmdStr) {
        model::Command cmd(cmdStr);
        String command = cmd.getCommand();
        int value = cmd.getValue();

        if (Commands::is_ignore_same_prev(command, prevCommand)) {
            return;
        }

        if (command == Commands::STOP) {
            stop();
            Serial.println("All motors stopped.");
        }
        else if (command == Commands::L_SPEED) {
            setLeftMotorSpeed(value);
        }
        else if (command == Commands::R_SPEED) {
            setRightMotorSpeed(value);
        }
        else if (command == Commands::START) {
            stop();
            Serial.println("Motors started.");
        }
        else if (command == Commands::GO_LEFT) {
            goLeft();
            Serial.println("Action: Rotate Left");
        }
        else if (command == Commands::GO_RIGHT) {
            goRight();
            Serial.println("Action: Rotate Right");
        }
        else if (command == Commands::GO_BACK) {
            goBack();
            Serial.println("Action: Go Back");
        }
        else if (command == Commands::GO_CENTER) {
            goFoward();
            Serial.println("Action: Forward");
        }
        else if (command == Commands::STOP_TEMP) {
            stop();
        }
        else if (command == Commands::SET_DEFAULT_SPEED) {
            DEFAULT_SPEED = value;
            Serial.println("Set Default speed to " + String(DEFAULT_SPEED));
        }
        else if (command == Commands::ROTATE_RIGHT) {
            rotateRight();
            Serial.println("Action: Rotate Right");
        }
        else if (command == Commands::ROTATE_LEFT) {
            rotateLeft();
            Serial.println("Action: Rotate Left");
        }

        if (operationMode == PID) {
            if (command == Commands::SPD_UP) {
                currentSpeed += 50;
                if (currentSpeed > MAX_SPEED) currentSpeed = MAX_SPEED;
                Omni.setCarAdvance(currentSpeed);
                Serial.println("Speed increased to: " + String(currentSpeed));
            }
            else if (command == Commands::SPD_DOWN) {
                currentSpeed -= 50;
                if (currentSpeed < MIN_SPEED) currentSpeed = MIN_SPEED;
                Omni.setCarAdvance(currentSpeed);
                Serial.println("Speed decreased to: " + String(currentSpeed));
            }
            else if (command == Commands::SPD_RESET) {
                currentSpeed = DEFAULT_SPEED;
                Omni.setCarAdvance(currentSpeed);
                Serial.println("Speed reset to: " + String(currentSpeed));
            }
        }

        prevCommand = command;
    }
} // namespace RFIDTR

// 受信したシリアル通信の内容
String receivedStr = "";

// シリアル通信でコマンドを取得
// realStringUntilを使用すると処理に時間がかかるため
// 意図的にこの処理にしています
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

void setup() {
    Serial.begin(19200);

    // PWM設定: 無くすと爆音になるよ
    TCCR1B = TCCR1B & 0xf8 | 0x01; // Pin9, Pin10 PWM 31250Hz
    TCCR2B = TCCR2B & 0xf8 | 0x01; // Pin3, Pin11 PWM 31250Hz

    // 0.26, 100.0, 10.0, 10    Works fine (Forced)
    // 0.26, 0.02, 0, 10        Works fine
    if (operationMode == PID) {
        Omni.PIDEnable(0.26, 0.02, 0, 10);
    }

    Serial.println("Setup completed");
}

void loop() {    
    if (Serial.available()) {
        String data = getSerialStr();
        RFIDTR::executeCommand(data);
    }

    unsigned long currentTime = millis();

    // 左モーターの速度変更
    if (RFIDTR::leftSpeedChanging) {
        if (currentTime - RFIDTR::leftSpeedChangeStartTime >= RFIDTR::SPEED_CHANGE_INTERVAL) {
            RFIDTR::leftSpeedChangeStartTime = currentTime;
            RFIDTR::leftSpeedStep++;

            // ステップごとの速度計算
            float stepFraction = (float)RFIDTR::leftSpeedStep / RFIDTR::SPEED_CHANGE_STEPS;
            if (stepFraction > 1.0) stepFraction = 1.0;

            // 目標速度に向かって線形補間
            RFIDTR::currentLeftSpeed = RFIDTR::currentLeftSpeed + (RFIDTR::targetLeftSpeed - RFIDTR::currentLeftSpeed) * stepFraction;

            // モーターに設定
            Omni.wheelLeftSetSpeedMMPS(RFIDTR::currentLeftSpeed);

            // ステップが完了したら速度変更を終了
            if (RFIDTR::leftSpeedStep >= RFIDTR::SPEED_CHANGE_STEPS) {
                RFIDTR::leftSpeedChanging = false;
                RFIDTR::currentLeftSpeed = RFIDTR::targetLeftSpeed;
                Omni.wheelLeftSetSpeedMMPS(RFIDTR::currentLeftSpeed);
            }
        }
    }

    // 右モーターの速度変更
    if (RFIDTR::rightSpeedChanging) {
        if (currentTime - RFIDTR::rightSpeedChangeStartTime >= RFIDTR::SPEED_CHANGE_INTERVAL) {
            RFIDTR::rightSpeedChangeStartTime = currentTime;
            RFIDTR::rightSpeedStep++;

            // ステップごとの速度計算
            float stepFraction = (float)RFIDTR::rightSpeedStep / RFIDTR::SPEED_CHANGE_STEPS;
            if (stepFraction > 1.0) stepFraction = 1.0;

            // 目標速度に向かって線形補間
            RFIDTR::currentRightSpeed = RFIDTR::currentRightSpeed + (RFIDTR::targetRightSpeed - RFIDTR::currentRightSpeed) * stepFraction;

            // モーターに設定
            Omni.wheelRightSetSpeedMMPS(RFIDTR::currentRightSpeed, DIR_BACKOFF);

            // ステップが完了したら速度変更を終了
            if (RFIDTR::rightSpeedStep >= RFIDTR::SPEED_CHANGE_STEPS) {
                RFIDTR::rightSpeedChanging = false;
                RFIDTR::currentRightSpeed = RFIDTR::targetRightSpeed;
                Omni.wheelRightSetSpeedMMPS(RFIDTR::currentRightSpeed, DIR_BACKOFF);
            }
        }
    }

    if (operationMode == PID) {
        Omni.PIDRegulate();
    }
}
