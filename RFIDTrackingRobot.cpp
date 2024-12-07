#include "Commands.hpp"
#include "PIDDualMotorControl.h"

PIDMotorControl* _motorLeft = nullptr;
PIDMotorControl* _motorRight = nullptr;
PIDDualMotorControl* motor = nullptr;

bool reconnect = false;

namespace RFIDTR {

int DEFAULT_SPEED = 300;
int currentSpeed = DEFAULT_SPEED;
const int TIME_APPLY_SPEED = 1500;

String prevCommand = Commands::STOP;

const float WHEEL_RADIUS = 1.055;
const float WHEEL_BASE = 0.3;

float totalDistance = 0.0;
unsigned long lastUpdateTime = millis();

void calculateDistance() {
    unsigned long currentTime = millis();
    float deltaTime = (currentTime - lastUpdateTime) / 1000.0;
    lastUpdateTime = currentTime;

    float leftSpeed = _motorLeft->getCurrentSpeed() * WHEEL_RADIUS / 1000.0;
    float rightSpeed = _motorRight->getCurrentSpeed() * WHEEL_RADIUS / 1000.0;

    float linearVelocity = (leftSpeed + rightSpeed) / 2.0;
    float angularVelocity = (rightSpeed - leftSpeed) / WHEEL_BASE;

    totalDistance += (linearVelocity * deltaTime);
}

float getTotalDistanceInMeters() { return totalDistance; }

float getMMPS(int percent) {
    return ((float)(percent) / 100.0) * DEFAULT_SPEED;
}

/**
 * @brief 左モーター速度を設定
 *
 * @param percent 左モーターの速度パーセンテージ
 */
void setLeftMotorSpeed(int percent, bool invert = false, bool fixed = false) {
    int speed = fixed ? percent : getMMPS(percent);
    motor->setLeft(speed, invert ? REVERSE : FORWARD);
}

/**
 * @brief 右モーター速度を設定
 *
 * @param percent 右モーターの速度パーセンテージ
 */
void setRightMotorSpeed(int percent, bool invert = false, bool fixed = false) {
    int speed = fixed ? percent : getMMPS(percent);
    motor->setRight(speed, invert ? REVERSE : FORWARD);
}

void goLeft() {
    setLeftMotorSpeed(0);
    setRightMotorSpeed(100);
}

void goRight() {
    setRightMotorSpeed(0);
    setLeftMotorSpeed(100);
}

void goFoward() {
    setLeftMotorSpeed(100);
    setRightMotorSpeed(100);
}

void goBack() {
    motor->set(300, REVERSE);
}

void stop() {
    motor->stop();
}

void rotateRight() {
    motor->rotateRight(250);
}

void rotateLeft() {
    motor->rotateLeft(250);
}

void turnRight() {
    setRightMotorSpeed(40);
    setLeftMotorSpeed(100);
}

void turnLeft() {
    setRightMotorSpeed(100);
    setLeftMotorSpeed(40);
}

unsigned long lastCheck = millis();

// コマンドを処理
void executeCommand(String cmdStr) {
    model::Command cmd(cmdStr);
    String command = cmd.getCommand();
    int value = cmd.getValue();
    int id = cmd.getID();

    if (Commands::is_ignore_same_prev(command, prevCommand)) {
        return;
    }

    // 現在は不使用 (GET_DISTANCEで代用)
    if (command == Commands::CHECK) {
        lastCheck = millis();
    }
    if (command == Commands::GET_DISTANCE) {
        Serial.print(id);
        Serial.print(":");
        Serial.println(getTotalDistanceInMeters());
        lastCheck = millis();
    }
    if (command == Commands::RESET_DISTANCE) {
        totalDistance = 0.0;
        Serial.println("Reset distance.");
    }
    if (command == Commands::RESET_MOTOR) {
        Serial.println("Waiting for recconect...");
        reconnect = true;
    }

    if (command == Commands::STOP) {
        stop();
        Serial.println("All motors stopped.");
    } else if (command == Commands::L_SPEED) {
        setLeftMotorSpeed(value);
    } else if (command == Commands::R_SPEED) {
        setRightMotorSpeed(value);
    } else if (command == Commands::L_SPEED_REV) {
        setLeftMotorSpeed(value, true);
    } else if (command == Commands::R_SPEED_REV) {
        setRightMotorSpeed(value, true);
    } else if (command == Commands::START) {
        stop();
        Serial.println("Motors started.");
    } else if (command == Commands::GO_LEFT) {
        goLeft();
        Serial.println("Action: Rotate Left");
    } else if (command == Commands::GO_RIGHT) {
        goRight();
        Serial.println("Action: Rotate Right");
    } else if (command == Commands::GO_BACK) {
        goBack();
        Serial.println("Action: Go Back");
    } else if (command == Commands::GO_CENTER) {
        goFoward();
        Serial.println("Action: Forward");
    } else if (command == Commands::STOP_TEMP) {
        stop();
    } else if (command == Commands::SET_DEFAULT_SPEED) {
        DEFAULT_SPEED = value;
        Serial.println("Set Default speed to " + String(DEFAULT_SPEED));
    } else if (command == Commands::ROTATE_RIGHT) {
        rotateRight();
        Serial.println("Action: Rotate Right");
    } else if (command == Commands::ROTATE_LEFT) {
        rotateLeft();
        Serial.println("Action: Rotate Left");
    } else if (command == Commands::TURN_LEFT) {
        turnLeft();
        Serial.println("Action: Turn Left");
    } else if (command == Commands::TURN_RIGHT) {
        turnRight();
        Serial.println("Action: Turn Right");
    }

    if (command == Commands::SPD_UP) {
        currentSpeed += 50;
        setLeftMotorSpeed(currentSpeed, false, true);
        setRightMotorSpeed(currentSpeed, false, true);
        Serial.println("Speed increased to: " + String(currentSpeed));
    } else if (command == Commands::SPD_DOWN) {
        currentSpeed -= 50;
        setLeftMotorSpeed(currentSpeed, false, true);
        setRightMotorSpeed(currentSpeed, false, true);
        Serial.println("Speed decreased to: " + String(currentSpeed));
    } else if (command == Commands::SPD_RESET) {
        currentSpeed = DEFAULT_SPEED;
        setLeftMotorSpeed(currentSpeed, false, true);
        setRightMotorSpeed(currentSpeed, false, true);
        Serial.println("Speed reset to: " + String(currentSpeed));
    }

    prevCommand = command;
}
}  // namespace RFIDTR

// 受信したシリアル通信の内容
String receivedStr = "";

// シリアル通信でコマンドを取得
// realStringUntilを使用すると処理に時間がかかるため
// 意図的にこの処理にしています
String getSerialStr() {
    receivedStr = "";
    while (true) {
        while (!Serial.available())
            ;
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
    TCCR1B = TCCR1B & 0xf8 | 0x01;  // Pin9, Pin10 PWM 31250Hz
    TCCR2B = TCCR2B & 0xf8 | 0x01;  // Pin3, Pin11 PWM 31250Hz

    _motorRight = new PIDMotorControl(10, 11, REVERSE);
    _motorLeft = new PIDMotorControl(3, 2);
    motor = new PIDDualMotorControl(_motorLeft, _motorRight);

    Serial.println("Setup completed");
}

bool firstConnect = true;

void loop() {
    if (reconnect) {
        Serial.end();
        Serial.begin(19200);
        reconnect = false;
    }
    if (Serial.available()) {
        if (firstConnect) {
            Serial.println("Connected.");
            firstConnect = false;
        }
        String data = getSerialStr();
        RFIDTR::executeCommand(data);
    }

    unsigned long currentTime = millis();

    if (currentTime - RFIDTR::lastCheck > 500) {
        RFIDTR::stop();
    }

    RFIDTR::calculateDistance();

    motor->run();
}
