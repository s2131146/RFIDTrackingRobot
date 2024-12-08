/**
 * @file PIDMotorControl.cpp
 * @author Metoki Towa
 * @brief PIDを用いたモーター制御プログラム
 * @version 1.0
 * @date 2024-12-07
 */
#include "PIDMotorControl.h"

PIDMotorControl::PIDMotorControl(uint8_t pwmPin, uint8_t dirPin, Direction baseDir, float wheelDiameterMM, int maxRPM, float kp, float ki, float kd)
    : pwmPin(pwmPin), dirPin(dirPin), baseDir(baseDir),
      wheelCircumferenceMM(wheelDiameterMM * PI), maxPWM(255), kp(kp), ki(ki), kd(kd), currentDir(FORWARD),
      targetSpeedMMPS(0), currentSpeedMMPS(0), integral(0), previousError(0), speed(0), isForward(true) {
    pinMode(pwmPin, OUTPUT);
    pinMode(dirPin, OUTPUT);
    stop();
}

float PIDMotorControl::getCurrentSpeed() {
    return speed;
}

Direction PIDMotorControl::getCurrentDirection() {
    return currentDir;
}

bool PIDMotorControl::getIsForward() {
    return isForward;
}

void PIDMotorControl::set(float speedMMPS, Direction direction) {
    speed = speedMMPS;
    targetSpeedMMPS = constrain(speedMMPS, -wheelCircumferenceMM * maxPWM / 60.0, wheelCircumferenceMM * maxPWM / 60.0);
    currentDir = direction;
    run();
}

void PIDMotorControl::run() {
    int output = calculatePID();
    applyOutput(output, currentDir);
}

void PIDMotorControl::stop() {
    targetSpeedMMPS = 0;
    speed = 0;
    analogWrite(pwmPin, 0);
    digitalWrite(dirPin, LOW);
    reset();
}

void PIDMotorControl::reset() {
    integral = 0;
    previousError = 0;
}

int PIDMotorControl::calculatePID() {
    float error = abs(targetSpeedMMPS) - currentSpeedMMPS;

    integral += error;
    integral = constrain(integral, -1000, 1000);

    float derivative = error - previousError;
    previousError = error;

    int output = kp * error + ki * integral + kd * derivative;

    return constrain(output, -maxPWM, maxPWM);
}

void PIDMotorControl::applyOutput(int output, Direction direction) {
    isForward = direction == FORWARD ? output >= 0 : output < 0;
    isForward = baseDir == FORWARD ? isForward : !isForward;
    int pwmValue = abs(output);

    digitalWrite(dirPin, isForward ? HIGH : LOW);
    analogWrite(pwmPin, pwmValue);
}
