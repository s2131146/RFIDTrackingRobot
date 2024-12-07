#include "HardwareSerial.h"
#include "Arduino.h"
#include "PIDDualMotorControl.h"

PIDDualMotorControl::PIDDualMotorControl(PIDMotorControl* leftMotor, PIDMotorControl* rightMotor, float wheelBaseMM)
    : leftMotor(leftMotor), rightMotor(rightMotor), wheelBaseMM(wheelBaseMM) {}

void PIDDualMotorControl::wait(float durationMS) {
    if (durationMS != -1) {
        unsigned long startTime = millis();

        while (millis() - startTime < durationMS) {
            run();
        }
    }
}

void PIDDualMotorControl::waitAndStop(float durationMS) {
    if (durationMS != -1) {
        wait(durationMS);
        stop();
    }
}

void PIDDualMotorControl::set(float speedMMPS, Direction direction, float durationMS) {
    float leftSpeed = speedMMPS;
    float rightSpeed = speedMMPS;

    if (direction == REVERSE) {
        leftSpeed = -speedMMPS;
        rightSpeed = -speedMMPS;
    }

    targetLeftSpeed = leftSpeed;
    targetRightSpeed = rightSpeed;
    waitAndStop(durationMS);
}

void PIDDualMotorControl::rotateLeft(float speedMMPS, float durationMS) {
    targetLeftSpeed = -speedMMPS;
    targetRightSpeed = speedMMPS;
    waitAndStop(durationMS);
}

void PIDDualMotorControl::rotateRight(float speedMMPS, float durationMS) {
    targetLeftSpeed = speedMMPS;
    targetRightSpeed = -speedMMPS;
    waitAndStop(durationMS);
}

void PIDDualMotorControl::setLeft(float speedMMPS, Direction direction) {
    targetLeftSpeed = (direction == FORWARD) ? speedMMPS : -speedMMPS;
}

void PIDDualMotorControl::setRight(float speedMMPS, Direction direction) {
    targetRightSpeed = (direction == FORWARD) ? speedMMPS : -speedMMPS;
}

void PIDDualMotorControl::stop() {
    targetLeftSpeed = 0;
    targetRightSpeed = 0;
    leftMotor->stop();
    rightMotor->stop();
}

void PIDDualMotorControl::run() {
    float currentTime = millis();
    float currentLeftSpeed = leftMotor->getCurrentSpeed();
    float currentRightSpeed = rightMotor->getCurrentSpeed();

    if (currentTargetLeftSpeed != targetLeftSpeed || targetLeftSpeed >= 0 != (leftMotor->getCurrentDirection() == FORWARD)) {
        leftSpeedChangeStartTime = currentTime;
        currentTargetLeftSpeed = targetLeftSpeed;
        if (currentLeftSpeed == 0) {
            leftMotor->set(350, (targetLeftSpeed >= 0 ? FORWARD : REVERSE));
            wait(500);
        }
    }
    if (currentTargetRightSpeed != targetRightSpeed || targetRightSpeed >= 0 != (leftMotor->getCurrentDirection() == FORWARD)) {
        rightSpeedChangeStartTime = currentTime;
        currentTargetRightSpeed = targetRightSpeed;
        if (currentRightSpeed == 0) {
            rightMotor->set(350, (targetRightSpeed >= 0 ? FORWARD : REVERSE));
            wait(500);
        }
    }

    unsigned long timeElapsedL = currentTime - leftSpeedChangeStartTime;
    if (timeElapsedL > SPEED_CHANGE_DURATION) timeElapsedL = SPEED_CHANGE_DURATION;
    float progress = (float)timeElapsedL / SPEED_CHANGE_DURATION;
    currentLeftSpeed = (1.0 - progress) * currentLeftSpeed + progress * targetLeftSpeed;
    leftMotor->set(abs(currentLeftSpeed), leftMotor->getCurrentDirection());

    unsigned long timeElapsedR = currentTime - rightSpeedChangeStartTime;
    if (timeElapsedR > SPEED_CHANGE_DURATION) timeElapsedR = SPEED_CHANGE_DURATION;
    progress = (float)timeElapsedR / SPEED_CHANGE_DURATION;
    currentRightSpeed = (1.0 - progress) * currentRightSpeed + progress * targetRightSpeed;
    rightMotor->set(abs(currentRightSpeed), rightMotor->getCurrentDirection());

    leftMotor->run();
    rightMotor->run();
}
