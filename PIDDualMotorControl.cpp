#include "PIDDualMotorControl.h"

PIDDualMotorControl::PIDDualMotorControl(PIDMotorControl* leftMotor, PIDMotorControl* rightMotor, float wheelBaseMM)
    : leftMotor(leftMotor), rightMotor(rightMotor), wheelBaseMM(wheelBaseMM) {}

void PIDDualMotorControl::wait(float durationMS) {
    if (durationMS != -1) {
        unsigned long startTime = millis();

        while (millis() - startTime < durationMS) {
            leftMotor->run();
            rightMotor->run();
        }
    }
}

void PIDDualMotorControl::waitAndStop(float durationMS) {
    wait(durationMS);
    stop();
}

void PIDDualMotorControl::set(float speedMMPS, Direction direction, float durationMS) {
    float leftSpeed = speedMMPS;
    float rightSpeed = speedMMPS;

    if (direction == REVERSE) {
        leftSpeed = -speedMMPS;
        rightSpeed = -speedMMPS;
    }

    calculateAndSetMotorSpeeds(leftSpeed, rightSpeed);
    waitAndStop(durationMS);
}

void PIDDualMotorControl::rotateLeft(float speedMMPS, float durationMS) {
    calculateAndSetMotorSpeeds(-speedMMPS, speedMMPS);
    waitAndStop(durationMS);
}

void PIDDualMotorControl::rotateRight(float speedMMPS, float durationMS) {
    calculateAndSetMotorSpeeds(speedMMPS, -speedMMPS);
    waitAndStop(durationMS);
}

void PIDDualMotorControl::setLeft(float speedMMPS, Direction direction) {
    leftMotor->set(speedMMPS, direction);
}

void PIDDualMotorControl::setRight(float speedMMPS, Direction direction) {
    rightMotor->set(speedMMPS, direction);
}

void PIDDualMotorControl::stop() {
    leftMotor->stop();
    rightMotor->stop();
}

void PIDDualMotorControl::run() {
    leftMotor->run();
    rightMotor->run();
}

void PIDDualMotorControl::update(int leftEncoderTicks, int rightEncoderTicks, int ticksPerRevolution, unsigned long deltaTimeMS) {
    leftMotor->updateSpeedFromEncoder(leftEncoderTicks, ticksPerRevolution, deltaTimeMS);
    rightMotor->updateSpeedFromEncoder(rightEncoderTicks, ticksPerRevolution, deltaTimeMS);
}

void PIDDualMotorControl::calculateAndSetMotorSpeeds(float leftSpeedMMPS, float rightSpeedMMPS) {
    if (currentLeftSpeed != leftSpeedMMPS) {
        leftMotor->set(leftSpeedMMPS, leftSpeedMMPS > 0 ? FORWARD : REVERSE);
        currentLeftSpeed = leftSpeedMMPS;
    }
    if (currentRightSpeed != rightSpeedMMPS) {
        rightMotor->set(rightSpeedMMPS, rightSpeedMMPS > 0 ? FORWARD : REVERSE);
        currentRightSpeed = rightSpeedMMPS;
    }
}
