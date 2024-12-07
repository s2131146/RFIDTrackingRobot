#include "HardwareSerial.h"
#include "Arduino.h"
#include "PIDDualMotorControl.h"

PIDDualMotorControl::PIDDualMotorControl(PIDMotorControl* leftMotor, PIDMotorControl* rightMotor, float wheelBaseMM)
    : leftMotor(leftMotor), rightMotor(rightMotor), wheelBaseMM(wheelBaseMM) {}

void PIDDualMotorControl::wait(float durationMS, bool defaultRun) {
    if (durationMS != -1) {
        unsigned long startTime = millis();

        while (millis() - startTime < durationMS) {
            if (defaultRun) {
                run();
            } else {
                leftMotor->run();
                rightMotor->run();
            }
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
    if (targetLeftSpeed == 0) {
        leftMotor->stop();
    }
    if (targetRightSpeed == 0) {
        rightMotor->stop();
    }

    bool leftDirChange = targetLeftSpeed >= 0 != (leftMotor->getCurrentDirection() == FORWARD);
    bool rightDirChange = targetRightSpeed >= 0 != (rightMotor->getCurrentDirection() == FORWARD);
    if ((leftMotor->getCurrentSpeed() == 0 && targetLeftSpeed != 0) || leftDirChange) {
        leftMotor->set(350, (targetLeftSpeed >= 0 ? FORWARD : REVERSE));
        boostStartTimeLeft = millis();
    }
    if ((rightMotor->getCurrentSpeed() == 0 && targetRightSpeed != 0) || rightDirChange) {
        rightMotor->set(350, (targetRightSpeed >= 0 ? FORWARD : REVERSE));
        boostStartTimeRight = millis();
    }

    bool boost = false;
    if (millis() - boostStartTimeLeft < 500) {
        boost = true;
        leftMotor->run();
    }
    if (millis() - boostStartTimeRight < 500) {
        boost = true;
        rightMotor->run();
    }
    if (boost) {
        return;
    }
    
    if (targetLeftSpeed != 0 && leftMotor->getCurrentSpeed() != targetLeftSpeed || leftDirChange) {
        leftMotor->set(abs(targetLeftSpeed), targetLeftSpeed > 0 ? FORWARD : REVERSE);
    }
    if (targetRightSpeed != 0 && rightMotor->getCurrentSpeed() != targetRightSpeed || rightDirChange) {
        rightMotor->set(abs(targetRightSpeed), targetRightSpeed >= 0 ? FORWARD : REVERSE);
    }

    leftMotor->run();
    rightMotor->run();
}
