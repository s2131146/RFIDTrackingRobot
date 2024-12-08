/**
 * @file PIDDualMotorControl.h
 * @author Metoki Towa
 * @brief オムニホイールロボット制御プログラム
 * @version 1.0
 * @date 2024-12-07
 */
#ifndef PID_DUAL_MOTOR_CONTROL_H
#define PID_DUAL_MOTOR_CONTROL_H

#include <math.h>
#include "PIDMotorControl.h"

class PIDDualMotorControl {
public:
    PIDMotorControl* leftMotor;
    PIDMotorControl* rightMotor;
    
    PIDDualMotorControl(PIDMotorControl* leftMotor, PIDMotorControl* rightMotor, float wheelBaseMM = 150.0);
    void waitAndStop(float durationMS);
    void wait(float durationMS, bool defaultRun = true);
    void set(float speedMMPS, Direction direction = FORWARD, float durationMS = -1);
    void rotateLeft(float speedMMPS, float durationMS = -1);
    void rotateRight(float speedMMPS, float durationSec = -1);
    void setLeft(float speedMMPS, Direction direction = FORWARD);
    void setRight(float speedMMPS, Direction direction = FORWARD);
    void stop(float durationMS = -1);
    void run();
    void update(int leftEncoderTicks, int rightEncoderTicks, int ticksPerRevolution, unsigned long deltaTimeMS);

private:
    float targetLeftSpeed;
    float targetRightSpeed;
    float currentTargetLeftSpeed;
    float currentTargetRightSpeed;
    float wheelBaseMM;
    float boostStartTimeLeft;
    float boostStartTimeRight;
    void calculateAndSetMotorSpeeds(float leftSpeedMMPS, float rightSpeedMMPS);
};

#endif
