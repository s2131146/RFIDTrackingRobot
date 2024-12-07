#include <math.h>
#include "PIDMotorControl.h"

#ifndef PID_DUAL_MOTOR_CONTROL_H
#define PID_DUAL_MOTOR_CONTROL_H

class PIDDualMotorControl {
public:
    PIDMotorControl* leftMotor;
    PIDMotorControl* rightMotor;
    
    PIDDualMotorControl(PIDMotorControl* leftMotor, PIDMotorControl* rightMotor, float wheelBaseMM = 150.0);
    void waitAndStop(float durationMS);
    void wait(float durationMS);
    void set(float speedMMPS, Direction direction = FORWARD, float durationMS = -1);
    void rotateLeft(float speedMMPS, float durationMS = -1);
    void rotateRight(float speedMMPS, float durationSec = -1);
    void setLeft(float speedMMPS, Direction direction = FORWARD);
    void setRight(float speedMMPS, Direction direction = FORWARD);
    void stop();
    void run();
    void update(int leftEncoderTicks, int rightEncoderTicks, int ticksPerRevolution, unsigned long deltaTimeMS);

private:
    float targetLeftSpeed;
    float targetRightSpeed;
    float currentTargetLeftSpeed;
    float currentTargetRightSpeed;
    float wheelBaseMM;

    unsigned long leftSpeedChangeStartTime;
    unsigned long rightSpeedChangeStartTime;

    const int SPEED_CHANGE_DURATION = 1500;

    void calculateAndSetMotorSpeeds(float leftSpeedMMPS, float rightSpeedMMPS);
};

#endif
