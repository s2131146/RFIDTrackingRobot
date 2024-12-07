#ifndef PID_MOTOR_CONTROL_H
#define PID_MOTOR_CONTROL_H

#include <Arduino.h>
#include "HardwareSerial.h"

enum Direction {
    FORWARD,
    REVERSE
};

class PIDMotorControl {
public:
    PIDMotorControl(uint8_t pwmPin, uint8_t dirPin, Direction baseDir = FORWARD, float wheelDiameterMM = 65.0, int maxRPM = 500, float kp = 0.26, float ki = 0.02, float kd = 0);
    void set(float speedMMPS, Direction direction = FORWARD);
    void updateSpeedFromEncoder(int encoderTicks, int ticksPerRevolution, unsigned long deltaTimeMS);
    float getCurrentSpeed();
    Direction getCurrentDirection();
    bool getIsForward();
    void run();
    void stop();
    void reset();

private:
    uint8_t pwmPin;
    uint8_t dirPin;
    float kp, ki, kd;
    float wheelCircumferenceMM;
    int maxPWM;
    float targetSpeedMMPS;
    float currentSpeedMMPS;
    float speed;
    float integral;
    float previousError;
    bool isForward;
    Direction currentDir;
    Direction baseDir;
    int calculatePID();

    void applyOutput(int output, Direction direction);
};

#endif
