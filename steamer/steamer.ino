#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <WiFi.h>
#include <WebServer.h>
#include <math.h>

// WiFi credentials
const char* ap_ssid = "ESP32_Arm_Control";
const char* ap_password = "password123";

// Potentiometer pin
const uint8_t potPin = 34;  // Try 34, 35, 36, or 39
int potValue = 0;

// Web server on port 80
WebServer server(80);

// I2C pins
#define SDA_PIN 26
#define SCL_PIN 25

// Servo controller
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo channels
const int SERVO1_CHANNEL = 1;  // Shoulder
const int SERVO2_CHANNEL = 0;  // Elbow
const int SERVO3_CHANNEL = 2;  // Base rotation

// Arm dimensions (cm)
const float L1 = 15.00;
const float L2 = 13.40;

// PWM pulse width limits
const int SERVO_MIN = 105;
const int SERVO_MAX = 512;

// Servo state tracking
enum Motor {
    RIGHT_BASE,
    RIGHT_SHOULDER,
    RIGHT_ELBOW,
    LEFT_BASE,
    LEFT_SHOULDER,
    LEFT_ELBOW,
    NUM_MOTORS
};

float servoAngles[Motor::NUM_MOTORS];       // current angles
float targetAngles[Motor::NUM_MOTORS];      // target angles
float actuatorPins[Motor::NUM_MOTORS] = {0,1,2,3,4,5}; // pins of left, then riight arms. Defined in enum Motor
float speeds[Motor::NUM_MOTORS];            // deg per updateInterva
unsigned long lastUpdateTime[Motor::NUM_MOTORS];

const unsigned long updateInterval = 20;        // ms

// Read potentiometer value
int readPotentiometer(uint8_t potPin) {
    potValue = analogRead(potPin);  // Read the analog value (0–4095 on ESP32)
    return potValue;
}

// Convert angle to PWM pulse
int angleToPulse(float angle) {
    angle = constrain(angle, 0, 180);
    return map((int)angle, 0, 270, SERVO_MIN, SERVO_MAX);
}

// Inverse kinematics
bool inverseKinematics(float x_d, float y_d, float& theta1, float& theta2) {
    const float L1 = 15.0;
    const float L2 = 13.4;

    float r = sqrt(x_d * x_d + y_d * y_d);
    if (r > (L1 + L2) || r < fabs(L1 - L2)) {
        Serial.println("Error: Position not reachable");
        return false;
    }

    float cos_theta2 = (x_d * x_d + y_d * y_d - L1 * L1 - L2 * L2) / (2 * L1 * L2);
    if (cos_theta2 < -1 || cos_theta2 > 1) {
        Serial.println("Error: Invalid cosine value");
        return false;
    }

    float theta2_rad = acos(cos_theta2);
    float theta2_deg = theta2_rad * 180.0 / PI;

    float k1 = L1 + L2 * cos(theta2_rad);
    float k2 = L2 * sin(theta2_rad);
    float theta1_rad = atan2(y_d, x_d) - atan2(k2, k1);
    float theta1_deg = theta1_rad * 180.0 / PI;
    if (theta1_deg < 0) theta1_deg += 360;

    if (theta1_deg > 270 || theta2_deg > 270) {
        theta2_rad = -acos(cos_theta2);
        theta2_deg = theta2_rad * 180.0 / PI;
        k1 = L1 + L2 * cos(theta2_rad);
        k2 = L2 * sin(theta2_rad);
        theta1_rad = atan2(y_d, x_d) - atan2(k2, k1);
        theta1_deg = theta1_rad * 180.0 / PI;
        if (theta1_deg < 0) theta1_deg += 360;

        if (theta1_deg > 270 || theta2_deg < 0 || theta2_deg > 270) {
            Serial.println("Error: Angles out of range");
            return false;
        }
    }

    theta1 = theta1_deg;
    theta2 = theta2_deg;
    return true;
}

// Directly set servo (no smooth motion)
void setServoAngle(uint8_t channel, float angle) {
    angle = constrain(angle, 0, 180);
    int pulse = angleToPulse(angle);
    // Serial.printf("Arm %s set to %.1f degrees\n", channel == RIGHT_BASE ? "Right Base" : channel == RIGHT_SHOULDER ? "Right Shoulder" : channel == RIGHT_ELBOW ? "Right Elbow" : channel == LEFT_BASE ? "Left Base" : channel == LEFT_SHOULDER ? "Left Shoulder" : "Left Elbow", angle);
    pwm.setPWM(actuatorPins[channel], 0, pulse);
    servoAngles[channel] = angle;
}

// Called regularly to update servo positions
void updateServos(uint8_t channel) {
    unsigned long now = millis();
    if (now - lastUpdateTime[channel] < updateInterval) return;
    lastUpdateTime[channel] = now;

    float diff = targetAngles[channel] - servoAngles[channel];
    if (abs(diff) > 0.5) {
        float step = constrain(diff, -speeds[channel], speeds[channel]);
        float newAngle = servoAngles[channel] + step;
        setServoAngle(channel, newAngle);
    }
}

void handleSetAngle() {
    if (server.hasArg("servo") && server.hasArg("angle")) {
        int servo = server.arg("servo").toInt();
        float angle = server.arg("angle").toFloat();
        setServoTarget(servo, angle);
        server.send(200, "text/plain", "Target angle set");
    } else {
        server.send(400, "text/plain", "Missing parameters");
    }
}

// Set target for servo (non-blocking move)
void setServoTarget(uint8_t channel, float angle) {
    if ((channel == LEFT_BASE) || (channel == LEFT_SHOULDER)) {
      angle = 180 - angle;
    }
    targetAngles[channel] = constrain(angle, 0, 180);
}

void handleSetMultipleActuators() {

    String rightArmBase     = "rightArmBaseAngle";
    String leftArmBase      = "leftArmBaseAngle";
    String rightArmShoulder = "rightArmShoulderAngle";
    String leftArmShoulder  = "leftArmShoulderAngle";
    String rightArmElbow    = "rightArmElbowAngle";
    String leftArmElbow     = "leftArmElbowAngle";
    String speed            = "speed";

    if (server.hasArg(rightArmBase)) {
      float angle = server.arg(rightArmBase).toFloat();
      setServoTarget(RIGHT_BASE, angle);
    }

    if (server.hasArg(leftArmBase)) {
      float angle = server.arg(leftArmBase).toFloat();
      setServoTarget(LEFT_BASE, angle);
    }

    if (server.hasArg(rightArmShoulder)) {
      float angle = server.arg(rightArmShoulder).toFloat();
      setServoTarget(RIGHT_SHOULDER, angle);
    }

    if (server.hasArg(leftArmShoulder)) {
      float angle = server.arg(leftArmShoulder).toFloat();
      setServoTarget(LEFT_SHOULDER, angle);
    }

    if (server.hasArg(rightArmElbow)) {
      float angle = server.arg(rightArmElbow).toFloat();
      setServoTarget(RIGHT_ELBOW, angle);
    }

    if (server.hasArg(leftArmElbow)) {
      float angle = server.arg(leftArmElbow).toFloat();
      setServoTarget(LEFT_ELBOW, angle);
    }

    if (server.hasArg(speed)) {
      for (int i = 0; i < Motor::NUM_MOTORS; i++) {
          speeds[i] = server.arg(speed).toFloat();
      }
    }

    // TODO feature read pot from all angles
    int potReading = readPotentiometer(0);
    String response = "Multiple angles set, Potentiometer value: " + String(potReading);
    server.send(200, "text/plain", response);
}

void handleMoveIK() {
    if (server.hasArg("x") && server.hasArg("y")) {
        float x = server.arg("x").toFloat();
        float y = server.arg("y").toFloat();
        float theta1, theta2;
        if (inverseKinematics(x, y, theta1, theta2)) {
            setServoTarget(SERVO1_CHANNEL, theta1);
            setServoTarget(SERVO3_CHANNEL, theta2);
            server.send(200, "text/plain", "Moved to position");
        } else {
            server.send(400, "text/plain", "IK failed");
        }
    } else {
        server.send(400, "text/plain", "Missing parameters");
    }
}

// ===================== SETUP =====================

void setup() {
    Serial.begin(115200);
    Wire.begin(SDA_PIN, SCL_PIN);
    pwm.begin();
    pwm.setPWMFreq(50);

    WiFi.softAP(ap_ssid, ap_password);
    IPAddress myIP = WiFi.softAPIP();
    Serial.print("AP IP address: ");
    Serial.println(myIP);

    server.on("/set_angle", handleSetAngle);
    server.on("/set_actuators", handleSetMultipleActuators);
    server.on("/move_ik", handleMoveIK);
    server.begin();

    for (int i = 0; i < Motor::NUM_MOTORS; i++) {
        servoAngles[i] = 0.0;
        targetAngles[i] = 0.0;
        speeds[i] = 2.0;
        lastUpdateTime[i] = 0.0;
    }

    // Move to neutral start
    setServoTarget(RIGHT_BASE, 90);
    setServoTarget(RIGHT_SHOULDER, 90);
    setServoTarget(RIGHT_ELBOW, 90);
    setServoTarget(LEFT_BASE, 90);
    setServoTarget(LEFT_SHOULDER, 90);
    setServoTarget(LEFT_ELBOW, 90);
}

// ===================== LOOP =====================

void loop() {
    server.handleClient();
    updateServos(RIGHT_BASE);
    updateServos(RIGHT_SHOULDER);
    updateServos(RIGHT_ELBOW);
    updateServos(LEFT_BASE);
    updateServos(LEFT_SHOULDER);
    updateServos(LEFT_ELBOW);
}