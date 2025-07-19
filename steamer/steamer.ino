#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <WiFi.h>
#include <WebServer.h>
#include <math.h>

// WiFi credentials
const char* ap_ssid = "ESP32_Arm_Control";
const char* ap_password = "password123";

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
float servoAngles[3] = {90.0, 90.0, 90.0};       // current angles
float targetAngles[3] = {90.0, 90.0, 90.0};      // target angles
float speeds[3] = {1.0, 1.0, 1.0};               // deg per update
unsigned long lastUpdateTime = 0;
const unsigned long updateInterval = 20;        // ms

// Convert angle to PWM pulse
int angleToPulse(float angle) {
    angle = constrain(angle, 0, 180);
    return map((int)angle, 0, 270, SERVO_MIN, SERVO_MAX);
}

// Directly set servo (no smooth motion)
void setServoAngle(uint8_t channel, float angle) {
    angle = constrain(angle, 0, 180);
    int pulse = angleToPulse(angle);
    pwm.setPWM(channel, 0, pulse);
    servoAngles[channel] = angle;
}

// Set target for servo (non-blocking move)
void setServoTarget(uint8_t channel, float angle) {
    if (channel < 3) {
        targetAngles[channel] = constrain(angle, 0, 180);
    }
}

// Called regularly to update servo positions
void updateServos() {
    unsigned long now = millis();
    if (now - lastUpdateTime < updateInterval) return;
    lastUpdateTime = now;

    for (int i = 0; i < 3; i++) {
        float diff = targetAngles[i] - servoAngles[i];
        if (abs(diff) > 0.5) {
            float step = constrain(diff, -speeds[i], speeds[i]);
            float newAngle = servoAngles[i] + step;
            setServoAngle(i, newAngle);
        }
    }
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

// ===================== HTTP HANDLERS =====================

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

void handleSetMultipleAngles() {
    for (int i = 0; i < 3; i++) {
        String sName = "servo" + String(i);
        String aName = "angle" + String(i);
        if (server.hasArg(sName) && server.hasArg(aName)) {
            int servo = server.arg(sName).toInt();
            float angle = server.arg(aName).toFloat();
            setServoTarget(servo, angle);
        }
    }
    server.send(200, "text/plain", "Multiple angles set");
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
    server.on("/set_angles", handleSetMultipleAngles);
    server.on("/move_ik", handleMoveIK);
    server.begin();

    // Move to neutral start
    setServoAngle(SERVO1_CHANNEL, 90.0);
    setServoAngle(SERVO2_CHANNEL, 90.0);
    setServoAngle(SERVO3_CHANNEL, 90.0);
}

// ===================== LOOP =====================

void loop() {
    server.handleClient();
    updateServos();
}