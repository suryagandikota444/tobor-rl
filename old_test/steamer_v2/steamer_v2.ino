#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <WiFi.h>
#include <WebServer.h>
#include <math.h>

// ======================= CONFIGURATION ========================= //
namespace Config {
  const char* AP_SSID = "ESP32_Arm_Control";
  const char* AP_PASSWORD = "password123";

  const int SDA_PIN = 26;
  const int SCL_PIN = 25;

  const float L1 = 15.00;   // Upper arm
  const float L2 = 13.40;   // Lower arm

  const float MAX_SPEED_DEG_PER_UPDATE = 3.0;

  const float SERVO_MIN_ANGLE = 0.0;
  const float SERVO_MAX_ANGLE = 180.0;

  const int PWM_MIN = 105;
  const int PWM_MAX = 512;

  const int SERVO1_CHANNEL = 1;  // Shoulder
  const int SERVO2_CHANNEL = 2;  // Elbow
  const int SERVO3_CHANNEL = 0;  // Base rotation

  const float ORIGIN_X = 0.0;  // Define the base of arm as (0, 0)
  const float ORIGIN_Y = 0.0;
}

// ======================= GLOBALS ========================= //
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
WebServer server(80);

float lastAngles[3] = {90.0, 90.0, 90.0};  // Safe neutral startup

// ======================= UTILITIES ========================= //
int angleToPWM(float angle) {
  angle = constrain(angle, Config::SERVO_MIN_ANGLE, Config::SERVO_MAX_ANGLE);
  return map((int)angle, 0, 180, Config::PWM_MIN, Config::PWM_MAX);
}

void safeSetServoAngle(uint8_t channel, float targetAngle) {
  targetAngle = constrain(targetAngle, Config::SERVO_MIN_ANGLE, Config::SERVO_MAX_ANGLE);
  float& current = lastAngles[channel];

  // Smooth transition to target angle
  float delta = targetAngle - current;
  delta = constrain(delta, -Config::MAX_SPEED_DEG_PER_UPDATE, Config::MAX_SPEED_DEG_PER_UPDATE);
  current += delta;

  pwm.setPWM(channel, 0, angleToPWM(current));
}

void moveArm(float theta1, float theta2, float baseAngle) {
  safeSetServoAngle(Config::SERVO1_CHANNEL, theta1);
  safeSetServoAngle(Config::SERVO2_CHANNEL, theta2);
  safeSetServoAngle(Config::SERVO3_CHANNEL, baseAngle);
}

// ======================= INVERSE KINEMATICS ========================= //
bool computeIK(float x, float y, float& theta1_deg, float& theta2_deg) {
  x -= Config::ORIGIN_X;
  y -= Config::ORIGIN_Y;

  float dist2 = x * x + y * y;
  float c2 = (dist2 - Config::L1 * Config::L1 - Config::L2 * Config::L2) / (2 * Config::L1 * Config::L2);
  if (c2 < -1 || c2 > 1) return false;

  float theta2 = acos(c2);
  float k1 = Config::L1 + Config::L2 * cos(theta2);
  float k2 = Config::L2 * sin(theta2);
  float theta1 = atan2(y, x) - atan2(k2, k1);

  theta1_deg = degrees(theta1);
  theta2_deg = degrees(theta2);
  return true;
}

// ======================= HTTP HANDLERS ========================= //
void handleSetAngle() {
  if (!server.hasArg("servo") || !server.hasArg("angle")) {
    server.send(400, "text/plain", "Missing parameters");
    return;
  }

  int servo = server.arg("servo").toInt();
  float angle = server.arg("angle").toFloat();
  safeSetServoAngle(servo, angle);
  server.send(200, "text/plain", "Angle set");
}

void handleMoveIK() {
  if (!server.hasArg("x") || !server.hasArg("y")) {
    server.send(400, "text/plain", "Missing x/y");
    return;
  }

  float x = server.arg("x").toFloat();
  float y = server.arg("y").toFloat();
  float t1, t2;
  if (!computeIK(x, y, t1, t2)) {
    server.send(400, "text/plain", "Position unreachable");
    return;
  }

  moveArm(t1, t2, lastAngles[Config::SERVO3_CHANNEL]);
  server.send(200, "text/plain", "Moved to position");
}

void handleSweepMotion() {
  float x = server.arg("x").toFloat();
  float y_start = server.arg("y_start").toFloat();
  float y_end = server.arg("y_end").toFloat();
  float step = server.arg("step").toFloat();
  float base_init = server.arg("base_init").toFloat();
  float base_end = server.arg("base_end").toFloat();
  float base_step = server.arg("base_step").toFloat();

  for (float base = base_init; base <= base_end; base += base_step) {
    safeSetServoAngle(Config::SERVO3_CHANNEL, base);
    for (float y_curr = y_start; y_curr >= y_end; y_curr -= step) {
      float t1, t2;
      if (computeIK(x, y_curr, t1, t2)) moveArm(t1, t2, base);
      delay(50);
    }
    for (float y_curr = y_end; y_curr <= y_start; y_curr += step) {
      float t1, t2;
      if (computeIK(x, y_curr, t1, t2)) moveArm(t1, t2, base);
      delay(50);
    }
  }

  server.send(200, "text/plain", "Sweep complete");
}

// ======================= SETUP & LOOP ========================= //
void setup() {
  Serial.begin(115200);
  Wire.begin(Config::SDA_PIN, Config::SCL_PIN);
  pwm.begin();
  pwm.setPWMFreq(50);

  WiFi.softAP(Config::AP_SSID, Config::AP_PASSWORD);
  Serial.print("Connect to WiFi: ");
  Serial.println(WiFi.softAPIP());

  server.on("/set_angle", handleSetAngle);
  server.on("/move_ik", handleMoveIK);
  server.on("/sweep", handleSweepMotion);
  server.begin();

  // Start at neutral position
  moveArm(90.0, 90.0, 90.0);
}

void loop() {
  server.handleClient();
}