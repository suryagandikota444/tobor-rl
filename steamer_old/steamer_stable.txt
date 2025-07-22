#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <WiFi.h>
#include <WebServer.h>

// WiFi credentials for Access Point mode
const char* ap_ssid = "ESP32_Arm_Control"; // Name of the WiFi network the ESP32 will create
const char* ap_password = "password123";   // Password for the ESP32's WiFi network (8 chars min)

// Create web server on port 80
WebServer server(80);

// Define I2C pins
#define SDA_PIN 15
#define SCL_PIN 14

// Initialize PCA9685 with default I2C address (0x40)
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Define servo channels
const int SERVO1_CHANNEL = 1;  // Base servo (θ₁)
const int SERVO2_CHANNEL = 0;  // Elbow servo (θ₂)
const int SERVO3_CHANNEL = 2;  // Base rotation (θ0)

// Arm lengths (in cm)
const float L1 = 28.00;
const float L2 = 25.30;

// PWM pulse width limits
const int SERVO_MIN = 105; // From specs
const int SERVO_MAX = 512;

// Function to set servo angle
void setServoAngle(uint8_t channel, float angle) {
    int pulse = map((int)angle, 0, 270, SERVO_MIN, SERVO_MAX);
    pwm.setPWM(channel, 0, pulse);
}

// Compute inverse kinematics
bool inverseKinematics(float x_d, float y_d, float& theta1, float& theta2) {
    float c = ((x_d * x_d) + (y_d * y_d) - (L1 * L1) - (L2 * L2)) / (2 * L1 * L2);
    if (c < -1 || c > 1) {
        Serial.println("Error: Position not reachable");
        theta1 = 0;
        theta2 = 0;
        return false;
    }

    theta2 = acos(c);
    float a = L1 + L2 * cos(theta2);
    float b = L2 * sin(theta2);
    theta1 = atan2(a * x_d - b * y_d, b * x_d + a * y_d);

    // Convert to degrees
    theta1 = theta1 * 180.0 / PI;
    theta2 = theta2 * 180.0 / PI;
    return true;
}

// Move servos using inverse kinematics
void moveServos(float theta1, float theta2) {
    float servo1_angle = -theta1;  
    float servo2_angle = theta2;       

    setServoAngle(SERVO1_CHANNEL, servo1_angle);
    setServoAngle(SERVO2_CHANNEL, servo2_angle);
}

// Handle request to set a specific servo angle
void handleSetAngle() {
    if (server.hasArg("servo") && server.hasArg("angle")) {
        int servo = server.arg("servo").toInt();
        float angle = server.arg("angle").toFloat();
        setServoAngle(servo, angle);
        server.send(200, "text/plain", "Servo moved");
    } else {
        server.send(400, "text/plain", "Missing parameters");
    }
}

// Handle request to move to a specific (x, y) position
void handleMoveIK() {
    if (server.hasArg("x") && server.hasArg("y")) {
        float x = server.arg("x").toFloat();
        float y = server.arg("y").toFloat();
        float theta1, theta2;
        inverseKinematics(x, y, theta1, theta2);
        moveServos(theta1, theta2);
        server.send(200, "text/plain", "Moved to position");
    } else {
        server.send(400, "text/plain", "Missing parameters");
    }
}

void handleServoSweep() {
    // Check if all required parameters are present in the request
    if (server.hasArg("y_init") && server.hasArg("x_init") && server.hasArg("step_size") &&
        server.hasArg("base_angle_init") && server.hasArg("base_angle_final") &&
        server.hasArg("base_step") && server.hasArg("y_min")) {
        
        // Parse all parameters from the server request
        float y_init = server.arg("y_init").toFloat();
        float x_init = server.arg("x_init").toFloat();
        float step_size = server.arg("step_size").toFloat();
        float base_angle_init = server.arg("base_angle_init").toFloat();
        float base_angle_final = server.arg("base_angle_final").toFloat();
        float base_step = server.arg("base_step").toFloat();
        float y_min = server.arg("y_min").toFloat();

        // Execute the servo sweep with received parameters
        float last_down = 0;
        float last_up   = y_init;
        
        for (float base_angle = base_angle_init; base_angle <= base_angle_final; base_angle += base_step) {
            Serial.print("Base angle: ");
            Serial.println(base_angle);
            setServoAngle(SERVO3_CHANNEL, base_angle);
            for (float curr_y_down = last_up; curr_y_down >= y_min; curr_y_down -= step_size) {
                float theta1, theta2;
                Serial.print("Current Y down: ");
                Serial.println(curr_y_down);
                bool success = inverseKinematics(x_init, curr_y_down, theta1, theta2);
                Serial.print("Theta1 up: ");
                Serial.print(theta1);
                Serial.print(" Theta2 up: ");
                Serial.println(theta2);
                if (success) {
                  last_down = curr_y_down;
                  moveServos(theta1, theta2);
                }
                delay(100);  // Small delay between servo movements
            }

            for (float curr_y_up = last_down; curr_y_up <= y_init; curr_y_up += step_size) {
                float theta1, theta2;
                Serial.print("Current Y up: ");
                Serial.println(curr_y_up);
                bool success = inverseKinematics(x_init, curr_y_up, theta1, theta2);
                Serial.print("Theta1 down: ");
                Serial.print(theta1);
                Serial.print(" Theta2 down: ");
                Serial.println(theta2);
                if (success) {
                  last_down = curr_y_up;
                  moveServos(theta1, theta2);
                }
                delay(100);  // Small delay between servo movements
            }
        }

        // Send success response
        server.send(200, "text/plain", "Servo sweep completed");
    } else {
        // Send error response if parameters are missing
        server.send(400, "text/plain", "Missing parameters");
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(4, OUTPUT);
    digitalWrite(4, LOW);

    Wire.begin(SDA_PIN, SCL_PIN);
    pwm.begin();
    pwm.setPWMFreq(50);
    Serial.println("PCA9685 initialized");

    // Configure ESP32 as a WiFi Access Point
    Serial.print("Setting up AP: ");
    Serial.println(ap_ssid);
    WiFi.softAP(ap_ssid, NULL);

    IPAddress myIP = WiFi.softAPIP();
    Serial.print("AP IP address: ");
    Serial.println(myIP);
    Serial.println("Connect to this WiFi network and use the IP address above to send commands.");

    // Define web server routes
    server.on("/set_angle", handleSetAngle);
    server.on("/move_ik", handleMoveIK);
    server.on("/sweep", handleServoSweep);
    
    // Start server
    server.begin();
}

void loop() {
    server.handleClient();
}
