#include <iostream>
#include <cmath>

// Mock Serial class for non-Arduino environments
class SerialClass {
public:
    template<typename T>
    void println(T msg) {
        std::cout << msg << std::endl;
    }
};
SerialClass Serial;

// Inverse Kinematics function
bool inverseKinematics(float x_d, float y_d, float& theta1, float& theta2) {
    const float L1 = 15.0;  // cm
    const float L2 = 13.4;  // cm
    const float PI = 3.1415926535;

    // Compute distance to target
    float r = sqrt(x_d * x_d + y_d * y_d);

    // Check reachability
    if (r > (L1 + L2) || r < fabs(L1 - L2)) {
        Serial.println("Error: Position not reachable");
        theta1 = 0;
        theta2 = 0;
        return false;
    }

    // Compute theta2 (elbow angle, CCW relative to Arm 1)
    float cos_theta2 = (x_d * x_d + y_d * y_d - L1 * L1 - L2 * L2) / (2 * L1 * L2);
    if (cos_theta2 < -1 || cos_theta2 > 1) {
        Serial.println("Error: Invalid cosine value");
        theta1 = 0;
        theta2 = 0;
        return false;
    }

    // Choose elbow-up configuration (positive theta2)
    float theta2_rad = acos(cos_theta2);
    float theta2_deg = theta2_rad * 180.0 / PI;

    // Compute theta1 (shoulder angle, clockwise from +x)
    float k1 = L1 + L2 * cos(theta2_rad);
    float k2 = L2 * sin(theta2_rad);
    float theta1_rad = atan2(y_d, x_d) - atan2(k2, k1);
    float theta1_deg = theta1_rad * 180.0 / PI;

    // Normalize theta1 to [0, 360) and check range
    if (theta1_deg < 0) theta1_deg += 360;
    if (theta1_deg > 270) {
        // Try elbow-down configuration
        theta2_rad = -acos(cos_theta2);
        theta2_deg = theta2_rad * 180.0 / PI;
        k1 = L1 + L2 * cos(theta2_rad);
        k2 = L2 * sin(theta2_rad);
        theta1_rad = atan2(y_d, x_d) - atan2(k2, k1);
        theta1_deg = theta1_rad * 180.0 / PI;
        if (theta1_deg < 0) theta1_deg += 360;
        if (theta1_deg > 270 || theta2_deg < 0 || theta2_deg > 270) {
            Serial.println("Error: Angles out of range");
            theta1 = 0;
            theta2 = 0;
            return false;
        }
    } else if (theta2_deg < 0 || theta2_deg > 270) {
        // Try elbow-down configuration
        theta2_rad = -acos(cos_theta2);
        theta2_deg = theta2_rad * 180.0 / PI;
        k1 = L1 + L2 * cos(theta2_rad);
        k2 = L2 * sin(theta2_rad);
        theta1_rad = atan2(y_d, x_d) - atan2(k2, k1);
        theta1_deg = theta1_rad * 180.0 / PI;
        if (theta1_deg < 0) theta1_deg += 360;
        if (theta1_deg > 270 || theta2_deg < 0 || theta2_deg > 270) {
            Serial.println("Error: Angles out of range");
            theta1 = 0;
            theta2 = 0;
            return false;
        }
    }

    theta1 = theta1_deg;
    theta2 = theta2_deg;

    return true;
}

// Forward kinematics to verify results
void forwardKinematics(float theta1, float theta2, float& x, float& y) {
    const float L1 = 15.0;  // cm
    const float L2 = 13.4;  // cm
    const float PI = 3.1415926535;
    float theta1_rad = theta1 * PI / 180.0;
    float theta2_rad = theta2 * PI / 180.0;
    x = L1 * cos(theta1_rad) + L2 * cos(theta1_rad + theta2_rad);
    y = L1 * sin(theta1_rad) + L2 * sin(theta1_rad + theta2_rad);
}

int main() {
    // Test cases: {x, y}
    float test_cases[][2] = {
        {28.4, 0.0},  // 0 position (theta1 = 0, theta2 = 0)
        {1.6, 0.0},   // Near base (theta1 = 0, theta2 = 180)
        {15.0, 0.0},  // Mid-workspace
        {0.0, 15.0},  // Quadrant II
        {30.0, 0.0},  // Unreachable (too far)
        {-15.0, 0.0}  // Quadrant III
    };
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

    for (int i = 0; i < num_tests; i++) {
        float x_d = test_cases[i][0];
        float y_d = test_cases[i][1];
        float theta1, theta2;
        std::cout << "\nTesting position (x, y) = (" << x_d << ", " << y_d << ")\n";
        bool success = inverseKinematics(x_d, y_d, theta1, theta2);
        if (success) {
            std::cout << "Success: theta1 = " << theta1 << " deg, theta2 = " << theta2 << " deg\n";
            // Verify with forward kinematics
            float x_verify, y_verify;
            forwardKinematics(theta1, theta2, x_verify, y_verify);
            std::cout << "Verification (forward kinematics): (x, y) = (" << x_verify << ", " << y_verify << ")\n";
        } else {
            std::cout << "Failed to compute angles\n";
        }
    }

    return 0;
}