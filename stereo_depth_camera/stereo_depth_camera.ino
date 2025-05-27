// minimal_cam_server.ino

#include "esp_camera.h"
#include <WiFi.h>
#include "credentials.h"

WiFiServer server(80);

void startCamera();

void setup() {
  Serial.begin(115200);

  // Camera config for AI Thinker
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // Init camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }
  sensor_t *s = esp_camera_sensor_get();
  s->set_gain_ctrl(s, 1);       // enable auto gain
  s->set_exposure_ctrl(s, 1);   // enable auto exposure
  s->set_brightness(s, 1);      // +1 brightness (can be -2 to +2)
  s->set_contrast(s, 1);        // optional
  s->set_saturation(s, 1);      // optional
  s->set_framesize(s, FRAMESIZE_QVGA);  // try SVGA if fast enough

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  WiFiClient client = server.available();
  if (!client) return;
  while (!client.available()) delay(1);

  String req = client.readStringUntil('\r');
  client.readStringUntil('\n'); // flush newline

  if (req.indexOf("GET /cam.jpg") != -1) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      client.println("HTTP/1.1 500 Internal Server Error\r\n\r\nCapture failed");
      return;
    }
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: image/jpeg");
    client.println("Content-Length: " + String(fb->len));
    client.println("Connection: close\r\n");
    client.write(fb->buf, fb->len);
    esp_camera_fb_return(fb);
  } else {
    client.println("HTTP/1.1 404 Not Found\r\n\r\nNot found");
  }
  delay(1);
  client.stop();
}
