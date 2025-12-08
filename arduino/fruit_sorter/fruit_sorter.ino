/*
 * Fruit Sorter Arduino Code
 * =========================
 * Receives commands from Python sorting_app.py via Serial
 * Controls servo motor to push rotten fruits off conveyor
 * 
 * Wiring:
 *   Servo Signal -> Pin 9
 *   Red LED      -> Pin 10 (optional)
 *   Green LED    -> Pin 11 (optional)
 */

#include <Servo.h>

Servo sorterServo;

// Pin configuration
const int SERVO_PIN = 9;
const int LED_ROTTEN = 10;  // Red LED
const int LED_GOOD = 11;    // Green LED

// Servo angles
const int REST_ANGLE = 0;    // Normal position
const int PUSH_ANGLE = 90;   // Push position

// Timing
const int PUSH_DURATION = 300;  // ms - time to hold push position

void setup() {
    // Initialize serial
    Serial.begin(9600);
    
    // Initialize servo
    sorterServo.attach(SERVO_PIN);
    sorterServo.write(REST_ANGLE);
    
    // Initialize LEDs
    pinMode(LED_ROTTEN, OUTPUT);
    pinMode(LED_GOOD, OUTPUT);
    
    // Test LEDs on startup
    digitalWrite(LED_ROTTEN, HIGH);
    digitalWrite(LED_GOOD, HIGH);
    delay(500);
    digitalWrite(LED_ROTTEN, LOW);
    digitalWrite(LED_GOOD, LOW);
    
    Serial.println("READY");
}

void loop() {
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        
        if (command == "SORT") {
            // Trigger sorting mechanism for rotten fruit
            digitalWrite(LED_ROTTEN, HIGH);
            
            // Push and return
            sorterServo.write(PUSH_ANGLE);
            delay(PUSH_DURATION);
            sorterServo.write(REST_ANGLE);
            
            digitalWrite(LED_ROTTEN, LOW);
            Serial.println("SORTED");
        }
        else if (command == "GOOD") {
            // Just blink green LED for good fruit
            digitalWrite(LED_GOOD, HIGH);
            delay(100);
            digitalWrite(LED_GOOD, LOW);
            Serial.println("PASSED");
        }
        else if (command == "TEST") {
            // Test command - blink both LEDs
            for (int i = 0; i < 3; i++) {
                digitalWrite(LED_ROTTEN, HIGH);
                digitalWrite(LED_GOOD, HIGH);
                delay(200);
                digitalWrite(LED_ROTTEN, LOW);
                digitalWrite(LED_GOOD, LOW);
                delay(200);
            }
            Serial.println("TEST_OK");
        }
        else if (command == "SERVO_TEST") {
            // Test servo movement
            sorterServo.write(PUSH_ANGLE);
            delay(500);
            sorterServo.write(REST_ANGLE);
            Serial.println("SERVO_OK");
        }
    }
}
