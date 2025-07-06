# Health Monitoring System

## Project Description
My project aims to fully detect a patient's vital signs through a hand placement device, a camera, a stethoscope, and a temperature sensor. We're trying to streamline the patient diagnosis process by making vital sign detection easier. The vitals it detects are: SpO2, Pulse, Body Temperature, Respiratory Rate, Blood Pressure, ECG. The camera is also involved in the diagnosis process in which computer vision helps in providing additional information to the main diagnoser model.

## Why We Made This Project
We live in Ontario, Canada where there is a severe shortage of family doctors. Over 2.5 million Canadians living in Ontario lack family doctors and we want to fix this. We are already very advanced in the software side of the project but believe that hardware used to track vital signs are becoming increasingly important so we decided to build a project which can detect all of a patient's basic vital signs without doctor intervention. 

### 3D Model
All stl components in 3dprints folder

### Software
The software contains all the code for controller the raspberry pi as well as calibrating and outputting results with the sensors. The GPU AI doctor backend is hosted on the cloud and then accessed via an endpoint. All code is in the software folder.

### Wiring Diagram
wiring/circutwiring.png

## Bill of Materials

| Category | Part Name | Quantity | Description | Link |
|----------|-----------|----------|-------------|------|
| Heart Rate & SpO2 | MAX30105 Sensor Module | 1 | Heart rate and SpO2 sensor with particle photodetectors | [Link](https://www.amazon.com/Accuracy-MAX30105-Particle-Photodetectors-Detection/dp/B0F6MLWF8K) |
| ECG | ECG Sensor Module | 1 | Electrocardiogram sensor for heart monitoring | [Link](https://www.amazon.com/gp/product/B0FB54XWFG) |
| Temperature | MLX90614 Temperature Sensor | 1 | Non-contact infrared temperature sensor module | [Link](https://www.amazon.com/SHILLEHTEK-Pre-Soldered-Non-Touch-Temperature-Microcontrollers/dp/B0CNM72R4L) |
| Display | Raspberry Pi 7 inch Touchscreen | 1 | 7-inch LCD touchscreen display for Raspberry Pi | [Link](https://www.canakit.com/raspberry-pi-lcd-display-touchscreen.html) |
| Audio Input | ReSpeaker Mic Array v2.0 | 1 | Microphone array with 4 microphones for voice input | [Link](https://www.seeedstudio.com/ReSpeaker-Mic-Array-v2-0.html) |
| Audio Output | HiFiBerry MiniAmp | 1 | Audio amplifier expansion board for Raspberry Pi | [Link](https://www.amazon.com/Inno-Maker-Raspberry-Amplifier-Expansion-Capacitor/dp/B07CZZ95B9) |
| Audio Output | Sound Town 4-inch Speakers | 1 | Full-range driver speaker pair | [Link](https://www.amazon.com/Sound-Town-Replacement-Speakers-STLF-EZ4-PAIR/dp/B09GXBF9QB) |
| Blood Pressure | Omron Blood Pressure Monitor | 1 | Clinically validated blood pressure monitor | [Link](https://www.amazon.com/Pressure-Clinically-Validated-Unlimited-Measurements/dp/B0DD46HGC9) |
| Actuator | 12V Linear Actuator | 1 | 50mm stroke linear actuator with built-in limit switches | [Link](https://www.amazon.com/Actuator-Internal-Automotive-Industrial-50mm-8mm/dp/B09FT562LB) |
| Motor Control | L298N Motor Driver Module | 1 | H-bridge motor driver for controlling the linear actuator | [Link](https://www.amazon.com/BOJACK-H-Bridge-Controller-Intelligent-Mega2560/dp/B0C5JCF5RS) |
| Camera | Logitech C920 Webcam | 1 | HD webcam with microphone for video capture | [Link](https://www.amazon.com/Logitech-Mic-Disabled-Certified-Microsoft-Compliant/dp/B08CS18WVP) |
| Servo | MG996R Servo | 1 | High torque digital servo for pan-tilt mechanism | [Link](https://www.amazon.com/4-Pack-MG996R-Torque-Digital-Helicopter/dp/B07MFK266B) |
| Servo | SG90 Micro Servo | 1 | Micro servo for pan-tilt mechanism | [Link](https://www.amazon.com/Micro-Helicopter-Airplane-Remote-Control/dp/B072V529YD) |
| Mounting | Aluminum Pan-Tilt Bracket Kit | 1 | Metal bracket kit for pan-tilt camera mounting | [Link](https://www.digikey.com/en/products/detail/dfrobot/FIT0004/7597177) |
| Mounting | Camera Mounting Plate | 1 | Anti-twist camera mounting plate | [Link](https://www.amazon.com/CAMVATE-Camera-Mount-Plate-Anti-Twist/dp/B0DB8BBFX7) |
| Servo Control | PCA9685 Servo Driver Board | 1 | 16-channel PWM servo driver board | [Link](https://www.amazon.com/SunFounder-PCA9685-Channel-Arduino-Raspberry/dp/B014KTSMLA) |
| Acoustic Sensor | 35mm Piezo Contact Mic | 1 | Piezoelectric contact microphone for acoustic sensing | [Link](https://www.amazon.com/DZS-Elec-Transducer-Microphone-Instrument/dp/B07TF5Q74Z) |
| Audio Amplifier | LM386 Breakout Board | 1 | Audio amplifier module for piezo microphone | [Link](https://www.amazon.com/HiLetgo-LM386-Audio-Amplifier-Module/dp/B00LNACGTY) |
| ADC | MCP3008 ADC Module | 1 | 8-channel 10-bit analog-to-digital converter | [Link](https://www.amazon.com/raspberryads1115-Breakout-soldered-Development-Converter/dp/B0C4QBYWYH) | 