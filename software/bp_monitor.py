import time
import serial
import logging
from threading import Thread, Lock
import RPi.GPIO as GPIO
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BloodPressureMonitor:
    def __init__(self, actuator_pin1=18, actuator_pin2=19, enable_pin=12, 
                 uart_port='/dev/ttyUSB0', baudrate=9600):
        """
        Initialize blood pressure monitor with linear actuator control
        
        Args:
            actuator_pin1: GPIO pin for actuator direction control
            actuator_pin2: GPIO pin for actuator direction control  
            enable_pin: GPIO pin for actuator enable (PWM)
            uart_port: Serial port for BP monitor communication
            baudrate: Serial communication speed
        """
        self.actuator_pin1 = actuator_pin1
        self.actuator_pin2 = actuator_pin2
        self.enable_pin = enable_pin
        self.uart_port = uart_port
        self.baudrate = baudrate
        
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.actuator_pin1, GPIO.OUT)
        GPIO.setup(self.actuator_pin2, GPIO.OUT)
        GPIO.setup(self.enable_pin, GPIO.OUT)
        
        # Initialize PWM for speed control
        self.pwm = GPIO.PWM(self.enable_pin, 1000)  # 1kHz frequency
        self.pwm.start(0)  # Start with 0% duty cycle
        
        # Serial connection
        self.serial = None
        self.serial_lock = Lock()
        
        # State variables
        self.is_measuring = False
        self.last_measurement = None
        self.measurement_lock = Lock()
        
        # Actuator state
        self.actuator_position = 0  # 0=retracted, 1=extended
        self.actuator_speed = 70  # PWM duty cycle (0-100)
        
        self.setup_serial()
    
    def setup_serial(self):
        """Setup serial connection to BP monitor"""
        try:
            self.serial = serial.Serial(
                self.uart_port, 
                self.baudrate,
                timeout=1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            logger.info(f"Serial connection established on {self.uart_port}")
        except Exception as e:
            logger.error(f"Failed to setup serial connection: {e}")
            self.serial = None
    
    def actuator_extend(self):
        """Extend the linear actuator"""
        if self.actuator_position == 1:
            return
        
        logger.info("Extending actuator...")
        GPIO.output(self.actuator_pin1, GPIO.HIGH)
        GPIO.output(self.actuator_pin2, GPIO.LOW)
        self.pwm.ChangeDutyCycle(self.actuator_speed)
        
        # Run for calculated time (depends on actuator stroke and speed)
        time.sleep(2.0)  # Adjust based on your actuator specifications
        
        self.actuator_stop()
        self.actuator_position = 1
        logger.info("Actuator extended")
    
    def actuator_retract(self):
        """Retract the linear actuator"""
        if self.actuator_position == 0:
            return
        
        logger.info("Retracting actuator...")
        GPIO.output(self.actuator_pin1, GPIO.LOW)
        GPIO.output(self.actuator_pin2, GPIO.HIGH)
        self.pwm.ChangeDutyCycle(self.actuator_speed)
        
        # Run for calculated time
        time.sleep(2.0)  # Adjust based on your actuator specifications
        
        self.actuator_stop()
        self.actuator_position = 0
        logger.info("Actuator retracted")
    
    def actuator_stop(self):
        """Stop the linear actuator"""
        self.pwm.ChangeDutyCycle(0)
        GPIO.output(self.actuator_pin1, GPIO.LOW)
        GPIO.output(self.actuator_pin2, GPIO.LOW)
    
    def send_bp_command(self, command):
        """Send command to BP monitor"""
        if not self.serial:
            return None
        
        with self.serial_lock:
            try:
                self.serial.write(command.encode() + b'\r\n')
                time.sleep(0.1)
                response = self.serial.readline().decode().strip()
                return response
            except Exception as e:
                logger.error(f"Serial communication error: {e}")
                return None
    
    def parse_bp_data(self, data):
        """Parse blood pressure data from monitor"""
        try:
            # Example parsing for common Omron format
            # Actual format depends on your specific monitor model
            if 'SYS' in data and 'DIA' in data and 'PUL' in data:
                parts = data.split(',')
                systolic = int(parts[0].split(':')[1])
                diastolic = int(parts[1].split(':')[1])
                pulse = int(parts[2].split(':')[1])
                
                return {
                    'systolic': systolic,
                    'diastolic': diastolic,
                    'pulse': pulse,
                    'timestamp': datetime.now().isoformat(),
                    'quality': 'good'  # Could be determined from monitor flags
                }
        except:
            pass
        
        return None
    
    def start_measurement(self):
        """Start blood pressure measurement"""
        if self.is_measuring:
            logger.warning("Measurement already in progress")
            return False
        
        with self.measurement_lock:
            self.is_measuring = True
        
        # Position actuator for measurement
        self.actuator_extend()
        
        # Wait for stabilization
        time.sleep(2)
        
        # Start measurement on BP monitor
        logger.info("Starting blood pressure measurement...")
        response = self.send_bp_command("START")
        
        if response and "OK" in response:
            logger.info("BP measurement started successfully")
            return True
        else:
            logger.error("Failed to start BP measurement")
            with self.measurement_lock:
                self.is_measuring = False
            self.actuator_retract()
            return False
    
    def check_measurement_status(self):
        """Check if measurement is complete"""
        if not self.is_measuring:
            return None
        
        response = self.send_bp_command("STATUS")
        
        if response and "COMPLETE" in response:
            # Get measurement data
            data_response = self.send_bp_command("DATA")
            
            if data_response:
                measurement = self.parse_bp_data(data_response)
                
                if measurement:
                    with self.measurement_lock:
                        self.last_measurement = measurement
                        self.is_measuring = False
                    
                    # Retract actuator
                    self.actuator_retract()
                    
                    logger.info(f"Measurement complete: {measurement}")
                    return measurement
        
        return None
    
    def get_measurement_blocking(self, timeout=120):
        """Get measurement with blocking wait"""
        if not self.start_measurement():
            return None
        
        start_time = time.time()
        
        while self.is_measuring and (time.time() - start_time) < timeout:
            result = self.check_measurement_status()
            if result:
                return result
            time.sleep(1)
        
        # Timeout or error
        if self.is_measuring:
            logger.error("Measurement timeout")
            with self.measurement_lock:
                self.is_measuring = False
            self.actuator_retract()
        
        return None
    
    def get_last_measurement(self):
        """Get the last measurement"""
        with self.measurement_lock:
            return self.last_measurement
    
    def is_measurement_in_progress(self):
        """Check if measurement is in progress"""
        with self.measurement_lock:
            return self.is_measuring
    
    def calibrate_actuator(self):
        """Calibrate actuator positions"""
        logger.info("Calibrating actuator...")
        
        # Retract fully
        self.actuator_retract()
        time.sleep(1)
        
        # Extend fully
        self.actuator_extend()
        time.sleep(1)
        
        # Return to neutral
        self.actuator_retract()
        
        logger.info("Actuator calibration complete")
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up blood pressure monitor...")
        
        with self.measurement_lock:
            self.is_measuring = False
        
        self.actuator_stop()
        self.actuator_retract()
        
        if self.serial:
            self.serial.close()
        
        self.pwm.stop()
        GPIO.cleanup([self.actuator_pin1, self.actuator_pin2, self.enable_pin])

# Example usage
if __name__ == "__main__":
    bp_monitor = BloodPressureMonitor()
    
    try:
        # Calibrate actuator first
        bp_monitor.calibrate_actuator()
        
        # Take a measurement
        print("Starting blood pressure measurement...")
        measurement = bp_monitor.get_measurement_blocking(timeout=60)
        
        if measurement:
            print(f"Blood Pressure: {measurement['systolic']}/{measurement['diastolic']} mmHg")
            print(f"Pulse: {measurement['pulse']} BPM")
            print(f"Time: {measurement['timestamp']}")
        else:
            print("Measurement failed or timed out")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        bp_monitor.cleanup()