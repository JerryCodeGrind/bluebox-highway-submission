import sys
import os
import time
import logging
import threading
import signal
import json
from datetime import datetime
from pathlib import Path

# Import all sensor modules
try:
    from pulse_oximeter import PulseOximeter
    from temp_checker import TemperatureChecker
    from ecg_sensor import ECGSensor
    from blood_pressure_monitor import BloodPressureMonitor
    from audio_system import AudioSystem
    from camera_control import CameraControl
    from acoustic_sensor import AcousticSensor
    from display_controller import MedicalDisplayUI, SensorDataInterface
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Make sure all sensor modules are in the same directory")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_device.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedicalDeviceController:
    def __init__(self, config_file='config.json'):
        """
        Initialize the main medical device controller
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self.load_config()
        
        # System state
        self.running = False
        self.measurement_active = False
        self.sensors_initialized = False
        
        # Sensor objects
        self.sensors = {}
        self.sensor_threads = {}
        
        # Data storage
        self.data_history = {
            'measurements': [],
            'events': [],
            'errors': []
        }
        
        # Thread locks
        self.data_lock = threading.Lock()
        self.sensor_lock = threading.Lock()
        
        # UI components
        self.display_ui = None
        self.data_interface = None
        
        # Safety monitoring
        self.safety_monitor = SafetyMonitor(self)
        self.last_health_check = time.time()
        
        logger.info("Medical Device Controller initialized")
    
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            "sensors": {
                "pulse_oximeter": {
                    "enabled": True,
                    "sample_rate": 100,
                    "adc_channel": 0
                },
                "temperature": {
                    "enabled": True,
                    "sensor_address": 0x5A
                },
                "ecg": {
                    "enabled": True,
                    "sample_rate": 250,
                    "adc_channel": 0
                },
                "blood_pressure": {
                    "enabled": True,
                    "actuator_pin1": 18,
                    "actuator_pin2": 19,
                    "enable_pin": 12
                },
                "audio": {
                    "enabled": True,
                    "input_device": "ReSpeaker",
                    "output_device": "HiFiBerry"
                },
                "camera": {
                    "enabled": True,
                    "camera_index": 0,
                    "servo_channels": [0, 1]
                },
                "acoustic": {
                    "enabled": True,
                    "sample_rate": 1000,
                    "adc_channel": 1
                }
            },
            "display": {
                "fullscreen": True,
                "brightness": 80,
                "auto_dim": True
            },
            "safety": {
                "heart_rate_min": 40,
                "heart_rate_max": 120,
                "temperature_max": 39.0,
                "bp_systolic_max": 160,
                "spo2_min": 90
            },
            "data": {
                "save_interval": 300,  # 5 minutes
                "backup_interval": 3600,  # 1 hour
                "max_log_size": "100MB"
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                default_config.update(loaded_config)
            else:
                # Save default config
                self.save_config(default_config)
            
            return default_config
        
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def save_config(self, config=None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def initialize_sensors(self):
        """Initialize all enabled sensors"""
        logger.info("Initializing sensors...")
        
        with self.sensor_lock:
            sensor_configs = self.config['sensors']
            
            # Initialize Pulse Oximeter
            if sensor_configs['pulse_oximeter']['enabled']:
                try:
                    self.sensors['pulse_oximeter'] = PulseOximeter(
                        sample_rate=sensor_configs['pulse_oximeter']['sample_rate']
                    )
                    logger.info("Pulse oximeter initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize pulse oximeter: {e}")
            
            # Initialize Temperature Sensor
            if sensor_configs['temperature']['enabled']:
                try:
                    self.sensors['temperature'] = TemperatureChecker(
                        sensor_address=sensor_configs['temperature']['sensor_address']
                    )
                    logger.info("Temperature sensor initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize temperature sensor: {e}")
            
            # Initialize ECG Sensor
            if sensor_configs['ecg']['enabled']:
                try:
                    self.sensors['ecg'] = ECGSensor(
                        sample_rate=sensor_configs['ecg']['sample_rate'],
                        adc_channel=sensor_configs['ecg']['adc_channel']
                    )
                    logger.info("ECG sensor initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize ECG sensor: {e}")
            
            # Initialize Blood Pressure Monitor
            if sensor_configs['blood_pressure']['enabled']:
                try:
                    self.sensors['blood_pressure'] = BloodPressureMonitor(
                        actuator_pin1=sensor_configs['blood_pressure']['actuator_pin1'],
                        actuator_pin2=sensor_configs['blood_pressure']['actuator_pin2'],
                        enable_pin=sensor_configs['blood_pressure']['enable_pin']
                    )
                    logger.info("Blood pressure monitor initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize blood pressure monitor: {e}")
            
            # Initialize Audio System
            if sensor_configs['audio']['enabled']:
                try:
                    self.sensors['audio'] = AudioSystem(
                        input_device_name=sensor_configs['audio']['input_device'],
                        output_device_name=sensor_configs['audio']['output_device']
                    )
                    logger.info("Audio system initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize audio system: {e}")
            
            # Initialize Camera Control
            if sensor_configs['camera']['enabled']:
                try:
                    self.sensors['camera'] = CameraControl(
                        camera_index=sensor_configs['camera']['camera_index'],
                        servo_channels=sensor_configs['camera']['servo_channels']
                    )
                    logger.info("Camera control initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize camera control: {e}")
            
            # Initialize Acoustic Sensor
            if sensor_configs['acoustic']['enabled']:
                try:
                    self.sensors['acoustic'] = AcousticSensor(
                        sample_rate=sensor_configs['acoustic']['sample_rate'],
                        adc_channel=sensor_configs['acoustic']['adc_channel']
                    )
                    logger.info("Acoustic sensor initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize acoustic sensor: {e}")
        
        self.sensors_initialized = True
        logger.info(f"Initialized {len(self.sensors)} sensors")
    
    def start_sensors(self):
        """Start all sensors"""
        if not self.sensors_initialized:
            self.initialize_sensors()
        
        logger.info("Starting sensors...")
        
        with self.sensor_lock:
            for name, sensor in self.sensors.items():
                try:
                    if hasattr(sensor, 'start'):
                        sensor.start()
                        logger.info(f"Started {name}")
                except Exception as e:
                    logger.error(f"Failed to start {name}: {e}")
        
        # Start audio recording
        if 'audio' in self.sensors:
            try:
                self.sensors['audio'].start_recording()
            except Exception as e:
                logger.error(f"Failed to start audio recording: {e}")
        
        logger.info("All sensors started")
    
    def stop_sensors(self):
        """Stop all sensors"""
        logger.info("Stopping sensors...")
        
        with self.sensor_lock:
            for name, sensor in self.sensors.items():
                try:
                    if hasattr(sensor, 'stop'):
                        sensor.stop()
                    elif hasattr(sensor, 'cleanup'):
                        sensor.cleanup()
                    logger.info(f"Stopped {name}")
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")
        
        logger.info("All sensors stopped")
    
    def initialize_display(self):
        """Initialize the display UI"""
        self.display_ui = MedicalDisplayUI(
            fullscreen=self.config['display']['fullscreen']
        )
        
        # Create data interface
        self.data_interface = SensorDataInterface(self.display_ui)
            
    def initialize_display(self):
        """Initialize the display UI"""
        try:
            self.display_ui = MedicalDisplayUI(
                fullscreen=self.config['display']['fullscreen']
            )
            
            # Create data interface
            self.data_interface = SensorDataInterface(self.display_ui)
            
            # Register sensors with data interface
            for sensor_name, sensor in self.sensors.items():
                self.data_interface.register_sensor(sensor_name, sensor)
            
            logger.info("Display UI initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize display: {e}")
            return False
    
    def start_measurement_cycle(self):
        """Start a complete measurement cycle"""
        if self.measurement_active:
            logger.warning("Measurement already in progress")
            return False
        
        self.measurement_active = True
        logger.info("Starting measurement cycle")
        
        # Start measurement thread
        measurement_thread = threading.Thread(
            target=self._measurement_cycle,
            daemon=True
        )
        measurement_thread.start()
        
        return True
    
    def _measurement_cycle(self):
        """Execute a complete measurement cycle"""
        try:
            # Audio announcement
            if 'audio' in self.sensors:
                self.sensors['audio'].speak("Starting measurement cycle")
            
            measurements = {}
            timestamp = datetime.now()
            
            # Collect data from each sensor
            measurement_sequence = [
                ('pulse_oximeter', 'Heart rate and SpO2'),
                ('temperature', 'Body temperature'),
                ('blood_pressure', 'Blood pressure'),
                ('ecg', 'ECG reading'),
                ('acoustic', 'Heart sounds')
            ]
            
            for sensor_name, description in measurement_sequence:
                if not self.measurement_active:
                    break
                
                if sensor_name in self.sensors:
                    logger.info(f"Measuring {description}")
                    
                    if 'audio' in self.sensors:
                        self.sensors['audio'].speak(f"Measuring {description}")
                    
                    # Get measurement based on sensor type
                    try:
                        if sensor_name == 'pulse_oximeter':
                            data = self.sensors[sensor_name].get_current_data()
                            measurements['heart_rate'] = data.get('heart_rate', 0)
                            measurements['spo2'] = data.get('spo2', 0)
                        
                        elif sensor_name == 'temperature':
                            data = self.sensors[sensor_name].get_current_data()
                            measurements['temperature'] = data.get('temperature', 0.0)
                        
                        elif sensor_name == 'blood_pressure':
                            # Take blood pressure measurement
                            bp_data = self.sensors[sensor_name].get_measurement_blocking(timeout=60)
                            if bp_data:
                                measurements['blood_pressure_sys'] = bp_data.get('systolic', 0)
                                measurements['blood_pressure_dia'] = bp_data.get('diastolic', 0)
                                measurements['pulse'] = bp_data.get('pulse', 0)
                        
                        elif sensor_name == 'ecg':
                            data = self.sensors[sensor_name].get_current_data()
                            measurements['ecg_heart_rate'] = data.get('heart_rate', 0)
                        
                        elif sensor_name == 'acoustic':
                            data = self.sensors[sensor_name].get_current_data()
                            measurements['acoustic_heart_rate'] = data.get('heart_rate', 0)
                            measurements['respiratory_rate'] = data.get('respiratory_rate', 0)
                        
                        time.sleep(2)  # Brief pause between measurements
                        
                    except Exception as e:
                        logger.error(f"Error measuring {sensor_name}: {e}")
                        measurements[f'{sensor_name}_error'] = str(e)
            
            # Store measurement results
            measurement_record = {
                'timestamp': timestamp.isoformat(),
                'measurements': measurements,
                'patient_id': 'default',  # Could be configured
                'device_id': 'medical_monitor_001'
            }
            
            with self.data_lock:
                self.data_history['measurements'].append(measurement_record)
            
            # Save to file
            self.save_measurement_data(measurement_record)
            
            # Check for safety alerts
            self.safety_monitor.check_measurements(measurements)
            
            # Audio completion
            if 'audio' in self.sensors:
                self.sensors['audio'].speak("Measurement cycle complete")
            
            logger.info("Measurement cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Measurement cycle error: {e}")
            
        finally:
            self.measurement_active = False
    
    def save_measurement_data(self, measurement_record):
        """Save measurement data to file"""
        try:
            # Create data directory if it doesn't exist
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Save individual measurement
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = data_dir / f"measurement_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(measurement_record, f, indent=4)
            
            # Append to daily log
            daily_log = data_dir / f"daily_log_{datetime.now().strftime('%Y%m%d')}.json"
            
            if daily_log.exists():
                with open(daily_log, 'r') as f:
                    daily_data = json.load(f)
            else:
                daily_data = {"measurements": []}
            
            daily_data["measurements"].append(measurement_record)
            
            with open(daily_log, 'w') as f:
                json.dump(daily_data, f, indent=4)
            
            logger.info(f"Measurement data saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save measurement data: {e}")
    
    def get_system_status(self):
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'system_running': self.running,
            'measurement_active': self.measurement_active,
            'sensors_initialized': self.sensors_initialized,
            'sensors': {},
            'last_health_check': self.last_health_check,
            'memory_usage': self.get_memory_usage(),
            'disk_usage': self.get_disk_usage()
        }
        
        # Get status from each sensor
        with self.sensor_lock:
            for name, sensor in self.sensors.items():
                try:
                    if hasattr(sensor, 'get_status'):
                        status['sensors'][name] = sensor.get_status()
                    elif hasattr(sensor, 'get_current_data'):
                        status['sensors'][name] = sensor.get_current_data()
                    else:
                        status['sensors'][name] = {'status': 'active'}
                except Exception as e:
                    status['sensors'][name] = {'status': 'error', 'error': str(e)}
        
        return status
    
    def get_memory_usage(self):
        """Get system memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            }
        except ImportError:
            return {'status': 'psutil not available'}
    
    def get_disk_usage(self):
        """Get disk usage"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            }
        except ImportError:
            return {'status': 'psutil not available'}
    
    def health_check(self):
        """Perform system health check"""
        try:
            current_time = time.time()
            
            # Check sensor health
            unhealthy_sensors = []
            with self.sensor_lock:
                for name, sensor in self.sensors.items():
                    try:
                        if hasattr(sensor, 'get_current_data'):
                            data = sensor.get_current_data()
                            # Check if data is recent (within last 30 seconds)
                            if 'timestamp' in data:
                                if current_time - data['timestamp'] > 30:
                                    unhealthy_sensors.append(name)
                    except Exception as e:
                        unhealthy_sensors.append(name)
                        logger.warning(f"Health check failed for {name}: {e}")
            
            # Log health status
            if unhealthy_sensors:
                logger.warning(f"Unhealthy sensors detected: {unhealthy_sensors}")
            else:
                logger.debug("All sensors healthy")
            
            self.last_health_check = current_time
            
            return {
                'healthy': len(unhealthy_sensors) == 0,
                'unhealthy_sensors': unhealthy_sensors,
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {'healthy': False, 'error': str(e)}
    
    def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logger.warning("Emergency shutdown initiated")
        
        try:
            # Stop all measurements
            self.measurement_active = False
            
            # Audio alert
            if 'audio' in self.sensors:
                self.sensors['audio'].speak("Emergency shutdown initiated")
            
            # Save current state
            emergency_data = {
                'timestamp': datetime.now().isoformat(),
                'event': 'emergency_shutdown',
                'system_status': self.get_system_status()
            }
            
            with open('emergency_log.json', 'w') as f:
                json.dump(emergency_data, f, indent=4)
            
            # Stop sensors safely
            self.stop_sensors()
            
            logger.info("Emergency shutdown completed")
            
        except Exception as e:
            logger.error(f"Emergency shutdown error: {e}")
    
    def start_system(self):
        """Start the complete medical device system"""
        logger.info("Starting medical device system...")
        
        try:
            self.running = True
            
            # Initialize and start sensors
            self.initialize_sensors()
            self.start_sensors()
            
            # Start safety monitoring
            self.safety_monitor.start()
            
            # Initialize display
            if self.initialize_display():
                # Start data collection for display
                self.data_interface.start_data_collection()
            
            # Start health monitoring
            self.start_health_monitoring()
            
            logger.info("Medical device system started successfully")
            
            # Audio confirmation
            if 'audio' in self.sensors:
                self.sensors['audio'].speak("Medical device system ready")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.emergency_shutdown()
            return False
    
    def start_health_monitoring(self):
        """Start background health monitoring"""
        def health_monitor_loop():
            while self.running:
                try:
                    self.health_check()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(10)
        
        health_thread = threading.Thread(target=health_monitor_loop, daemon=True)
        health_thread.start()
        logger.info("Health monitoring started")
    
    def stop_system(self):
        """Stop the medical device system"""
        logger.info("Stopping medical device system...")
        
        self.running = False
        self.measurement_active = False
        
        try:
            # Stop safety monitoring
            self.safety_monitor.stop()
            
            # Stop sensors
            self.stop_sensors()
            
            # Cleanup display
            if self.display_ui:
                self.display_ui.cleanup()
            
            # Save final state
            final_status = self.get_system_status()
            with open('final_status.json', 'w') as f:
                json.dump(final_status, f, indent=4)
            
            logger.info("Medical device system stopped")
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
    
    def run_interactive_mode(self):
        """Run in interactive mode with voice commands"""
        if 'audio' not in self.sensors:
            logger.error("Audio system not available for interactive mode")
            return
        
        logger.info("Starting interactive mode")
        audio = self.sensors['audio']
        
        try:
            while self.running:
                audio.speak("What would you like me to do?")
                
                command = audio.listen_for_speech(timeout=10)
                
                if command:
                    command_lower = command.lower()
                    
                    if any(word in command_lower for word in ['measure', 'start', 'test']):
                        audio.speak("Starting measurement cycle")
                        self.start_measurement_cycle()
                        
                    elif any(word in command_lower for word in ['status', 'health', 'check']):
                        status = self.get_system_status()
                        healthy_sensors = len([s for s in status['sensors'].values() 
                                             if s.get('status') != 'error'])
                        audio.speak(f"System status: {healthy_sensors} sensors healthy")
                        
                    elif any(word in command_lower for word in ['stop', 'exit', 'quit']):
                        audio.speak("Stopping system")
                        break
                        
                    elif 'emergency' in command_lower:
                        audio.speak("Emergency mode activated")
                        self.emergency_shutdown()
                        break
                        
                    else:
                        audio.speak("Command not recognized. Try: measure, status, or stop")
                else:
                    audio.speak("No command heard")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Interactive mode interrupted")
        except Exception as e:
            logger.error(f"Interactive mode error: {e}")

# Safety monitoring class
class SafetyMonitor:
    def __init__(self, controller):
        self.controller = controller
        self.running = False
        self.alerts = []
        self.alert_lock = threading.Lock()
        
    def start(self):
        """Start safety monitoring"""
        self.running = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("Safety monitoring started")
    
    def stop(self):
        """Stop safety monitoring"""
        self.running = False
        logger.info("Safety monitoring stopped")
    
    def _monitor_loop(self):
        """Main safety monitoring loop"""
        while self.running:
            try:
                # Get current measurements from all sensors
                measurements = {}
                
                with self.controller.sensor_lock:
                    for name, sensor in self.controller.sensors.items():
                        try:
                            if hasattr(sensor, 'get_current_data'):
                                data = sensor.get_current_data()
                                measurements.update(data)
                        except:
                            pass
                
                # Check for safety violations
                self.check_measurements(measurements)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                time.sleep(10)
    
    def check_measurements(self, measurements):
        """Check measurements against safety thresholds"""
        safety_config = self.controller.config['safety']
        alerts = []
        
        # Heart rate checks
        hr = measurements.get('heart_rate', 0)
        if hr > 0:
            if hr < safety_config['heart_rate_min']:
                alerts.append(f"Low heart rate: {hr} BPM")
            elif hr > safety_config['heart_rate_max']:
                alerts.append(f"High heart rate: {hr} BPM")
        
        # Temperature checks
        temp = measurements.get('temperature', 0)
        if temp > safety_config['temperature_max']:
            alerts.append(f"High temperature: {temp:.1f}°C")
        
        # SpO2 checks
        spo2 = measurements.get('spo2', 0)
        if spo2 > 0 and spo2 < safety_config['spo2_min']:
            alerts.append(f"Low SpO2: {spo2}%")
        
        # Blood pressure checks
        bp_sys = measurements.get('blood_pressure_sys', 0)
        if bp_sys > safety_config['bp_systolic_max']:
            alerts.append(f"High blood pressure: {bp_sys} mmHg systolic")
        
        # Handle alerts
        if alerts:
            self.handle_alerts(alerts)
    
    def handle_alerts(self, alerts):
        """Handle safety alerts"""
        with self.alert_lock:
            for alert in alerts:
                if alert not in [a['message'] for a in self.alerts[-10:]]:  # Avoid spam
                    alert_record = {
                        'timestamp': datetime.now().isoformat(),
                        'message': alert,
                        'level': 'warning'
                    }
                    
                    self.alerts.append(alert_record)
                    logger.warning(f"Safety alert: {alert}")
                    
                    # Audio alert
                    if 'audio' in self.controller.sensors:
                        self.controller.sensors['audio'].speak(f"Alert: {alert}")

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"Received signal {signum}, shutting down...")
    if 'controller' in globals():
        controller.stop_system()
    sys.exit(0)

# Main execution
if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start the medical device controller
    controller = MedicalDeviceController()
    
    try:
        # Start the system
        if controller.start_system():
            
            # Check command line arguments
            if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
                # Run in interactive voice mode
                controller.run_interactive_mode()
            else:
                # Run with GUI
                if controller.display_ui:
                    controller.display_ui.run()
                else:
                    # Fallback to command line mode
                    logger.info("GUI not available, running in background mode")
                    logger.info("Press Ctrl+C to stop")
                    
                    while controller.running:
                        time.sleep(1)
        
        else:
            logger.error("Failed to start system")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        controller.stop_system()
        logger.info("Medical device controller terminated")