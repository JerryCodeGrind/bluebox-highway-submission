import time
import numpy as np
from threading import Thread, Lock
import spidev
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECGSensor:
    def __init__(self, spi_bus=0, spi_device=0, adc_channel=0, sample_rate=250):
        """
        Initialize ECG sensor
        
        Args:
            spi_bus: SPI bus number (usually 0)
            spi_device: SPI device number (usually 0)
            adc_channel: ADC channel (0-7 for MCP3008)
            sample_rate: Sampling rate in Hz
        """
        self.spi_bus = spi_bus
        self.spi_device = spi_device
        self.adc_channel = adc_channel
        self.sample_rate = sample_rate
        self.sample_interval = 1.0 / sample_rate
        
        # Initialize SPI
        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_device)
        self.spi.max_speed_hz = 1000000  # 1MHz
        self.spi.mode = 0
        
        # Data storage
        self.buffer_size = sample_rate * 10  # 10 seconds of data
        self.ecg_buffer = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        
        # Threading
        self.running = False
        self.thread = None
        self.lock = Lock()
        
        # Heart rate detection
        self.heart_rate = 0
        self.last_peak_time = 0
        self.peak_threshold = 2.0  # Standard deviations above mean
        self.min_peak_distance = 0.4  # Minimum time between peaks (150 BPM max)
        
    def read_adc(self, channel):
        """Read from MCP3008 ADC"""
        if channel < 0 or channel > 7:
            raise ValueError("Channel must be between 0 and 7")
        
        # MCP3008 command: start bit, single-ended, channel select
        command = [1, (8 + channel) << 4, 0]
        result = self.spi.xfer2(command)
        
        # Convert to 10-bit value
        value = ((result[1] & 3) << 8) + result[2]
        
        # Convert to voltage (assuming 3.3V reference)
        voltage = (value / 1023.0) * 3.3
        return voltage
    
    def detect_peaks(self, data, threshold_std=2.0):
        """Simple peak detection for heart rate"""
        if len(data) < 10:
            return []
        
        # Calculate threshold
        mean_val = np.mean(data)
        std_val = np.std(data)
        threshold = mean_val + threshold_std * std_val
        
        peaks = []
        for i in range(1, len(data) - 1):
            if (data[i] > threshold and 
                data[i] > data[i-1] and 
                data[i] > data[i+1]):
                peaks.append(i)
        
        return peaks
    
    def calculate_heart_rate(self):
        """Calculate heart rate from recent ECG data"""
        with self.lock:
            if len(self.ecg_buffer) < self.sample_rate * 2:  # Need at least 2 seconds
                return 0
            
            # Get recent data
            recent_data = list(self.ecg_buffer)[-self.sample_rate * 5:]  # Last 5 seconds
            recent_times = list(self.timestamps)[-self.sample_rate * 5:]
            
        # Find peaks
        peaks = self.detect_peaks(recent_data, self.peak_threshold)
        
        if len(peaks) < 2:
            return 0
        
        # Calculate intervals between peaks
        intervals = []
        for i in range(1, len(peaks)):
            time_diff = recent_times[peaks[i]] - recent_times[peaks[i-1]]
            if time_diff > self.min_peak_distance:  # Filter out noise
                intervals.append(time_diff)
        
        if not intervals:
            return 0
        
        # Calculate heart rate
        avg_interval = np.mean(intervals)
        heart_rate = 60.0 / avg_interval
        
        # Sanity check
        if 40 <= heart_rate <= 200:
            return int(heart_rate)
        else:
            return 0
    
    def _sampling_thread(self):
        """Background thread for continuous sampling"""
        logger.info("ECG sampling started")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Read ECG value
                ecg_value = self.read_adc(self.adc_channel)
                
                # Store data
                with self.lock:
                    self.ecg_buffer.append(ecg_value)
                    self.timestamps.append(start_time)
                
                # Calculate heart rate every 10 samples
                if len(self.ecg_buffer) % 10 == 0:
                    self.heart_rate = self.calculate_heart_rate()
                
                # Maintain sample rate
                elapsed = time.time() - start_time
                sleep_time = max(0, self.sample_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"ECG sampling error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start ECG monitoring"""
        if self.running:
            return
        
        self.running = True
        self.thread = Thread(target=self._sampling_thread, daemon=True)
        self.thread.start()
        logger.info("ECG monitoring started")
    
    def stop(self):
        """Stop ECG monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("ECG monitoring stopped")
    
    def get_current_data(self):
        """Get current ECG data"""
        with self.lock:
            return {
                'heart_rate': self.heart_rate,
                'latest_value': self.ecg_buffer[-1] if self.ecg_buffer else 0,
                'buffer_size': len(self.ecg_buffer),
                'timestamp': time.time()
            }
    
    def get_raw_data(self, duration_seconds=5):
        """Get raw ECG data for specified duration"""
        samples_needed = int(duration_seconds * self.sample_rate)
        
        with self.lock:
            if len(self.ecg_buffer) < samples_needed:
                return [], []
            
            data = list(self.ecg_buffer)[-samples_needed:]
            times = list(self.timestamps)[-samples_needed:]
        
        return data, times
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        if hasattr(self, 'spi'):
            self.spi.close()

# Example usage
if __name__ == "__main__":
    ecg = ECGSensor(adc_channel=0)
    
    try:
        ecg.start()
        
        # Monitor for 30 seconds
        for i in range(30):
            time.sleep(1)
            data = ecg.get_current_data()
            print(f"Heart Rate: {data['heart_rate']} BPM, "
                  f"Latest Value: {data['latest_value']:.3f}V, "
                  f"Buffer: {data['buffer_size']} samples")
        
        # Get raw data for analysis
        raw_data, timestamps = ecg.get_raw_data(duration_seconds=5)
        print(f"\nCollected {len(raw_data)} samples over 5 seconds")
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ecg.cleanup()