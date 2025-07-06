import time
import numpy as np
import threading
import logging
from collections import deque
import spidev
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AcousticSensor:
    def __init__(self, spi_bus=0, spi_device=0, adc_channel=1, sample_rate=1000):
        """
        Initialize acoustic sensor with piezo contact microphone
        
        Args:
            spi_bus: SPI bus number
            spi_device: SPI device number  
            adc_channel: ADC channel for piezo mic input
            sample_rate: Sampling rate in Hz
        """
        self.spi_bus = spi_bus
        self.spi_device = spi_device
        self.adc_channel = adc_channel
        self.sample_rate = sample_rate
        self.sample_interval = 1.0 / sample_rate
        
        # Initialize SPI for ADC
        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_device)
        self.spi.max_speed_hz = 1000000
        self.spi.mode = 0
        
        # Data storage
        self.buffer_size = sample_rate * 30  # 30 seconds
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        
        # Threading
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Heart sound detection
        self.heart_rate = 0
        self.s1_detected = False  # S1 heart sound (lub)
        self.s2_detected = False  # S2 heart sound (dub)
        self.last_heartbeat_time = 0
        
        # Frequency analysis
        self.heart_freq_range = (20, 200)    # Hz, typical heart sound range
        self.resp_freq_range = (0.1, 2.0)    # Hz, respiratory range
        
        # Filtering
        self.highpass_freq = 10   # Hz, remove DC and low freq noise
        self.lowpass_freq = 500   # Hz, anti-aliasing
        
        # Signal processing
        self.baseline_level = 0
        self.adaptive_threshold = 0.05
        self.noise_floor = 0
        
        # Respiratory monitoring
        self.respiratory_rate = 0
        self.last_breath_time = 0
        
        logger.info("Acoustic sensor initialized")
    
    def read_adc(self, channel):
        """Read from MCP3008 ADC with LM386 amplified piezo signal"""
        if channel < 0 or channel > 7:
            raise ValueError("Channel must be between 0 and 7")
        
        # MCP3008 command
        command = [1, (8 + channel) << 4, 0]
        result = self.spi.xfer2(command)
        
        # Convert to 10-bit value
        value = ((result[1] & 3) << 8) + result[2]
        
        # Convert to voltage (0-3.3V)
        voltage = (value / 1023.0) * 3.3
        
        # Center around 0 (assuming 1.65V bias from LM386)
        centered_voltage = voltage - 1.65
        
        return centered_voltage
    
    def apply_bandpass_filter(self, data, low_freq, high_freq):
        """Apply bandpass filter to data"""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure frequencies are in valid range
        low = max(0.01, min(0.99, low))
        high = max(0.01, min(0.99, high))
        
        if low >= high:
            return data
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, data)
            return filtered
        except:
            return data
    
    def detect_heart_sounds(self, data, threshold_factor=2.0):
        """Detect S1 and S2 heart sounds using acoustic analysis"""
        if len(data) < 100:
            return []
        
        # Apply heart sound frequency filter
        filtered = self.apply_bandpass_filter(data, 
                                            self.heart_freq_range[0], 
                                            self.heart_freq_range[1])
        
        # Calculate adaptive threshold
        rms = np.sqrt(np.mean(filtered**2))
        threshold = rms * threshold_factor
        
        # Find peaks above threshold
        peaks = []
        min_distance = int(0.2 * self.sample_rate)  # 200ms minimum between sounds
        
        for i in range(min_distance, len(filtered) - min_distance):
            if (filtered[i] > threshold and 
                filtered[i] > filtered[i-1] and 
                filtered[i] > filtered[i+1]):
                
                # Check if it's sufficiently separated from previous peaks
                too_close = False
                for prev_peak in peaks:
                    if abs(i - prev_peak) < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    peaks.append(i)
        
        return peaks
    
    def detect_respiratory_sounds(self, data):
        """Detect respiratory patterns"""
        if len(data) < self.sample_rate:  # Need at least 1 second
            return 0
        
        # Apply respiratory frequency filter
        filtered = self.apply_bandpass_filter(data, 
                                            self.resp_freq_range[0], 
                                            self.resp_freq_range[1])
        
        # Calculate power in respiratory frequency band
        fft_data = fft(filtered)
        freqs = fftfreq(len(filtered), 1/self.sample_rate)
        
        # Find power in respiratory band
        resp_mask = (freqs >= self.resp_freq_range[0]) & (freqs <= self.resp_freq_range[1])
        resp_power = np.sum(np.abs(fft_data[resp_mask])**2)
        
        # Simple respiratory rate estimation
        # This is a basic implementation - more sophisticated methods exist
        resp_peaks = self.detect_peaks_simple(filtered, min_distance=int(2 * self.sample_rate))
        
        if len(resp_peaks) >= 2:
            intervals = np.diff(resp_peaks) / self.sample_rate
            avg_interval = np.mean(intervals)
            resp_rate = 60.0 / avg_interval  # breaths per minute
            
            # Sanity check (normal range: 12-30 breaths/min)
            if 8 <= resp_rate <= 40:
                return int(resp_rate)
        
        return 0
    
    def detect_peaks_simple(self, data, min_distance=100):
        """Simple peak detection"""
        if len(data) < min_distance * 2:
            return []
        
        threshold = np.std(data) * 1.5
        peaks = []
        
        for i in range(min_distance, len(data) - min_distance):
            if (data[i] > threshold and 
                data[i] > data[i-1] and 
                data[i] > data[i+1]):
                
                # Check minimum distance
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        
        return peaks
    
    def calculate_heart_rate(self):
        """Calculate heart rate from detected heart sounds"""
        with self.lock:
            if len(self.audio_buffer) < self.sample_rate * 5:  # Need 5 seconds
                return 0
            
            # Get recent data
            recent_data = np.array(list(self.audio_buffer)[-self.sample_rate * 10:])
        
        # Detect heart sounds
        heart_peaks = self.detect_heart_sounds(recent_data)
        
        if len(heart_peaks) < 2:
            return 0
        
        # Calculate time intervals between peaks
        intervals = []
        for i in range(1, len(heart_peaks)):
            interval = (heart_peaks[i] - heart_peaks[i-1]) / self.sample_rate
            
            # Filter reasonable intervals (0.3-2.0 seconds)
            if 0.3 <= interval <= 2.0:
                intervals.append(interval)
        
        if not intervals:
            return 0
        
        # Calculate heart rate
        avg_interval = np.mean(intervals)
        heart_rate = 60.0 / avg_interval
        
        # Sanity check
        if 40 <= heart_rate <= 200:
            return int(heart_rate)
        
        return 0
    
    def analyze_acoustic_spectrum(self, duration=5):
        """Analyze frequency spectrum of recent acoustic data"""
        with self.lock:
            if len(self.audio_buffer) < duration * self.sample_rate:
                return None
            
            data = np.array(list(self.audio_buffer)[-duration * self.sample_rate:])
        
        # Perform FFT
        fft_data = fft(data)
        freqs = fftfreq(len(data), 1/self.sample_rate)
        
        # Calculate power spectrum
        power = np.abs(fft_data)**2
        
        # Analyze different frequency bands
        analysis = {
            'heart_band_power': 0,
            'resp_band_power': 0,
            'noise_power': 0,
            'dominant_freq': 0,
            'total_power': np.sum(power)
        }
        
        # Heart sound band (20-200 Hz)
        heart_mask = (freqs >= 20) & (freqs <= 200)
        analysis['heart_band_power'] = np.sum(power[heart_mask])
        
        # Respiratory band (0.1-2 Hz)  
        resp_mask = (freqs >= 0.1) & (freqs <= 2)
        analysis['resp_band_power'] = np.sum(power[resp_mask])
        
        # Noise band (above 300 Hz)
        noise_mask = freqs >= 300
        analysis['noise_power'] = np.sum(power[noise_mask])
        
        # Find dominant frequency
        if len(power) > 0:
            dominant_idx = np.argmax(power[:len(power)//2])  # Only positive frequencies
            analysis['dominant_freq'] = abs(freqs[dominant_idx])
        
        return analysis
    
    def _sampling_thread(self):
        """Background sampling thread"""
        logger.info("Acoustic sampling started")
        
        # Initialize baseline
        baseline_samples = []
        
        while self.running:
            try:
                start_time = time.time()
                
                # Read acoustic data
                acoustic_value = self.read_adc(self.adc_channel)
                
                # Store data
                with self.lock:
                    self.audio_buffer.append(acoustic_value)
                    self.timestamps.append(start_time)
                
                # Update baseline (first 100 samples)
                if len(baseline_samples) < 100:
                    baseline_samples.append(acoustic_value)
                elif len(baseline_samples) == 100:
                    self.baseline_level = np.mean(baseline_samples)
                    self.noise_floor = np.std(baseline_samples)
                    logger.info(f"Baseline established: {self.baseline_level:.3f}V, "
                              f"Noise floor: {self.noise_floor:.3f}V")
                    baseline_samples.append(0)  # Mark as done
                
                # Calculate heart rate periodically
                if len(self.audio_buffer) % (self.sample_rate // 2) == 0:  # Every 0.5 seconds
                    self.heart_rate = self.calculate_heart_rate()
                    self.respiratory_rate = self.detect_respiratory_sounds(
                        list(self.audio_buffer)[-self.sample_rate * 15:]
                    )
                
                # Maintain sample rate
                elapsed = time.time() - start_time
                sleep_time = max(0, self.sample_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Acoustic sampling error: {e}")
                time.sleep(0.01)
    
    def start(self):
        """Start acoustic monitoring"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._sampling_thread, daemon=True)
        self.thread.start()
        logger.info("Acoustic monitoring started")
    
    def stop(self):
        """Stop acoustic monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("Acoustic monitoring stopped")
    
    def get_current_data(self):
        """Get current acoustic data"""
        with self.lock:
            current_value = self.audio_buffer[-1] if self.audio_buffer else 0
            buffer_size = len(self.audio_buffer)
        
        return {
            'heart_rate': self.heart_rate,
            'respiratory_rate': self.respiratory_rate,
            'current_value': current_value,
            'baseline_level': self.baseline_level,
            'noise_floor': self.noise_floor,
            'buffer_size': buffer_size,
            'signal_strength': abs(current_value - self.baseline_level),
            'timestamp': time.time()
        }
    
    def get_raw_data(self, duration_seconds=10):
        """Get raw acoustic data for analysis"""
        samples_needed = int(duration_seconds * self.sample_rate)
        
        with self.lock:
            if len(self.audio_buffer) < samples_needed:
                return [], []
            
            data = list(self.audio_buffer)[-samples_needed:]
            times = list(self.timestamps)[-samples_needed:]
        
        return data, times
    
    def save_acoustic_data(self, filename, duration=30):
        """Save acoustic data to file for analysis"""
        data, timestamps = self.get_raw_data(duration)
        
        if not data:
            logger.error("No data to save")
            return False
        
        try:
            # Save as numpy array
            np.savez(filename, 
                    data=data, 
                    timestamps=timestamps,
                    sample_rate=self.sample_rate,
                    baseline=self.baseline_level,
                    noise_floor=self.noise_floor)
            
            logger.info(f"Acoustic data saved to {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return False
    
    def detect_anomalies(self, sensitivity=2.0):
        """Detect acoustic anomalies (irregular heartbeat, murmurs, etc.)"""
        with self.lock:
            if len(self.audio_buffer) < self.sample_rate * 10:
                return []
            
            data = np.array(list(self.audio_buffer)[-self.sample_rate * 10:])
        
        anomalies = []
        
        # Check for irregular heart rhythms
        heart_peaks = self.detect_heart_sounds(data)
        
        if len(heart_peaks) >= 3:
            intervals = np.diff(heart_peaks) / self.sample_rate
            
            # Check for irregular intervals (arrhythmia detection)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            for i, interval in enumerate(intervals):
                if abs(interval - mean_interval) > sensitivity * std_interval:
                    anomalies.append({
                        'type': 'irregular_rhythm',
                        'time_offset': heart_peaks[i] / self.sample_rate,
                        'interval': interval,
                        'expected': mean_interval,
                        'severity': abs(interval - mean_interval) / std_interval
                    })
        
        # Check for unusual frequency content (possible murmurs)
        spectrum = self.analyze_acoustic_spectrum(duration=5)
        
        if spectrum:
            # High frequency content might indicate murmurs
            if spectrum['heart_band_power'] > 0:
                high_freq_ratio = spectrum['noise_power'] / spectrum['heart_band_power']
                
                if high_freq_ratio > 0.5:  # Threshold for unusual high frequency content
                    anomalies.append({
                        'type': 'unusual_frequency_content',
                        'high_freq_ratio': high_freq_ratio,
                        'severity': min(high_freq_ratio, 2.0)
                    })
        
        # Check for very weak or very strong signals
        current_data = self.get_current_data()
        signal_strength = current_data['signal_strength']
        
        if signal_strength < self.noise_floor * 0.5:
            anomalies.append({
                'type': 'weak_signal',
                'signal_strength': signal_strength,
                'severity': 1.0
            })
        elif signal_strength > self.noise_floor * 10:
            anomalies.append({
                'type': 'strong_signal',
                'signal_strength': signal_strength,
                'severity': min(signal_strength / (self.noise_floor * 10), 2.0)
            })
        
        return anomalies
    
    def plot_acoustic_data(self, duration=10, show_analysis=True):
        """Plot acoustic data with analysis"""
        data, timestamps = self.get_raw_data(duration)
        
        if not data:
            logger.error("No data to plot")
            return
        
        # Create time axis
        time_axis = np.array(timestamps) - timestamps[0]
        data_array = np.array(data)
        
        if show_analysis:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Raw signal
            axes[0].plot(time_axis, data_array)
            axes[0].set_title("Raw Acoustic Signal")
            axes[0].set_ylabel("Amplitude (V)")
            axes[0].grid(True)
            
            # Heart sound filtered
            heart_filtered = self.apply_bandpass_filter(data_array, 20, 200)
            axes[1].plot(time_axis, heart_filtered, 'r-', label='Heart sounds')
            
            # Mark detected heart sounds
            heart_peaks = self.detect_heart_sounds(data_array)
            if heart_peaks:
                peak_times = [time_axis[p] for p in heart_peaks]
                peak_values = [heart_filtered[p] for p in heart_peaks]
                axes[1].scatter(peak_times, peak_values, color='red', s=50, zorder=5)
            
            axes[1].set_title("Heart Sound Detection")
            axes[1].set_ylabel("Amplitude (V)")
            axes[1].legend()
            axes[1].grid(True)
            
            # Frequency spectrum
            fft_data = fft(data_array)
            freqs = fftfreq(len(data_array), 1/self.sample_rate)
            power = np.abs(fft_data)**2
            
            # Only plot positive frequencies up to 300 Hz
            mask = (freqs >= 0) & (freqs <= 300)
            axes[2].semilogy(freqs[mask], power[mask])
            axes[2].set_title("Frequency Spectrum")
            axes[2].set_xlabel("Frequency (Hz)")
            axes[2].set_ylabel("Power")
            axes[2].grid(True)
            
            # Mark frequency bands
            axes[2].axvspan(20, 200, alpha=0.3, color='red', label='Heart sounds')
            axes[2].axvspan(0.1, 2, alpha=0.3, color='blue', label='Respiratory')
            axes[2].legend()
            
        else:
            plt.figure(figsize=(12, 6))
            plt.plot(time_axis, data_array)
            plt.title("Acoustic Signal")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (V)")
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_diagnostic_info(self):
        """Get comprehensive diagnostic information"""
        current_data = self.get_current_data()
        spectrum = self.analyze_acoustic_spectrum()
        anomalies = self.detect_anomalies()
        
        return {
            'current_data': current_data,
            'spectrum_analysis': spectrum,
            'detected_anomalies': anomalies,
            'system_status': {
                'sample_rate': self.sample_rate,
                'buffer_utilization': len(self.audio_buffer) / self.buffer_size,
                'baseline_established': self.baseline_level != 0,
                'is_running': self.running
            }
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        if hasattr(self, 'spi'):
            self.spi.close()

# Example usage
if __name__ == "__main__":
    acoustic = AcousticSensor(adc_channel=1, sample_rate=1000)
    
    try:
        acoustic.start()
        
        # Monitor for 30 seconds
        for i in range(30):
            time.sleep(1)
            data = acoustic.get_current_data()
            print(f"HR: {data['heart_rate']} BPM, "
                  f"RR: {data['respiratory_rate']} BPM, "
                  f"Signal: {data['signal_strength']:.4f}V, "
                  f"Buffer: {data['buffer_size']} samples")
            
            # Check for anomalies every 10 seconds
            if i % 10 == 0 and i > 0:
                anomalies = acoustic.detect_anomalies()
                if anomalies:
                    print(f"Anomalies detected: {len(anomalies)}")
                    for anomaly in anomalies:
                        print(f"  - {anomaly['type']}: severity {anomaly['severity']:.2f}")
        
        # Save data and show analysis
        acoustic.save_acoustic_data("acoustic_recording.npz", duration=20)
        
        # Show diagnostic info
        diagnostics = acoustic.get_diagnostic_info()
        print("\nDiagnostic Summary:")
        print(f"Heart Rate: {diagnostics['current_data']['heart_rate']} BPM")
        print(f"Respiratory Rate: {diagnostics['current_data']['respiratory_rate']} BPM")
        if diagnostics['spectrum_analysis']:
            print(f"Dominant Frequency: {diagnostics['spectrum_analysis']['dominant_freq']:.1f} Hz")
        print(f"Anomalies: {len(diagnostics['detected_anomalies'])}")
        
        # Uncomment to show plot (requires display)
        # acoustic.plot_acoustic_data(duration=10)
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        acoustic.cleanup()