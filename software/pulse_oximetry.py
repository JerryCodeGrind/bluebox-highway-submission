import time
import qwiic_max3010x
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import deque

# Initialize sensor
sensor = qwiic_max3010x.QwiicMax3010x()
if not sensor.connected:
    print("MAX3010x not connected. Check wiring.")
    exit()

sensor.setup()
print("Place finger on sensor. Press Ctrl+C to stop.\n")

# Configuration
BUFFER_SIZE = 250
red_values = deque(maxlen=BUFFER_SIZE)
ir_values = deque(maxlen=BUFFER_SIZE)
timestamps = deque(maxlen=BUFFER_SIZE)

# Smoothing
hr_history = deque(maxlen=3)
spo2_history = deque(maxlen=3)

start_time = time.time()

def simple_moving_average(data, window=3):
    """Simple moving average"""
    if len(data) < window:
        return data
    averaged = []
    for i in range(window//2, len(data) - window//2):
        averaged.append(np.mean(data[i-window//2:i+window//2+1]))
    return np.array(averaged)

def detect_finger(red_vals, ir_vals):
    """Simple finger detection"""
    if len(red_vals) < 20:
        return False
    
    recent_red = red_vals[-30:]
    recent_ir = ir_vals[-30:]
    
    # Check if signal is reasonable
    red_mean = np.mean(recent_red)
    ir_mean = np.mean(recent_ir)
    
    # Very lenient thresholds
    if red_mean < 5000 or ir_mean < 5000:
        return False
    if red_mean > 1000000 or ir_mean > 1000000:
        return False
        
    # Check for any variation at all
    return np.std(recent_red) > 100 and np.std(recent_ir) > 100

def calculate_bpm(signal, times):
    """Very sensitive BPM calculation"""
    if len(signal) < 40:
        return None
    
    # Use recent data
    recent_signal = signal[-100:] if len(signal) > 100 else signal
    recent_times = times[-100:] if len(times) > 100 else times
    
    # Simple preprocessing
    # Remove trend
    detrended = recent_signal - np.mean(recent_signal)
    
    # Light smoothing
    if len(detrended) > 6:
        smoothed = simple_moving_average(detrended, 3)
        smooth_times = recent_times[1:-1]  # Adjust for smoothing
    else:
        smoothed = detrended
        smooth_times = recent_times
    
    # Very lenient peak detection
    signal_std = np.std(smoothed)
    
    if signal_std < 10:  # Signal too flat
        return None
    
    # Find peaks with minimal restrictions
    peaks, _ = find_peaks(
        smoothed,
        distance=8,  # Very small distance (about 0.3 seconds)
        prominence=signal_std * 0.05,  # Very low prominence
        height=0  # Any peak above zero
    )
    
    print(f"Debug: Found {len(peaks)} peaks, std={signal_std:.1f}")
    
    if len(peaks) < 2:
        return None
    
    # Calculate intervals
    peak_times = np.array(smooth_times)[peaks]
    intervals = np.diff(peak_times)
    
    print(f"Debug: Intervals: {intervals}")
    
    # Accept wide range of heart rates
    valid_intervals = []
    for interval in intervals:
        if 0.3 < interval < 3.0:  # 20-200 BPM range
            valid_intervals.append(interval)
    
    if not valid_intervals:
        return None
    
    # Use median
    avg_interval = np.median(valid_intervals)
    bpm = 60 / avg_interval
    
    print(f"Debug: Calculated BPM = {bpm:.1f}")
    
    # Final validation - very wide range
    if 30 <= bpm <= 200:
        return bpm
    
    return None

def calculate_spo2_simple(red_vals, ir_vals):
    """Simple SpO2 calculation"""
    if len(red_vals) < 50:
        return None
    
    # Use recent data
    recent_red = red_vals[-80:]
    recent_ir = ir_vals[-80:]
    
    # Basic AC/DC calculation
    red_ac = np.std(recent_red)
    red_dc = np.mean(recent_red)
    ir_ac = np.std(recent_ir)
    ir_dc = np.mean(recent_ir)
    
    if red_dc < 100 or ir_dc < 100 or red_ac < 10 or ir_ac < 10:
        return None
    
    # R calculation
    R = (red_ac / red_dc) / (ir_ac / ir_dc)
    
    # Simple SpO2 formula
    if 0.3 <= R <= 4.0:
        spo2 = 100 - 5 * R  # Very simple linear relationship
        return max(85, min(100, spo2))
    
    return None

def smooth_reading(history, new_val):
    """Smooth readings"""
    if new_val is not None:
        history.append(new_val)
    
    if len(history) >= 2:
        return np.mean(list(history))
    elif len(history) == 1:
        return history[0]
    
    return None

# Setup plot
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

try:
    while True:
        # Read sensor
        red = sensor.getRed()
        ir = sensor.getIR()
        now = time.time() - start_time
        
        red_values.append(red)
        ir_values.append(ir)
        timestamps.append(now)
        
        if len(red_values) < 20:
            continue
        
        # Convert to arrays
        red_arr = np.array(red_values)
        ir_arr = np.array(ir_values)
        time_arr = np.array(timestamps)
        
        # Check finger
        finger_ok = detect_finger(red_arr, ir_arr)
        
        if finger_ok:
            # Calculate vitals
            raw_bpm = calculate_bpm(red_arr, time_arr)
            raw_spo2 = calculate_spo2_simple(red_arr, ir_arr)
            
            # Smooth results
            final_bpm = smooth_reading(hr_history, raw_bpm)
            final_spo2 = smooth_reading(spo2_history, raw_spo2)
            
            # Display
            if final_bpm:
                bpm_text = f"Heart Rate: {final_bpm:.0f} BPM"
                bpm_color = 'darkgreen'
            else:
                bpm_text = "Heart Rate: Detecting..."
                bpm_color = 'orange'
                
            if final_spo2:
                spo2_text = f"SpO₂: {final_spo2:.0f}%"
                spo2_color = 'darkgreen'
            else:
                spo2_text = "SpO₂: Calculating..."
                spo2_color = 'orange'
                
            status_text = "Finger detected - Good signal"
            status_color = 'green'
            
        else:
            bpm_text = "Heart Rate: No finger"
            spo2_text = "SpO₂: No finger"
            status_text = "Place finger on sensor"
            bpm_color = spo2_color = status_color = 'red'
            hr_history.clear()
            spo2_history.clear()
        
        # Plot signals
        ax1.clear()
        plot_len = min(150, len(time_arr))
        plot_times = time_arr[-plot_len:]
        plot_red = red_arr[-plot_len:]
        plot_ir = ir_arr[-plot_len:]
        
        ax1.plot(plot_times, plot_red, 'r-', label='Red', linewidth=2)
        ax1.plot(plot_times, plot_ir, 'purple', label='IR', linewidth=2)
        ax1.set_ylabel('Signal')
        ax1.set_title('Raw Sensor Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Show processed signal with peaks if finger detected
        if finger_ok and len(red_arr) > 40:
            recent = red_arr[-80:] if len(red_arr) > 80 else red_arr
            recent_t = time_arr[-80:] if len(time_arr) > 80 else time_arr
            processed = recent - np.mean(recent)
            
            # Find and mark peaks for visualization
            try:
                peaks, _ = find_peaks(processed, distance=8, prominence=np.std(processed)*0.05)
                if len(peaks) > 0:
                    ax1.plot(recent_t[peaks], recent[peaks], 'go', markersize=8, label='Detected Peaks')
                    ax1.legend()
            except:
                pass
        
        # Results display
        ax2.clear()
        ax2.text(0.05, 0.75, bpm_text, fontsize=32, weight='bold', color=bpm_color)
        ax2.text(0.05, 0.55, spo2_text, fontsize=32, weight='bold', color=spo2_color)
        ax2.text(0.05, 0.35, status_text, fontsize=20, color=status_color)
        
        if not finger_ok:
            ax2.text(0.05, 0.15, "Tips:\n• Cover sensor completely\n• Press gently but firmly\n• Keep finger still", 
                    fontsize=16, color='gray')
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.pause(0.08)
        time.sleep(0.02)
        
except KeyboardInterrupt:
    print("\nStopped.")
    plt.ioff()
    plt.show()