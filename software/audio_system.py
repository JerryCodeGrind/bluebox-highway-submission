import pyaudio
import wave
import numpy as np
import threading
import time
import logging
from collections import deque
import speech_recognition as sr
import pyttsx3
from scipy import signal
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioSystem:
    def __init__(self, input_device_name="ReSpeaker", output_device_name="HiFiBerry"):
        """
        Initialize audio system with ReSpeaker mic array and HiFiBerry output
        
        Args:
            input_device_name: Name/part of input device name
            output_device_name: Name/part of output device name
        """
        self.input_device_name = input_device_name
        self.output_device_name = output_device_name
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 4  # ReSpeaker has 4 mics
        self.format = pyaudio.paInt16
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Find devices
        self.input_device_index = self.find_device_index(input_device_name, is_input=True)
        self.output_device_index = self.find_device_index(output_device_name, is_input=False)
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=self.input_device_index)
        
        # Text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speaking rate
        self.tts_engine.setProperty('volume', 0.8)  # Volume level
        
        # Recording state
        self.is_recording = False
        self.recording_thread = None
        self.audio_buffer = deque(maxlen=self.sample_rate * 30)  # 30 seconds buffer
        self.recording_lock = threading.Lock()
        
        # Voice activation detection
        self.vad_threshold = 0.01  # Voice activity threshold
        self.vad_window_size = 1024
        self.silence_duration = 2.0  # Seconds of silence before stopping
        
        # Audio processing
        self.noise_gate_threshold = 0.005
        self.apply_noise_reduction = True
        
        logger.info(f"Audio system initialized - Input: {self.input_device_index}, Output: {self.output_device_index}")
    
    def find_device_index(self, device_name, is_input=True):
        """Find audio device index by name"""
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            if device_name.lower() in device_info['name'].lower():
                if is_input and device_info['maxInputChannels'] > 0:
                    logger.info(f"Found input device: {device_info['name']} (index {i})")
                    return i
                elif not is_input and device_info['maxOutputChannels'] > 0:
                    logger.info(f"Found output device: {device_info['name']} (index {i})")
                    return i
        
        # Fallback to default device
        if is_input:
            logger.warning(f"Device '{device_name}' not found, using default input")
            return None
        else:
            logger.warning(f"Device '{device_name}' not found, using default output")
            return None
    
    def calculate_rms(self, audio_data):
        """Calculate RMS (Root Mean Square) of audio data"""
        return np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
    
    def apply_noise_gate(self, audio_data):
        """Apply noise gate to reduce background noise"""
        rms = self.calculate_rms(audio_data)
        if rms < self.noise_gate_threshold:
            return np.zeros_like(audio_data)
        return audio_data
    
    def beamforming(self, multi_channel_data):
        """Simple beamforming using mic array"""
        # Convert to proper shape (samples, channels)
        if len(multi_channel_data.shape) == 1:
            return multi_channel_data
        
        # Simple delay-and-sum beamforming
        # This is a basic implementation - more sophisticated algorithms exist
        beamformed = np.mean(multi_channel_data, axis=1)
        return beamformed.astype(np.int16)
    
    def noise_reduction(self, audio_data):
        """Apply basic noise reduction"""
        if not self.apply_noise_reduction:
            return audio_data
        
        # Apply high-pass filter to remove low-frequency noise
        nyquist = self.sample_rate / 2
        low_cutoff = 300  # Hz
        high_cutoff = 3400  # Hz
        
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio_data.astype(np.float32))
        
        return filtered.astype(np.int16)
    
    def start_recording(self):
        """Start continuous audio recording"""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()
        logger.info("Audio recording started")
    
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        logger.info("Audio recording stopped")
    
    def _recording_loop(self):
        """Main recording loop"""
        try:
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size
            )
            
            while self.is_recording:
                try:
                    # Read audio data
                    audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Convert to numpy array
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Reshape for multi-channel (if applicable)
                    if self.channels > 1:
                        audio_np = audio_np.reshape(-1, self.channels)
                        # Apply beamforming
                        audio_np = self.beamforming(audio_np)
                    
                    # Apply audio processing
                    audio_np = self.apply_noise_gate(audio_np)
                    audio_np = self.noise_reduction(audio_np)
                    
                    # Store in buffer
                    with self.recording_lock:
                        self.audio_buffer.extend(audio_np)
                
                except Exception as e:
                    logger.error(f"Recording error: {e}")
                    time.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Recording loop error: {e}")
    
    def get_audio_level(self):
        """Get current audio level (0-1)"""
        with self.recording_lock:
            if len(self.audio_buffer) < self.chunk_size:
                return 0.0
            
            recent_audio = np.array(list(self.audio_buffer)[-self.chunk_size:])
            rms = self.calculate_rms(recent_audio)
            
            # Normalize to 0-1 range
            return min(rms / 0.1, 1.0)
    
    def is_speech_detected(self):
        """Check if speech is currently being detected"""
        return self.get_audio_level() > self.vad_threshold
    
    def listen_for_speech(self, timeout=5, phrase_time_limit=None):
        """Listen for speech and return recognized text"""
        try:
            logger.info("Listening for speech...")
            
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for audio
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            # Recognize speech
            try:
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Recognized: {text}")
                return text
            except sr.UnknownValueError:
                logger.info("Could not understand audio")
                return None
            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                return None
                
        except sr.WaitTimeoutError:
            logger.info("Listening timeout")
            return None
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None
    
    def speak(self, text, blocking=True):
        """Convert text to speech and play it"""
        logger.info(f"Speaking: {text}")
        
        try:
            if blocking:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                # Non-blocking speech
                thread = threading.Thread(target=self._speak_async, args=(text,), daemon=True)
                thread.start()
        
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    def _speak_async(self, text):
        """Asynchronous speech synthesis"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"Async TTS error: {e}")
    
    def save_audio_clip(self, filename, duration=5):
        """Save recent audio to file"""
        samples_needed = int(duration * self.sample_rate)
        
        with self.recording_lock:
            if len(self.audio_buffer) < samples_needed:
                logger.warning(f"Not enough audio data for {duration}s clip")
                return False
            
            # Get recent audio
            audio_data = np.array(list(self.audio_buffer)[-samples_needed:])
        
        # Save to file
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono after beamforming
                wav_file.setsampwidth(self.p.get_sample_size(self.format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            logger.info(f"Audio saved to {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False
    
    def get_audio_stats(self):
        """Get audio system statistics"""
        with self.recording_lock:
            buffer_duration = len(self.audio_buffer) / self.sample_rate
            current_level = self.get_audio_level()
        
        return {
            'is_recording': self.is_recording,
            'buffer_duration': buffer_duration,
            'current_level': current_level,
            'speech_detected': self.is_speech_detected(),
            'sample_rate': self.sample_rate,
            'channels': self.channels
        }
    
    def interactive_mode(self):
        """Interactive voice control mode"""
        logger.info("Starting interactive voice mode...")
        self.start_recording()
        
        try:
            while True:
                self.speak("How can I help you?")
                
                # Listen for command
                command = self.listen_for_speech(timeout=10)
                
                if command:
                    if "stop" in command.lower() or "exit" in command.lower():
                        self.speak("Goodbye!")
                        break
                    elif "measure" in command.lower():
                        self.speak("Starting measurement...")
                        # Integration point for other sensors
                    elif "status" in command.lower():
                        stats = self.get_audio_stats()
                        self.speak(f"Audio level is {stats['current_level']:.2f}")
                    else:
                        self.speak("I didn't understand that command.")
                else:
                    self.speak("I didn't hear anything. Try again.")
                
                time.sleep(1)
        
        finally:
            self.stop_recording()
    
    def cleanup(self):
        """Cleanup audio resources"""
        logger.info("Cleaning up audio system...")
        self.stop_recording()
        self.tts_engine.stop()
        self.p.terminate()

# Example usage
if __name__ == "__main__":
    audio_system = AudioSystem()
    
    try:
        # Test basic functionality
        audio_system.start_recording()
        
        # Monitor audio for 10 seconds
        for i in range(10):
            stats = audio_system.get_audio_stats()
            print(f"Audio Level: {stats['current_level']:.3f}, "
                  f"Speech: {stats['speech_detected']}, "
                  f"Buffer: {stats['buffer_duration']:.1f}s")
            time.sleep(1)
        
        # Test speech recognition
        audio_system.speak("Please say something")
        text = audio_system.listen_for_speech(timeout=5)
        if text:
            audio_system.speak(f"You said: {text}")
        
        # Save audio clip
        audio_system.save_audio_clip("test_recording.wav", duration=3)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        audio_system.cleanup()