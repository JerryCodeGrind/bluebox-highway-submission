import io
import requests
import time
from datetime import datetime
from picamera2 import Picamera2

# Configuration
GPU_SERVER_URL = "http://192.222.58.119:7860"

class SimpleMedicalClient:
    def __init__(self):
        self.camera = None
        self.init_camera()
        
    def init_camera(self):
        """Initialize camera"""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(main={"size": (640, 480)})
            self.camera.configure(config)
            print("✅ Camera ready")
        except Exception as e:
            print(f"❌ Camera failed: {e}")
            self.camera = None
            
    def check_server(self):
        """Check server status"""
        try:
            response = requests.get(f"{GPU_SERVER_URL}/", timeout=10)
            if response.status_code == 200:
                print("✅ Medical AI server online")
                return True
            return False
        except Exception as e:
            print(f"❌ Server check failed: {e}")
            return False
            
    def capture_medical_image(self):
        """Capture high-resolution medical image"""
        try:
            if not self.camera:
                print("❌ No camera available")
                return None
                
            # Start camera
            if not self.camera.started:
                self.camera.start()
                time.sleep(2)
            
            # Switch to high-res for medical analysis
            self.camera.stop()
            config = self.camera.create_still_configuration(main={"size": (1024, 768)})
            self.camera.configure(config)
            self.camera.start()
            time.sleep(1)
            
            print("\n📸 MEDICAL IMAGE CAPTURE")
            print("=" * 40)
            print("📷 Position the area of concern clearly in view")
            print("💡 Ensure good lighting for best analysis")
            
            # Countdown
            for i in range(3, 0, -1):
                print(f"📸 Capturing in {i}...")
                time.sleep(1)
            
            print("📸 *CLICK* - Image captured!")
            
            stream = io.BytesIO()
            self.camera.capture_file(stream, format='jpeg')
            stream.seek(0)
            image_bytes = stream.getvalue()
            
            print(f"✅ Image ready: {len(image_bytes)} bytes")
            
            # Return to preview mode
            self.camera.stop()
            config = self.camera.create_preview_configuration(main={"size": (640, 480)})
            self.camera.configure(config)
            self.camera.start()
            
            return image_bytes
            
        except Exception as e:
            print(f"❌ Image capture failed: {e}")
            return None
            
    def analyze_medical_image(self, image_bytes):
        """Send image for complete medical analysis"""
        try:
            print("\n🚀 MEDICAL ANALYSIS IN PROGRESS")
            print("=" * 50)
            print("🔍 Running object detection...")
            print("👁️ Analyzing image with BLIP AI...")
            print("🩺 Processing with medical AI...")
            print("⏳ This may take 30-90 seconds...")
            
            files = {'image': ('medical_scan.jpg', image_bytes, 'image/jpeg')}
            
            response = requests.post(
                f"{GPU_SERVER_URL}/analyze_frame",
                files=files,
                timeout=120  # 2 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    print("\n✅ MEDICAL ANALYSIS COMPLETE!")
                    return result
                else:
                    print(f"❌ Analysis failed: {result.get('error')}")
                    return None
            else:
                print(f"❌ Server error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print("⏱️ Analysis timed out - please try again")
            return None
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None
            
    def display_analysis_results(self, result):
        """Display complete medical analysis results"""
        print("\n" + "=" * 70)
        print("🏥 COMPLETE MEDICAL ANALYSIS RESULTS")
        print("=" * 70)
        
        # Detection info
        detection = result['detection']
        print(f"\n🎯 OBJECT DETECTION:")
        print(f"   Detected: {detection['class']} ({detection['confidence']:.1%} confidence)")
        
        # BLIP analysis
        print(f"\n👁️ VISUAL ANALYSIS (BLIP AI):")
        print("   " + "─" * 60)
        print(f"   {result['blip_analysis']}")
        print("   " + "─" * 60)
        
        # Combined patient info
        print(f"\n📋 PATIENT INFORMATION:")
        print("   " + "─" * 60)
        print(f"   {result['combined_info']}")
        print("   " + "─" * 60)
        
        # Clinical summary
        print(f"\n🩺 CLINICAL SUMMARY:")
        print("   " + "─" * 60)
        print(f"   {result['clinical_summary']}")
        print("   " + "─" * 60)
        
        # Diagnostic analysis
        print(f"\n🔬 DIAGNOSTIC ANALYSIS:")
        print("   " + "─" * 60)
        print(f"   {result['diagnostic_analysis']}")
        print("   " + "─" * 60)
        
        # Final medical question
        print(f"\n❓ AI DOCTOR QUESTION:")
        print("=" * 70)
        print(f"🩺 {result['medical_question']}")
        print("=" * 70)
        
        # Processing times
        print(f"\n⏱️ ANALYSIS TIMING:")
        print(f"   BLIP Processing: {result['blip_time']:.1f} seconds")
        print(f"   Total Pipeline: {result.get('total_time', 0):.1f} seconds")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"   Completed: {timestamp}")
        
        print("\n⚕️ MEDICAL DISCLAIMER:")
        print("   This AI analysis is for informational purposes only.")
        print("   Please consult a healthcare provider for proper medical evaluation.")
        
    def run_medical_scan(self):
        """Run complete medical scan workflow"""
        print("\n🏥 MEDICAL SCAN WORKFLOW")
        print("=" * 50)
        
        if not self.camera:
            print("❌ Camera not available")
            return False
            
        if not self.check_server():
            print("❌ Medical AI server not accessible")
            return False
            
        try:
            # Step 1: Capture medical image
            image_bytes = self.capture_medical_image()
            if not image_bytes:
                print("❌ Failed to capture image")
                return False
            
            # Step 2: Analyze with complete AI pipeline
            result = self.analyze_medical_image(image_bytes)
            if not result:
                print("❌ Medical analysis failed")
                return False
            
            # Step 3: Display comprehensive results
            self.display_analysis_results(result)
            
            print("\n✅ Medical scan completed successfully!")
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Medical scan interrupted")
            return False
        except Exception as e:
            print(f"❌ Scan error: {e}")
            return False
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.camera:
                if self.camera.started:
                    self.camera.stop()
                self.camera.close()
                print("📷 Camera closed")
        except:
            pass

def main():
    print("🏥 AI Medical Scanner")
    print("=" * 30)
    print("📋 Complete Pipeline: Camera → BLIP → AI Doctor")
    print(f"🌐 Server: {GPU_SERVER_URL}")
    print("⚕️ For informational purposes only")
    print("\nPress Ctrl+C to exit\n")
    
    client = SimpleMedicalClient()
    
    try:
        while True:
            print("\n" + "=" * 50)
            input("📋 Press Enter to start medical scan (or Ctrl+C to exit)...")
            
            success = client.run_medical_scan()
            
            if success:
                print("\n🎉 Scan completed!")
            else:
                print("\n❌ Scan failed!")
            
            # Ask if user wants another scan
            while True:
                try:
                    again = input("\n🔄 Perform another scan? (y/n): ").strip().lower()
                    if again in ['y', 'yes']:
                        break
                    elif again in ['n', 'no']:
                        print("👋 Thank you for using AI Medical Scanner!")
                        return
                    else:
                        print("Please enter 'y' or 'n'")
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    return
                    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        client.cleanup()

if __name__ == "__main__":
    main()