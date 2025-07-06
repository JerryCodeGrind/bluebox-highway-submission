import os
import json
import time
import torch
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import io

# Flask for API
from flask import Flask, request, jsonify

# YOLO for object detection
from ultralytics import YOLO

# Use BLIP instead of problematic LLaVA
from transformers import BlipProcessor, BlipForConditionalGeneration

# ============================================================================
# CONFIGURATION
# ============================================================================
YOLO_MODEL_PATH = "yolov8n.pt"
VISION_MODEL_NAME = "Salesforce/blip-image-captioning-large"  # More reliable than LLaVA

HF_TOKEN = "hf-token"

HAND_CONFIDENCE_THRESHOLD = 0.2
TARGET_CLASSES = ['person', 'hand']

app = Flask(__name__)

# Global model storage
models = {
    'yolo': None,
    'vision_processor': None,
    'vision_model': None
}

# ============================================================================
# FORCE GPU SETUP
# ============================================================================
def setup_gpu():
    """Force GPU setup and verification"""
    print("🔧 Setting up GPU environment...")
    
    # Force CUDA if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        print(f"✅ GPU setup complete: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    else:
        print("❌ No GPU detected - check nvidia-smi")
        return torch.device("cpu")

# ============================================================================
# MODEL LOADING WITH FORCED GPU
# ============================================================================
def load_yolo_model():
    try:
        print("🔄 Loading YOLO model...")
        model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO model loaded")
        return model
    except Exception as e:
        print(f"❌ YOLO load failed: {e}")
        return None

def load_vision_models():
    try:
        print("🔄 Loading BLIP vision model (GPU-optimized)...")
        
        # Force GPU device
        device = setup_gpu()
        
        processor = BlipProcessor.from_pretrained(VISION_MODEL_NAME)
        
        if device.type == "cuda":
            model = BlipForConditionalGeneration.from_pretrained(
                VISION_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto"
            ).to(device)
            print(f"✅ BLIP loaded on GPU: {model.device}")
        else:
            model = BlipForConditionalGeneration.from_pretrained(
                VISION_MODEL_NAME,
                torch_dtype=torch.float32
            ).to(device)
            print(f"⚠️ BLIP loaded on CPU: {model.device}")
        
        return processor, model
        
    except Exception as e:
        print(f"❌ BLIP load failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def initialize_models():
    print("🚀 Initializing all models with GPU optimization...")
    
    models['yolo'] = load_yolo_model()
    models['vision_processor'], models['vision_model'] = load_vision_models()
    
    success = all([
        models['yolo'] is not None,
        models['vision_processor'] is not None,
        models['vision_model'] is not None
    ])
    
    if success:
        print("✅ All models loaded successfully!")
        print(f"🎮 GPU Usage: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"📊 GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
    else:
        print("❌ Some models failed to load")
    
    return success

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def detect_objects(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = models['yolo'](image, conf=HAND_CONFIDENCE_THRESHOLD, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = models['yolo'].names[class_id]
                    if class_name in TARGET_CLASSES:
                        detections.append({
                            'class': class_name,
                            'confidence': confidence
                        })
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections[0] if detections else None
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return None

def analyze_with_blip(image_bytes):
    """Use BLIP for reliable image analysis"""
    try:
        print("🔄 Starting BLIP image analysis...")
        start_time = time.time()
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"📷 Image loaded: {image.size}")
        
        # Create medical prompt
        prompt = "a medical photograph showing"
        
        # Process with BLIP
        device = models['vision_model'].device
        inputs = models['vision_processor'](image, prompt, return_tensors="pt").to(device)
        
        print("🔄 BLIP generating analysis...")
        with torch.no_grad():
            out = models['vision_model'].generate(
                **inputs,
                max_length=100,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode result
        analysis = models['vision_processor'].decode(out[0], skip_special_tokens=True)
        
        # Remove the prompt from the output
        if prompt in analysis:
            analysis = analysis.replace(prompt, "").strip()
        
        total_time = time.time() - start_time
        
        print(f"✅ BLIP analysis completed in {total_time:.1f} seconds")
        print("=" * 60)
        print("👁️ BLIP MEDICAL IMAGE ANALYSIS:")
        print("=" * 60)
        print(analysis)
        print("=" * 60)
        
        return analysis, total_time
        
    except Exception as e:
        print(f"❌ BLIP analysis error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def run_analysis_pipeline(image_bytes):
    """Complete analysis pipeline"""
    
    # Step 1: Object Detection
    print("🔍 Step 1: YOLO Object Detection")
    start_total = time.time()
    
    detection = detect_objects(image_bytes)
    if not detection:
        return {"error": "No relevant objects detected"}
    
    print(f"✅ Detected: {detection['class']} ({detection['confidence']:.1%})")
    
    # Step 2: BLIP Analysis
    print("🔍 Step 2: BLIP Image Analysis")
    analysis, analysis_time = analyze_with_blip(image_bytes)
    
    if not analysis:
        return {"error": "Image analysis failed"}
    
    total_time = time.time() - start_total
    
    print(f"✅ PIPELINE COMPLETE in {total_time:.1f} seconds!")
    
    return {
        "success": True,
        "detection": detection,
        "image_analysis": analysis,
        "analysis_time": analysis_time,
        "total_time": total_time,
        "gpu_used": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route("/")
def home():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "gpu_memory_used_gb": torch.cuda.memory_allocated(0) / 1e9
        }
    
    return jsonify({
        "status": "online",
        "message": "Fixed GPU Medical Analysis Server",
        "models_loaded": {
            "yolo": models['yolo'] is not None,
            "vision": models['vision_model'] is not None
        },
        "gpu_available": torch.cuda.is_available(),
        "gpu_info": gpu_info
    })

@app.route("/quick_detect", methods=["POST"])
def quick_detect():
    """Quick YOLO detection only"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
            
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        detection = detect_objects(image_bytes)
        
        if detection:
            return jsonify({
                "detected": True,
                "detection": detection
            })
        else:
            return jsonify({
                "detected": False
            })
            
    except Exception as e:
        print(f"❌ Quick detection error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze_frame", methods=["POST"])
def analyze_frame():
    """Complete analysis pipeline"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
            
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        print(f"🚀 Starting analysis pipeline for {len(image_bytes)} byte image")
        
        result = run_analysis_pipeline(image_bytes)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "yolo": models['yolo'] is not None,
            "vision": models['vision_model'] is not None
        },
        "gpu_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    })

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("🏥 Fixed GPU Medical Analysis Server")
    print("=" * 50)
    print("⚡ Using BLIP for reliable image analysis")
    print("🚀 Optimized for GPU performance")
    
    # Check GPU immediately
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected - will use CPU")
        print("💡 Check: nvidia-smi")
    
    if not initialize_models():
        print("❌ Failed to initialize models - exiting")
        return
    
    print("\n🌐 Starting Flask server...")
    print("📋 Endpoints:")
    print("  GET  /          - Server status with GPU info")
    print("  POST /quick_detect - Fast YOLO detection")
    print("  POST /analyze_frame - Complete analysis pipeline")
    print("  GET  /health    - Health check")
    print("\n🚀 Server ready!")
    
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)

if __name__ == "__main__":
    main()