from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import base64
from google import genai
from google.genai import types
import re
from dotenv import load_dotenv
from fast_plate_ocr import LicensePlateRecognizer

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION FROM .ENV ====================

# Model Paths
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "./models/best.pt")
FASTPLATE_MODEL_NAME = os.getenv("FASTPLATE_MODEL_NAME", "cct-xs-v1-global-model")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Server Configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
SERVER_RELOAD = os.getenv("SERVER_RELOAD", "True").lower() == "true"

# API Configuration
API_TITLE = os.getenv("API_TITLE", "License Plate Detection & Recognition API")
API_DESCRIPTION = os.getenv("API_DESCRIPTION", "YOLO + FastPlate OCR + Gemini AI Pipeline")
API_VERSION = os.getenv("API_VERSION", "2.0.0")

# YOLO Configuration
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.25"))

# OCR Configuration
OCR_MIN_TEXT_LENGTH = int(os.getenv("OCR_MIN_TEXT_LENGTH", "2"))

# Image Processing
PLATE_PADDING = int(os.getenv("PLATE_PADDING", "10"))
MIN_PLATE_HEIGHT = int(os.getenv("MIN_PLATE_HEIGHT", "50"))
MIN_PLATE_WIDTH = int(os.getenv("MIN_PLATE_WIDTH", "150"))
MIN_SCALE_FACTOR = float(os.getenv("MIN_SCALE_FACTOR", "2.0"))

# Gemini Configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "1000"))

# CORS Configuration
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "True").lower() == "true"

# ==================== INITIALIZE FASTAPI APP ====================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== INITIALIZE MODELS ====================

yolo_model = None
ocr_model = None
gemini_client = None

print("=" * 70)
print("🚀 INITIALIZING MODELS")
print("=" * 70)

# Load YOLO
try:
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"✅ YOLO model loaded successfully from {YOLO_MODEL_PATH}")
    else:
        print(f"❌ YOLO model not found at {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading YOLO: {e}")

# Load FastPlate OCR
try:
    print(f"📥 Loading FastPlate OCR model: {FASTPLATE_MODEL_NAME}")
    ocr_model = LicensePlateRecognizer(FASTPLATE_MODEL_NAME)
    print(f"✅ FastPlate OCR model '{FASTPLATE_MODEL_NAME}' loaded successfully")
except Exception as e:
    print(f"❌ Error loading FastPlate OCR: {e}")
    print("💡 Make sure you installed: pip install 'fast-plate-ocr[onnx]'")
    import traceback
    traceback.print_exc()

# Initialize Gemini
try:
    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini API initialized successfully")
    else:
        print("⚠️ Gemini API key not provided in .env file")
except Exception as e:
    print(f"❌ Error initializing Gemini: {e}")

print("=" * 70)

# ==================== HELPER FUNCTIONS ====================

def enhance_plate_image(image):
    """Enhance license plate image for better OCR"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        print(f"Enhancement error: {e}")
        return image

def clean_plate_text(text):
    """Clean and format plate text"""
    if not text:
        return ""
    cleaned = re.sub(r'[^A-Z0-9\-]', '', str(text).upper())
    cleaned = re.sub(r'\s+', '', cleaned).strip()
    return cleaned if len(cleaned) >= OCR_MIN_TEXT_LENGTH else ""

def get_vehicle_insights_from_gemini(plate_text, plate_crop_image=None, full_image=None):
    """
    Get comprehensive vehicle insights using Gemini Vision API
    
    Args:
        plate_text: Detected license plate text
        plate_crop_image: Cropped image of the license plate (numpy array RGB)
        full_image: Full vehicle image (numpy array RGB)
    """
    
    if not gemini_client:
        return {
            "error": "Gemini API not available"
        }
    
    try:
        print(f"\n{'='*60}")
        print(f"🤖 GEMINI VISION DEBUG")
        print(f"{'='*60}")
        print(f"Plate text: {plate_text}")
        print(f"Full image type: {type(full_image)}")
        print(f"Full image shape: {full_image.shape if full_image is not None else 'None'}")
        print(f"Plate crop type: {type(plate_crop_image)}")
        print(f"Plate crop shape: {plate_crop_image.shape if plate_crop_image is not None else 'None'}")
        
        # Prepare the prompt
        prompt = f"""
        Analyze this vehicle image and provide detailed information.
        
        **Detected License Plate Number: {plate_text}**
        
        Please provide a comprehensive analysis including:
        
        1. **License Plate Analysis:**
           - Country of origin
           - State/Region (if applicable)
           - Format validity
           - Registration type (private/commercial/government)
        
        2. **Vehicle Identification:**
           - Make and model (be as specific as possible)
           - Approximate year/generation
           - Body type (sedan, SUV, hatchback, truck, etc.)
           - Color
        
        3. **Visual Details:**
           - Any visible damage or modifications
           - Condition (excellent/good/fair/poor)
           - Special features or accessories visible
        
        4. **Additional Observations:**
           - Environment/location context
           - Time of day (if determinable)
           - Any other relevant details
        
        Provide the response in a clear, structured markdown format.
        If you cannot determine certain information, mention "Unable to determine" for that field.
        """
        
        # Prepare content parts
        content_parts = []
        
        # Add full image if available
        if full_image is not None:
            try:
                print(f"📸 Processing full image...")
                
                if len(full_image.shape) == 3 and full_image.shape[2] == 3:
                    full_image_bgr = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
                    success, buffer = cv2.imencode('.jpg', full_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if not success:
                        print(f"❌ Failed to encode image")
                        raise Exception("Image encoding failed")
                    
                    image_bytes = buffer.tobytes()
                    print(f"✅ Image encoded: {len(image_bytes)} bytes")
                    
                    image_part = types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    )
                    
                    content_parts.append(image_part)
                    print(f"✅ Image part created and added")
                else:
                    print(f"⚠️ Unexpected image shape: {full_image.shape}")
                    
            except Exception as img_err:
                print(f"❌ Failed to process image: {img_err}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ No full image provided")
        
        # Add plate crop if available
        if plate_crop_image is not None:
            try:
                print(f"📸 Processing plate crop...")
                
                if len(plate_crop_image.shape) == 3 and plate_crop_image.shape[2] == 3:
                    plate_crop_bgr = cv2.cvtColor(plate_crop_image, cv2.COLOR_RGB2BGR)
                    success, buffer = cv2.imencode('.jpg', plate_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if success:
                        plate_bytes = buffer.tobytes()
                        print(f"✅ Plate crop encoded: {len(plate_bytes)} bytes")
                        
                        plate_part = types.Part.from_bytes(
                            data=plate_bytes,
                            mime_type="image/jpeg"
                        )
                        
                        content_parts.append(plate_part)
                        print(f"✅ Plate crop part created and added")
                    
            except Exception as crop_err:
                print(f"⚠️ Failed to process plate crop: {crop_err}")
        
        # Add prompt at the BEGINNING
        content_parts.insert(0, prompt)
        
        print(f"\n📤 Sending to Gemini:")
        print(f"   Total parts: {len(content_parts)}")
        print(f"   Part types: {[type(p).__name__ for p in content_parts]}")
        
        # Make Gemini API call
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,  # This will now use "gemini-1.5-flash"
            contents=content_parts,
            config=types.GenerateContentConfig(
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_TOKENS
            )
        )
        
        # === THIS IS THE FIX FROM LAST TIME ===
        # Check if the response text is None (due to safety block, etc.)
        if response.text is None:
            print("❌ Gemini response was empty or blocked.")
            try:
                # Try to log safety feedback if it exists
                print(f"   Safety Feedback: {response.prompt_feedback}")
            except Exception:
                pass # No feedback available
            raise Exception("Gemini returned an empty or blocked response.")
        # === END OF FIX ===
        
        print(f"✅ Gemini response received: {len(response.text)} characters")
        print(f"{'='*60}\n")
        
        return {
            "raw_analysis": response.text,
            "plate_number": plate_text,
            "ai_provider": f"Gemini ({GEMINI_MODEL}) with Vision",
            "has_image_analysis": True
        }
        
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "plate_number": plate_text,
            "ai_provider": "Gemini (Error)"
        }
def draw_boxes_on_image(image, detections):
    """Draw bounding boxes and labels on image"""
    annotated = image.copy()
    
    for detection in detections:
        box = detection['box']
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        plate_text = detection['plate_text']
        yolo_conf = detection['yolo_confidence']
        
        # Draw rectangle (green box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Prepare label
        label = f"{plate_text} ({yolo_conf:.2f})"
        
        # Calculate label size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background (green)
        cv2.rectangle(annotated, 
                     (x1, y1 - label_height - 10), 
                     (x1 + label_width + 10, y1), 
                     (0, 255, 0), 
                     -1)
        
        # Draw label text (black)
        cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                   font, font_scale, (0, 0, 0), thickness)
    
    return annotated

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "License Plate Detection API",
        "status": "🚀 Running",
        "version": API_VERSION,
        "endpoints": {
            "/detect": "POST - Detect and recognize license plates",
            "/health": "GET - Check system health",
            "/test": "GET - Test model status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "yolo_loaded": yolo_model is not None,
        "ocr_loaded": ocr_model is not None,
        "gemini_loaded": gemini_client is not None,
        "ready": all([yolo_model, ocr_model])
    }

@app.get("/test")
async def test_models():
    """Test model availability"""
    return {
        "yolo_status": f"✅ Loaded" if yolo_model else "❌ Not loaded",
        "ocr_status": f"✅ Loaded ({FASTPLATE_MODEL_NAME})" if ocr_model else "❌ Not loaded",
        "gemini_status": "✅ Connected" if gemini_client else "❌ Not connected",
        "yolo_path": YOLO_MODEL_PATH,
        "ocr_model": FASTPLATE_MODEL_NAME,
        "system_ready": all([yolo_model, ocr_model])
    }

@app.post("/detect")
async def detect_license_plates(file: UploadFile = File(...)):
    """
    Main detection endpoint with bounding box visualization and Gemini Vision analysis
    """
    
    if not yolo_model or not ocr_model:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check /test endpoint for status."
        )
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        print(f"📷 Processing image: {image.shape}")
        
        # Convert to RGB for YOLO
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Run YOLO detection
        print(f"🎯 Running YOLO detection (conf={YOLO_CONFIDENCE_THRESHOLD})...")
        yolo_results = yolo_model.predict(image_rgb, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        
        detections = []
        
        for result in yolo_results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            print(f"🎯 YOLO detected {len(result.boxes)} plate(s)")
            
            for i, box in enumerate(result.boxes):
                try:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    yolo_conf = float(box.conf[0].cpu().numpy())
                    
                    # Validate coordinates
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Crop plate region with padding
                    pad = PLATE_PADDING
                    x1_p = max(0, x1 - pad)
                    y1_p = max(0, y1 - pad)
                    x2_p = min(w, x2 + pad)
                    y2_p = min(h, y2 + pad)
                    
                    plate_crop = image_rgb[y1_p:y2_p, x1_p:x2_p].copy()
                    
                    if plate_crop.size == 0:
                        continue
                    
                    # Enhance image
                    enhanced_plate = enhance_plate_image(plate_crop)
                    
                    # Resize for better OCR if too small
                    h_crop, w_crop = enhanced_plate.shape[:2]
                    if h_crop < MIN_PLATE_HEIGHT or w_crop < MIN_PLATE_WIDTH:
                        scale = max(MIN_PLATE_HEIGHT/h_crop, MIN_PLATE_WIDTH/w_crop, MIN_SCALE_FACTOR)
                        new_h = int(h_crop * scale)
                        new_w = int(w_crop * scale)
                        enhanced_plate = cv2.resize(enhanced_plate, (new_w, new_h))
                    
                    plate_text = "NO_TEXT_DETECTED"
                    ocr_confidence = 0.0
                    
                    try:
                        # Run FastPlate OCR
                        print(f"🔄 Running FastPlate OCR on detection #{i+1}...")
                        ocr_result = ocr_model.run(enhanced_plate)
                        
                        print(f"🔤 Raw OCR output for detection #{i+1}: {ocr_result}")
                        
                        # Handle different return types
                        if ocr_result:
                            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                                raw_text = str(ocr_result[0])
                            elif isinstance(ocr_result, str):
                                raw_text = ocr_result
                            else:
                                raw_text = str(ocr_result)
                            
                            plate_text = clean_plate_text(raw_text)
                            
                            if plate_text:
                                ocr_confidence = 0.92
                                print(f"✅ Cleaned plate text #{i+1}: '{plate_text}'")
                        
                    except Exception as ocr_error:
                        print(f"❌ OCR Error on detection #{i+1}: {ocr_error}")
                        import traceback
                        traceback.print_exc()
                        plate_text = "OCR_ERROR"
                    
                    # Get Gemini Vision insights - PASS THE IMAGES!
                    vehicle_insights = None
                    if plate_text and plate_text not in ["NO_TEXT_DETECTED", "OCR_ERROR", ""]:
                        print(f"🤖 Getting Gemini insights for '{plate_text}'...")
                        vehicle_insights = get_vehicle_insights_from_gemini(
                            plate_text=plate_text,
                            plate_crop_image=plate_crop,  # ← IMPORTANT: Pass plate crop
                            full_image=image_rgb           # ← IMPORTANT: Pass full image
                        )
                    
                    # Prepare detection result
                    detection = {
                        "box": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": x2 - x1,
                            "height": y2 - y1
                        },
                        "plate_text": plate_text,
                        "yolo_confidence": round(yolo_conf, 3),
                        "ocr_confidence": round(ocr_confidence, 3),
                        "vehicle_insights": vehicle_insights
                    }
                    
                    detections.append(detection)
                    print(f"✅ Detection #{i+1} completed: {plate_text}")
                    
                except Exception as box_error:
                    print(f"❌ Error processing detection {i}: {box_error}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"🎉 Total detections: {len(detections)}")
        
        # Draw boxes on image
        annotated_image = draw_boxes_on_image(image_rgb, detections)
        
        # Convert annotated image to base64
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', annotated_bgr)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse({
            "status": "success",
            "image_name": file.filename,
            "plates_detected": len(detections),
            "results": detections,
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}"
        })
        
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/analyze-plate")
async def analyze_plate_text(plate_number: str):
    """
    Standalone endpoint to get vehicle insights from plate number without image upload
    """
    if not gemini_client:
        raise HTTPException(
            status_code=503,
            detail="Gemini API not available"
        )
    
    try:
        insights = get_vehicle_insights_from_gemini(plate_number)
        return JSONResponse({
            "status": "success",
            "plate_number": plate_number,
            "insights": insights
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=SERVER_RELOAD
    )
