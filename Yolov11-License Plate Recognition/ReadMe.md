# License Plate Detection and OCR Pipeline

A comprehensive pipeline for detecting vehicles, tracking them across frames, detecting license plates, and extracting text using OCR.

## 📋 Overview

This project implements a multi-stage computer vision pipeline that:

1. **Detects vehicles** using YOLOv11
2. **Tracks vehicles** across frames using SORT (Simple Online and Realtime Tracking)
3. **Detects license plates** within vehicle regions of interest (ROI)
4. **Extracts license plate text** using EasyOCR
5. **Validates and formats** extracted text based on specific plate patterns

The pipeline is designed for both image and video processing, with built-in tracking to maintain vehicle identities across frames.

## 🏗️ Architecture

```
Input (Image/Video)
    │
    ├── Vehicle Detection (YOLOv11)
    │
    ├── Vehicle Tracking (SORT)
    │
    ├── ROI Extraction per Vehicle
    │
    ├── License Plate Detection (YOLOv11)
    │
    ├── Image Preprocessing (Grayscale + Thresholding)
    │
    ├── Text Extraction (EasyOCR)
    │
    ├── Text Validation & Formatting
    │
    └── Output Visualization & Results
```

## 🚀 Installation

### Prerequisites

- **Python 3.8+**
- **OpenCV**
- **PyTorch**
- **CUDA** (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install ultralytics opencv-python numpy filterpy easyocr torch
```

### Additional Setup

```bash
# For SORT tracker
git clone https://github.com/abewley/sort.git
# Or install via pip if available
```

## 📁 Project Structure

```
project/
│
├── Licensence_Plate_Detector_OCR.py  # Main pipeline class
├── requirements.txt                   # Dependencies
├── models/
│   ├── best.pt                       # Trained license plate model
│   └── yolo11m.pt                    # Vehicle detection model
├── inputs/                           # Input images/videos
├── outputs/                          # Results and processed files
└── test_script.py                    # Example usage
```

---

# 🧩 Class: `LicensePlateDetectorOCR`

### Initialization

```python
detector = LicensePlateDetectorOCR(
    license_plate_model_path="models/best.pt",
    car_yolo_model_path="models/yolo11m.pt",
    vehicle_classes=[2, 3, 5, 7],  # Car, Motorcycle, Bus, Truck
    max_age_sort_track=30,          # SORT: max frames to keep lost track
    min_hits_sort_track=3,          # SORT: min detections to start tracking
    conf_threshold=0.5,             # YOLO confidence threshold
    iou_threshold=0.4,              # YOLO IoU threshold
    device='cpu'                    # 'cpu' or 'cuda'
)
```

**Parameters Explained:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `license_plate_model_path` | Path to trained license plate detection model | `"models/best.pt"` |
| `car_yolo_model_path` | Path to YOLO vehicle detection model | `"models/yolo11m.pt"` |
| `vehicle_classes` | COCO class IDs for vehicles to detect | `[2, 3, 5, 7]` |
| `max_age_sort_track` | Maximum frames to keep lost tracks | `30` |
| `min_hits_sort_track` | Minimum detections to initiate tracking | `3` |
| `conf_threshold` | Minimum confidence for YOLO detections | `0.5` |
| `iou_threshold` | Intersection over Union threshold for NMS | `0.4` |
| `device` | Processing device (`'cpu'` or `'cuda'`) | `'cpu'` |

---

## 🔍 Method Details

### 1. `detect_vehicles(frame)`

Detects vehicles in a single frame using YOLO.

```python
detections = detector.detect_vehicles(frame)
```

**Parameters:**
- `frame` (numpy.ndarray): Input image/frame

**Returns:**
- `numpy.ndarray`: Array of vehicle detections `[x1, y1, x2, y2, confidence]`

**Example:**
```python
frame = cv2.imread("car_image.jpg")
detections = detector.detect_vehicles(frame)
```

---

### 2. `track_vehicles(detections)`

Tracks detected vehicles across frames using SORT algorithm.

```python
tracked_vehicles = detector.track_vehicles(detections)
```

**Parameters:**
- `detections` (numpy.ndarray): Vehicle detections from `detect_vehicles()`

**Returns:**
- `numpy.ndarray`: Tracked vehicles `[x1, y1, x2, y2, track_id]`

**Example:**
```python
tracked_vehicles = detector.track_vehicles(detections)
```

---

### 3. `track_car_of_interest(license_plate, tracked_vehicles)`

Associates a license plate detection with a specific tracked vehicle.

```python
vehicle = detector.track_car_of_interest(license_plate, tracked_vehicles)
```

**Parameters:**
- `license_plate` (list): License plate coordinates `[x1, y1, x2, y2, confidence]`
- `tracked_vehicles` (numpy.ndarray): Tracked vehicles from `track_vehicles()`

**Returns:**
- `list` or `None`: Vehicle that contains the license plate

**Example:**
```python
vehicle = detector.track_car_of_interest(license_plate, tracked_vehicles)
```

---

### 4. `detect_license_plates_in_roi(frame, roi_bbox)`

Detects license plates within a specific region (vehicle bounding box).

```python
license_plates, thresholded_crop = detector.detect_license_plates_in_roi(frame, roi_bbox)
```

**Parameters:**
- `frame` (numpy.ndarray): Input frame
- `roi_bbox` (tuple): Region coordinates `(x1, y1, x2, y2)`

**Returns:**
- `tuple`: `(license_plates, thresholded_crop)`
  - `license_plates`: List of detected plates `[x1, y1, x2, y2, confidence]`
  - `thresholded_crop`: Preprocessed image for OCR

**Example:**
```python
license_plates, crop = detector.detect_license_plates_in_roi(frame, vehicle_bbox)
```

---

### 5. `check_license_plate_compliance(text)`

Validates if extracted text matches the expected license plate format (7-character UK-style).

```python
is_valid = detector.check_license_plate_compliance("AB12CDE")
```

**Parameters:**
- `text` (str): Extracted license plate text

**Returns:**
- `bool`: `True` if format is valid

**Format Pattern:**
- **AA##AAA** (Letter-Letter-Number-Number-Letter-Letter-Letter)

**Example:**
```python
is_valid = detector.check_license_plate_compliance("AB12CDE")  # True
is_valid = detector.check_license_plate_compliance("A123BC")   # False
```

---

### 6. `format_license(text)`

Applies character correction based on position-specific mapping.

```python
formatted = detector.format_license("0I12CD3")
```

**Parameters:**
- `text` (str): Raw OCR text

**Returns:**
- `str`: Formatted license plate text

**Character Mapping:**
- **Positions 0, 1, 4, 5, 6**: Common OCR errors (0→O, 1→I, 5→S, etc.)
- **Positions 2, 3**: Numbers that might be misread as letters

**Example:**
```python
formatted = detector.format_license("0I12CD3")  # Returns "OIL2CDE"
formatted = detector.format_license("AB1ZCDE")  # Returns "AB12CDE"
```

---

### 7. `read_license_plate_ocr(license_plate_crop)`

Extracts text from a license plate image using EasyOCR.

```python
text, confidence = detector.read_license_plate_ocr(license_plate_crop)
```

**Parameters:**
- `license_plate_crop` (numpy.ndarray): Preprocessed license plate image

**Returns:**
- `tuple`: `(formatted_text, confidence_score)`

**Example:**
```python
text, confidence = detector.read_license_plate_ocr(license_plate_crop)
print(f"Plate: {text}, Confidence: {confidence:.2f}")
```

---

### 8. `process_image(image_path, output_path=None, visualize=True)`

Complete pipeline for single image processing.

```python
results = detector.process_image("input.jpg", output_path="output.jpg", visualize=True)
```

**Parameters:**
- `image_path` (str): Path to input image
- `output_path` (str, optional): Path to save annotated image
- `visualize` (bool): Whether to display results

**Returns:**
- `list`: Detection results for each vehicle

**Example:**
```python
results = detector.process_image(
    "input.jpg",
    output_path="output.jpg",
    visualize=True
)
```

---

### 9. `process_video(video_path, output_path=None, max_frames=None, skip_frames=0, visualize=False)`

Complete pipeline for video processing.

```python
results = detector.process_video("input.mp4", output_path="output.mp4", max_frames=300)
```

**Parameters:**
- `video_path` (str): Path to input video
- `output_path` (str, optional): Path to save processed video
- `max_frames` (int, optional): Limit processing to N frames
- `skip_frames` (int): Process every N+1th frame for speed
- `visualize` (bool): Display processing in real-time

**Returns:**
- `list`: Detection results for each frame

**Example:**
```python
results = detector.process_video(
    "input.mp4",
    output_path="output.mp4",
    max_frames=300,
    skip_frames=1,
    visualize=False
)
```

---

### 10. `save_results(results, save_path=None)`

Saves detection results to a text file.

```python
detector.save_results(results, "results.txt")
```

**Parameters:**
- `results` (list): Detection results from processing
- `save_path` (str, optional): Custom save path

**Example Output:**
```
License Plate Detection Results
==================================================

Detection 1:
  Frame: 42
  Vehicle ID: 5
  Vehicle BBox: (320, 180, 450, 280)
  License Plate BBox: (350, 220, 420, 240)
  License Plate Text: AB12CDE
  OCR Confidence: 0.892
  LP Detection Confidence: 0.956
```

---

## 📊 Output Format

Each detection result contains:

```python
{
    'frame': int,                    # Frame number
    'vehicle_id': int,               # Tracked vehicle ID
    'vehicle_bbox': tuple,           # (x1, y1, x2, y2)
    'lp_bbox': tuple,                # License plate coordinates
    'lp_text': str,                  # Extracted text or "NOT_FOUND"
    'ocr_confidence': float,         # OCR confidence (0-1)
    'lp_confidence': float           # Detection confidence (0-1)
}
```

---

## 🎯 Complete Usage Examples

### Example 1: Basic Image Processing

```python
from Licensence_Plate_Detector_OCR import LicensePlateDetectorOCR

# Initialize detector
detector = LicensePlateDetectorOCR(
    license_plate_model_path="models/best.pt",
    car_yolo_model_path="models/yolo11n.pt",
    conf_threshold=0.3
)

# Process single image
results = detector.process_image(
    "inputs/parking_lot.jpg",
    output_path="outputs/detected.jpg",
    visualize=True
)

# Print results
for result in results:
    print(f"Vehicle {result['vehicle_id']}: {result['lp_text']}")
```

### Example 2: Video Processing with Custom Settings

```python
detector = LicensePlateDetectorOCR(
    license_plate_model_path="models/best.pt",
    car_yolo_model_path="models/yolo11m.pt",
    vehicle_classes=[2, 5, 7],  # Only cars, buses, trucks
    conf_threshold=0.4,
    max_age_sort_track=20,
    device='cuda'  # Use GPU if available
)

# Process video
results = detector.process_video(
    video_path="inputs/traffic.mp4",
    output_path="outputs/processed_traffic.mp4",
    max_frames=1000,    # Process first 1000 frames
    skip_frames=2,      # Process every 3rd frame
    visualize=False      # Headless mode
)

# Analyze results
plates_detected = len([r for r in results if r['lp_text'] != "NOT_FOUND"])
print(f"Total license plates detected: {plates_detected}")
```

### Example 3: Custom Integration

```python
import cv2

# Custom processing loop
detector = LicensePlateDetectorOCR("models/best.pt")

cap = cv2.VideoCapture(0)  # Webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Your custom pipeline
    vehicles = detector.detect_vehicles(frame)
    tracked = detector.track_vehicles(vehicles)
    
    for vehicle in tracked:
        x1, y1, x2, y2, vid = vehicle
        plates, crop = detector.detect_license_plates_in_roi(frame, (x1, y1, x2, y2))
        
        if plates:
            text, conf = detector.read_license_plate_ocr(crop)
            if text:
                print(f"Vehicle {vid}: {text}")
    
    cv2.imshow('Custom Pipeline', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🎨 Visualization

The pipeline automatically annotates images/videos with:

- **Green boxes**: Detected vehicles with track ID (e.g., "V5")
- **Red boxes**: Detected license plates
- **Text overlay**: Extracted license plate text with OCR confidence

**Example Visualization Output:**
```
┌─────────────────────────────────────┐
│                                     │
│   ┌───────┐                         │
│   │  V5   │                         │
│   └───────┘                         │
│        ┌───────────┐                │
│        │ AB12CDE   │                │
│        │ (0.89)    │                │
│        └───────────┘                │
│                                     │
└─────────────────────────────────────┘
```

---

## ⚙️ Configuration Tips

### For Better Accuracy:

```python
detector = LicensePlateDetectorOCR(
    conf_threshold=0.6,      # Higher = fewer but more confident detections
    iou_threshold=0.5,       # Higher = more strict overlap requirements
    min_hits_sort_track=5    # Require more detections before tracking
)
```

### For Faster Processing:

```python
# Use smaller models and skip frames
detector = LicensePlateDetectorOCR(
    car_yolo_model_path="yolo11n.pt",  # Nano model is fastest
    device='cpu'                       # Or 'cuda' if available
)

# In process_video:
results = detector.process_video(
    skip_frames=4,  # Process every 5th frame
    max_frames=500  # Limit processing
)
```

### For Specific Use Cases:

```python
# Parking lot monitoring (mostly stationary vehicles)
detector = LicensePlateDetectorOCR(
    max_age_sort_track=60,    # Keep tracks longer
    vehicle_classes=[2]       # Only cars
)

# Highway traffic (fast-moving vehicles)
detector = LicensePlateDetectorOCR(
    max_age_sort_track=10,    # Shorter track memory
    conf_threshold=0.4        # Lower threshold for moving objects
)
```

---

## 🐛 Troubleshooting

### Common Issues:

| Issue | Solution |
|-------|----------|
| **No detections found** | Lower `conf_threshold` (try 0.3)<br>Check if models are loaded correctly<br>Verify input image/video quality |
| **Poor OCR accuracy** | Ensure proper lighting in input<br>Check `license_plate_crop_thresh` visualization<br>Adjust threshold value in `detect_license_plates_in_roi` |
| **Slow performance** | Use smaller YOLO models (nano instead of medium)<br>Increase `skip_frames` parameter<br>Enable GPU with `device='cuda'` |
| **Tracking issues** | Adjust SORT parameters (`max_age`, `min_hits`)<br>Ensure consistent vehicle detection across frames |

---

## 📈 Performance Metrics

| Component | Processing Time (CPU) | Notes |
|-----------|----------------------|-------|
| **Vehicle Detection** | ~50-100ms per frame | Depends on YOLO model size |
| **License Plate Detection** | ~30-60ms per vehicle | Within ROI only |
| **OCR Processing** | ~100-200ms per plate | EasyOCR inference time |
| **Total Pipeline** | ~200-400ms per frame | For 1-2 vehicles per frame |
| **Memory Usage** | ~1-2GB | Models + processing buffers |

---

## 🔮 Future Improvements

Potential enhancements for the pipeline:

1. **Multiple License Plate Formats**
   - Support for different countries/regions
   - Configurable validation patterns

2. **Database Integration**
   - Plate number lookup against databases
   - Stolen vehicle alerts

3. **Advanced Analytics**
   - Vehicle speed estimation from video
   - Traffic flow analysis
   - Parking duration monitoring

4. **Deployment Features**
   - Web interface for easy usage
   - Real-time alert system
   - API endpoints for integration

5. **Model Improvements**
   - Fine-tune on specific camera angles
   - Multi-language OCR support
   - Night vision/low-light enhancements

---

## 📄 License

This project is intended for **educational and research purposes**. Commercial use may require additional permissions and compliance with local regulations regarding surveillance and data privacy.

**Important:** Ensure you have the right to process license plate data in your jurisdiction.

---

## 🙏 Acknowledgments

- **YOLOv11** by Ultralytics
- **SORT tracker** by Alex Bewley
- **EasyOCR** by Jaided AI
- **OpenCV** community
- **COCO dataset** for vehicle classes

---

## ⚠️ Important Note

The character mapping dictionaries are configured for **UK-style license plates** (format: AA##AAA). To adapt for other plate formats:

1. Modify `dict_char_to_int` and `dict_int_to_char` mappings
2. Update `check_license_plate_compliance()` validation logic
3. Adjust `format_license()` position-specific corrections

**Example for US plates (ABC 1234 format):**
```python
# Update validation pattern
us_pattern = r'^[A-Z]{3}\s?\d{4}$'
# Adjust character mappings for US-specific OCR errors
```

---

## 📞 Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review the example scripts
3. Ensure all dependencies are correctly installed
4. Provide sample images/videos when reporting issues

---

**Happy License Plate Detection! 🚗🔍**