"""
Filename: license_plate_pipeline_fixed.py
"""

# Fix OpenMP error at the VERY BEGINNING
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Now import other libraries
import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Try to import SORT
try:
    from sort.sort import Sort
    SORT_AVAILABLE = True
except ImportError:
    print("SORT not found. Install with: pip install filterpy")
    SORT_AVAILABLE = False
    # Simple mock SORT for testing
    class Sort:
        def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
            self.max_age = max_age
            self.trackers = []
            self.frame_count = 0
            
        def update(self, detections):
            if len(detections) == 0:
                return np.empty((0, 5))
            
            tracks = []
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf = det
                track_id = i + 1  # Simple ID assignment
                tracks.append([x1, y1, x2, y2, track_id])
            
            return np.array(tracks)

# Try to import ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Ultralytics not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False
    sys.exit(1)


class LicensePlateDetectorOCR:
    def __init__(self, 
                 license_plate_model_path,
                 car_yolo_model_path="yolo11n.pt",
                 vehicle_classes=[2, 3, 5, 7],  # Car, Motorcycle, Bus, Truck
                 max_age_sort_track=30,
                 min_hits_sort_track=3,
                 conf_threshold=0.5,
                 iou_threshold=0.4,
                 device='cpu'):
        
        # Initialize Device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f"Using device: {self.device}")
        
        # Load models
        print("Loading license plate model...")
        self.license_plate_model = YOLO(license_plate_model_path)
        
        print("Loading vehicle detection model...")
        self.car_detection_model = YOLO(car_yolo_model_path)
        
        self.vehicle_classes = vehicle_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize SORT Tracker
        self.tracker = Sort(max_age=max_age_sort_track, 
                           min_hits=min_hits_sort_track, 
                           iou_threshold=iou_threshold)
        print(f"SORT tracker initialized (max_age={max_age_sort_track}, min_hits={min_hits_sort_track})")
        
        # Data storage
        self.license_plate_data = {}
        
        # Output directory
        self.base_output_dir = Path("Yolov11-License_Plate_Recognition/outputs")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 50)
        print("Pipeline initialized successfully!")
        print("=" * 50)
    
    def detect_vehicles(self, frame):
        """Detect vehicles in frame"""
        results = self.car_detection_model(
            frame, 
            conf=self.conf_threshold,
            verbose=False  # Disable verbose output
        )
        
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                if class_id not in self.vehicle_classes:
                    continue
                
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Filter small detections
                width = x2 - x1
                height = y2 - y1
                if width < 50 or height < 50:
                    continue
                
                detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections) if detections else np.empty((0, 5))
    
    def track_vehicles(self, detections):
        """Track vehicles using SORT"""
        return self.tracker.update(detections)
    
    def detect_license_plates_in_roi(self, frame, roi_bbox):
        """Detect license plates within a specific region"""
        x1, y1, x2, y2 = map(int, roi_bbox)
        
        # Ensure ROI is valid
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return []
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        
        # Detect license plates
        results = self.license_plate_model(
            roi,
            conf=self.conf_threshold,
            verbose=False  # Disable verbose output
        )
        
        license_plates = []
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Convert to original frame coordinates
                license_plates.append([
                    int(rx1 + x1), int(ry1 + y1),
                    int(rx2 + x1), int(ry2 + y1),
                    float(conf)
                ])
        
        return license_plates
    
    def associate_lp_to_vehicle(self, vehicle_bbox, license_plates):
        """Associate license plate to vehicle based on spatial relationship"""
        if not license_plates:
            return None
        
        # Simple association: use the license plate with highest confidence
        best_lp = max(license_plates, key=lambda x: x[4])
        return best_lp
    
    def process_frame(self, frame, frame_num):
        """Process a single frame"""
        # Step 1: Detect vehicles
        vehicle_detections = self.detect_vehicles(frame)
        
        # Step 2: Track vehicles
        tracked_vehicles = self.track_vehicles(vehicle_detections)
        
        # Step 3: For each tracked vehicle, detect license plates
        frame_results = {
            'vehicles': [],
            'license_plates': {}
        }
        
        for vehicle in tracked_vehicles:
            if len(vehicle) < 5:
                continue
                
            x1, y1, x2, y2, track_id = map(int, vehicle[:5])
            
            # Store vehicle info
            frame_results['vehicles'].append({
                'track_id': track_id,
                'bbox': (x1, y1, x2, y2)
            })
            
            # Detect license plates in vehicle ROI
            license_plates = self.detect_license_plates_in_roi(frame, (x1, y1, x2, y2))
            
            if license_plates:
                # Associate license plate to vehicle
                best_lp = self.associate_lp_to_vehicle((x1, y1, x2, y2), license_plates)
                
                if best_lp:
                    frame_results['license_plates'][track_id] = best_lp
                    
                    # Update global tracking data
                    if track_id not in self.license_plate_data:
                        self.license_plate_data[track_id] = {
                            'vehicle_bbox': (x1, y1, x2, y2),
                            'lp_bbox': best_lp[:4],
                            'confidence': best_lp[4],
                            'first_frame': frame_num,
                            'last_frame': frame_num,
                            'count': 1
                        }
                    else:
                        self.license_plate_data[track_id]['last_frame'] = frame_num
                        self.license_plate_data[track_id]['count'] += 1
                        
                        # Update if new detection has higher confidence
                        if best_lp[4] > self.license_plate_data[track_id]['confidence']:
                            self.license_plate_data[track_id]['lp_bbox'] = best_lp[:4]
                            self.license_plate_data[track_id]['confidence'] = best_lp[4]
        
        return frame_results
    
    def draw_results(self, frame, frame_results):
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()
        
        # Draw vehicle bounding boxes
        for vehicle in frame_results['vehicles']:
            track_id = vehicle['track_id']
            x1, y1, x2, y2 = vehicle['bbox']
            
            # Draw vehicle box (green)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"V{track_id}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
            
            # Draw license plate if detected
            if track_id in frame_results['license_plates']:
                lx1, ly1, lx2, ly2, conf = frame_results['license_plates'][track_id]
                
                # Draw license plate box (red)
                cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
                cv2.putText(annotated, f"LP: {conf:.2f}", 
                           (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 255), 2)
        
        # Add frame info
        cv2.putText(annotated, f"Vehicles: {len(frame_results['vehicles'])} | LPs: {len(frame_results['license_plates'])}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def process_video(self, video_path, output_path=None, max_frames=None, skip_frames=2):
        """Process video file"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height}, {fps} FPS")
        
        # Setup output video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None
        
        frame_num = 0
        processed_count = 0
        
        print("Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Limit frames if specified
            if max_frames and processed_count >= max_frames:
                break
            
            # Skip frames for faster processing
            if frame_num % (skip_frames + 1) != 0:
                continue
            
            # Process frame
            frame_results = self.process_frame(frame, frame_num)
            
            # Draw results
            annotated_frame = self.draw_results(frame, frame_results)
            
            # Write to output video
            if out:
                out.write(annotated_frame)
            
            # Save frame image occasionally for debugging
            if processed_count % 50 == 0 and processed_count > 0:
                debug_path = self.base_output_dir / f"frame_{frame_num}.jpg"
                cv2.imwrite(str(debug_path), annotated_frame)
            
            processed_count += 1
            
            # Progress update
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} frames...")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        
        print(f"Processing complete! Processed {processed_count} frames.")
        
        # Save results
        self.save_results(video_path)
        
        return self.license_plate_data
    
    def save_results(self, source_path):
        """Save detection results to file"""
        source_name = Path(source_path).stem
        results_file = self.base_output_dir / f"results_{source_name}.txt"
        
        with open(results_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("LICENSE PLATE DETECTION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Source: {source_path}\n")
            f.write(f"Total vehicles with license plates: {len(self.license_plate_data)}\n\n")
            
            for track_id, data in sorted(self.license_plate_data.items()):
                f.write(f"[Vehicle ID: {track_id}]\n")
                f.write(f"  Vehicle BBox: {data['vehicle_bbox']}\n")
                f.write(f"  License Plate BBox: {data['lp_bbox']}\n")
                f.write(f"  Confidence: {data['confidence']:.4f}\n")
                f.write(f"  First frame: {data['first_frame']}\n")
                f.write(f"  Last frame: {data['last_frame']}\n")
                f.write(f"  Detection count: {data['count']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"Results saved to: {results_file}")


def main():
    """Main function to test the pipeline"""
    print("License Plate Detection Pipeline")
    print("-" * 50)
    
    # Model paths
    license_plate_model = r"D:\Work Projects\AI & ML Projects\AI-ML-Projects\Yolov11-License Plate Recognition\Yolov11-License-Plate-Model\train5\weights\best.pt"  # Your trained model
    car_model = r"D:\Work Projects\AI & ML Projects\AI-ML-Projects\Yolov11-License Plate Recognition\yolo11s.pt"  # Vehicle detection model
    
    # Check if models exist
    if not os.path.exists(license_plate_model):
        print(f"Error: License plate model not found at {license_plate_model}")
        return
    
    # Initialize pipeline
    pipeline = LicensePlateDetectorOCR(
        license_plate_model_path=license_plate_model,
        car_yolo_model_path=car_model,
        conf_threshold=0.3,  # Lower threshold for more detections
        device='cuda'  # Change to 'cuda' if GPU available
    )
    
    # Test video path
    test_video = r"C:\Users\basel\Downloads\2103099-hd_1280_720_60fps.mp4"
    
    if os.path.exists(test_video):
        print(f"\nProcessing video: {test_video}")
        
        # Process video
        results = pipeline.process_video(
            video_path=test_video,
            output_path=r"Yolov11-License_Plate_Recognition\outputs\output_video.mp4",
            max_frames=None, 
            skip_frames=2 
        )
        
        if results:
            print(f"\nDetection Summary:")
            print(f"Total vehicles with license plates: {len(results)}")
            for track_id, data in results.items():
                print(f"  Vehicle {track_id}: Confidence {data['confidence']:.2f}")
    else:
        print(f"\nTest video not found: {test_video}")
        print("Testing with webcam...")
        
        # Try webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Process every 5th frame
            if frame_num % 5 == 0:
                frame_results = pipeline.process_frame(frame, frame_num)
                annotated = pipeline.draw_results(frame, frame_results)
                
                # Try to display, but don't crash if not possible
                try:
                    cv2.imshow('License Plate Detection', annotated)
                except:
                    print("Display not available in this environment")
                    # Save frame instead
                    if frame_num % 50 == 0:
                        save_path = pipeline.base_output_dir / f"webcam_{frame_num}.jpg"
                        cv2.imwrite(str(save_path), annotated)
                        print(f"Saved frame to {save_path}")
            
            # Check for exit key
            try:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                break
        
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    main()