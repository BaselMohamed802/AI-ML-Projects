"""
Filename: Main_pipeline_script.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/17/2025

Description:
    This script is the main pipeline for car recognition and classification project.
    The pipeline works as follows:
        1. User chooses to upload a video or an image.
        2. The cars are detected using a YOLOv11 model and the detections (bounding boxes) are extracted.
        3. SORT tracker assigns IDs to each car and tracks them across frames.
        4. Classification for car make, model and manufacturing year occurs on each unique car ID.
        5. Output is saved and documented extensively using JSON.
"""

import torch
import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

# Import Finished Classification model
from ClassificationInference import CarClassifierInference

# Import Object Tracker
from sort.sort import *

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        """
        Handle numpy types during JSON encoding

        Args:
            o: object to be encoded

        Returns:
            Encoded object
        """
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)

# Define The main car recognition and classification pipeline
class CarRecognitionPipeline:
    def __init__(self, 
                 detection_model_path,
                 classification_model_path = None,
                 conf_threshold: float = 0.5,
                 max_age: int = 30,  # SORT parameter: max frames to keep lost track
                 min_hits: int = 3,  # SORT parameter: min detections before track is confirmed
                 iou_threshold: float = 0.3):  # SORT parameter: IOU threshold for matching
        
        """
        Initialize the Car Recognition Pipeline.

        Args:
            detection_model_path (Path): Path to the YOLO model.
            classification_model_path (Optional[Path], optional): Path to the classification model. Defaults to None.
            conf_threshold (float, optional): Confidence threshold for detection. Defaults to 0.5.
            max_age (int): Maximum number of frames to keep a track alive without detection
            min_hits (int): Minimum number of detections before a track is confirmed
            iou_threshold (float): IOU threshold for association

        Raises:
            FileNotFoundError: If the YOLO model is not found at the specified path.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize SORT Tracker
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        print(f"SORT tracker initialized (max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold})")
        
        # Validate and load detection model
        detection_model_path = Path(detection_model_path)
        if not detection_model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at: {detection_model_path}")
        
        print(f"Loading YOLO model from: {detection_model_path}")
        self.detector = YOLO(str(detection_model_path))
        self.detector.to(self.device)
        self.conf_threshold = conf_threshold
        
        # Vehicle classes
        self.vehicle_classes = {1: 'person', 2: "car", 5: "bus", 7: "truck", 3: "motorcycle"}
        
        # Load classification model
        self.classifier = None
        if classification_model_path:
            classification_model_path = Path(classification_model_path)
            if classification_model_path.exists():
                print(f"Loading classification model from: {classification_model_path}")
                try:
                    self.classifier = CarClassifierInference(str(classification_model_path), device=self.device)
                    print("Classification model loaded")
                except Exception as e:
                    print(f"Warning: Failed to load classification model: {e}")
                    self.classifier = None
            else:
                print(f"Warning: Classification model not found at {classification_model_path}")
        
        # Track classified cars: {track_id: classification_data}
        self.classified_cars = {}
        
        # Create/Add Output directory
        self.base_output_dir = Path("Car Recognition & Classification/outputs")
        self.base_output_dir.mkdir(exist_ok=True)
        
        print("Pipeline initialized successfully")
    
    def process_image(self, 
                     image_path: str,
                     classify_cars: bool = True,
                     visualize: bool = True) -> Dict:

        """
        Process a single image and detect vehicles, classify cars, and visualize results
        
        Args:
            image_path (str): Path to image
            classify_cars (bool): Whether to classify detected cars (default: True)
            visualize (bool): Whether to visualize results (default: True)
        
        Returns:
            Dict: Output results in a JSON serializable format
        """

        # Setup output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_output_dir / f"image_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read image
        original_image = cv2.imread(image_path)
        if original_image is None:
            return {"error": "Failed to load image"}
        
        # Detect vehicles
        detections = self._detect_vehicles(original_image)
        print(f"Detected {len(detections)} vehicles")
        
        # Prepare results
        results = {
            'image_path': image_path,
            'detections': self._clean_detections(detections),
            'classifications': []
        }
        
        # Classify each car
        if classify_cars and self.classifier and detections:
            for i, detection in enumerate(detections):
                if detection['class_name'] != 'car':
                    continue
                
                # Crop car
                car_crop = self._crop_car(original_image, detection['bbox'])
                if car_crop is None:
                    continue
                
                # Save temp crop
                temp_path = output_dir / f"temp_car_{i}.jpg"
                cv2.imwrite(str(temp_path), car_crop)
                
                try:
                    # Run classification
                    cls_result = self.classifier.predict_single_image(str(temp_path))
                    
                    if cls_result.get('success', False):
                        # Clean classification result
                        clean_cls = {
                            'make': str(cls_result['make']['name']),
                            'model': str(cls_result['model']['name']),
                            'year': int(cls_result['year']['id']),
                            'make_id': int(cls_result['make']['id']),
                            'model_id': int(cls_result['model']['id'])
                        }
                        
                        # Update detection
                        detection.update({
                            'make': clean_cls['make'],
                            'model': clean_cls['model'],
                            'year': clean_cls['year']
                        })
                        
                        results['classifications'].append(clean_cls)
                        print(f"Car {i+1}: {clean_cls['make']} {clean_cls['model']} ({clean_cls['year']})")
                except Exception as e:
                    print(f"Error classifying car {i}: {e}")
                finally:
                    if temp_path.exists():
                        temp_path.unlink()
        
        # Clean detections again after updates
        results['detections'] = self._clean_detections(detections)
        
        # Create annotated image
        if visualize and detections:
            annotated = self._draw_boxes(original_image, detections)
            output_path = output_dir / f"result_{Path(image_path).name}"
            cv2.imwrite(str(output_path), annotated)
            results['output_image'] = str(output_path)
        
        # Save results
        results_path = output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        results['output_dir'] = str(output_dir)
        return results
    
    def process_video(self,
                     video_path: str,
                     classify_cars: bool = True,
                     process_every_n: int = 1) -> Dict:
        """
        Process video file with SORT tracking and classification
        
        Args:
            video_path (str): Path to video file
            classify_cars (bool): Whether to classify detected cars
            process_every_n (int): Process every N frames (1 = every frame)
        
        Returns:
            Dict: Processing results and statistics
        """
        # Setup output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        output_dir = self.base_output_dir / f"video_{video_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset tracker and classified cars for new video
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.classified_cars = {}
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Failed to open video"}
        
        # Get video info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video
        output_path = output_dir / f"processed_{Path(video_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"Processing video: {total_frames} frames at {fps:.1f} fps")
        print(f"Processing every {process_every_n} frames")
        print(f"SORT tracker active with {len(self.classified_cars)} classified cars")
        
        frame_count = 0
        processed = 0
        total_detections = 0
        total_classifications = 0
        start_time = time.time()
        
        # Video statistics
        frame_stats = []
        
        print("\nStarting video processing with SORT tracking...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if needed
            if frame_count % process_every_n != 0:
                # Still write frame to maintain timing
                out.write(frame)
                continue
            
            processed += 1
            
            # Detect vehicles in frame
            raw_detections = self._detect_vehicles(frame)
            total_detections += len(raw_detections)
            
            # Prepare detections for SORT (format: [x1, y1, x2, y2, score])
            detections_for_sort = []
            for det in raw_detections:
                if det['class_name'] == 'car':  # Only track cars
                    x1, y1, x2, y2 = det['bbox']
                    score = det['confidence']
                    detections_for_sort.append([x1, y1, x2, y2, score])
            
            # Convert to numpy array for SORT
            if detections_for_sort:
                detections_array = np.array(detections_for_sort)
            else:
                detections_array = np.empty((0, 5))
            
            # Update SORT tracker
            tracked_objects = self.tracker.update(detections_array)
            
            # Process tracked objects
            frame_detections = []
            new_classifications = 0
            
            for obj in tracked_objects:
                # SORT returns: [x1, y1, x2, y2, track_id]
                x1, y1, x2, y2, track_id = obj
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                track_id = int(track_id)
                
                # Create detection dict
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'track_id': track_id,
                    'class_name': 'car'
                }
                
                # Check if this track_id has been classified
                if track_id in self.classified_cars:
                    # Use existing classification
                    cls_data = self.classified_cars[track_id]
                    detection.update({
                        'make': cls_data['make'],
                        'model': cls_data['model'],
                        'year': cls_data['year']
                    })
                elif classify_cars and self.classifier:
                    # New track - classify it
                    # Check if car is reasonably large for classification
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width > 60 and height > 60:  # Minimum size for classification
                        car_crop = self._crop_car(frame, [x1, y1, x2, y2])
                        if car_crop is not None and car_crop.size > 0:
                            # Save temp crop
                            temp_path = output_dir / f"frame_{frame_count}_track_{track_id}.jpg"
                            cv2.imwrite(str(temp_path), car_crop)
                            
                            try:
                                # Run classification
                                cls_result = self.classifier.predict_single_image(str(temp_path))
                                
                                if cls_result.get('success', False):
                                    # Store classification
                                    cls_data = {
                                        'make': str(cls_result['make']['name']),
                                        'model': str(cls_result['model']['name']),
                                        'year': int(cls_result['year']['id']),
                                        'track_id': track_id,
                                        'frame_classified': frame_count,
                                        'confidence': {
                                            'make': float(cls_result['make']['confidence']),
                                            'model': float(cls_result['model']['confidence']),
                                            'year': float(cls_result['year']['confidence'])
                                        }
                                    }
                                    
                                    self.classified_cars[track_id] = cls_data
                                    total_classifications += 1
                                    new_classifications += 1
                                    
                                    # Update detection
                                    detection.update({
                                        'make': cls_data['make'],
                                        'model': cls_data['model'],
                                        'year': cls_data['year']
                                    })
                                    
                                    print(f"Frame {frame_count}: Track {track_id} classified as {cls_data['make']} {cls_data['model']} ({cls_data['year']})")
                            except Exception as e:
                                print(f"Error classifying track {track_id}: {e}")
                            finally:
                                if temp_path.exists():
                                    temp_path.unlink()
                
                frame_detections.append(detection)
            
            # Store frame stats
            frame_stats.append({
                'frame': frame_count,
                'detections': len(raw_detections),
                'tracked_objects': len(tracked_objects),
                'new_classifications': new_classifications,
                'total_classified': len(self.classified_cars)
            })
            
            # Draw tracked objects on frame
            annotated = self._draw_tracked_objects(frame, frame_detections)
            
            # Add overlay
            self._add_video_overlay(annotated, frame_count, frame_detections, len(self.classified_cars))
            
            out.write(annotated)
            
            # Show progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_count}/{total_frames} ({fps_current:.1f} fps) - Tracks: {len(tracked_objects)}, Classified: {len(self.classified_cars)}")
        
        # Clean up
        cap.release()
        out.release()
        
        # Calculate stats
        process_time = time.time() - start_time
        
        # Get unique car models
        unique_cars = []
        for track_id, car_data in self.classified_cars.items():
            car_id = f"{car_data['make']}_{car_data['model']}_{car_data['year']}"
            if car_id not in unique_cars:
                unique_cars.append(car_id)
        
        # Prepare results
        results = {
            'input_video': video_path,
            'output_video': str(output_path),
            'frames_total': int(total_frames),
            'frames_processed': int(processed),
            'total_detections': int(total_detections),
            'total_classifications': int(total_classifications),
            'unique_cars_detected': int(len(unique_cars)),
            'total_tracks': max(self.classified_cars.keys()) if self.classified_cars else 0,
            'processing_time': float(process_time),
            'fps_processed': float(processed / process_time) if process_time > 0 else 0.0,
            'output_dir': str(output_dir)
        }
        
        # Save video stats
        stats_path = output_dir / "video_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'summary': results,
                'frame_stats': frame_stats,
                'classified_cars': list(self.classified_cars.values())
            }, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n" + "="*50)
        print(f"VIDEO PROCESSING COMPLETE!")
        print(f"="*50)
        print(f"Total frames: {frame_count}/{total_frames}")
        print(f"Total detections: {total_detections}")
        print(f"Total classifications: {total_classifications}")
        print(f"Unique cars: {len(unique_cars)}")
        print(f"Total tracks: {max(self.classified_cars.keys()) if self.classified_cars else 0}")
        print(f"Processing time: {process_time:.1f}s")
        print(f"Processing speed: {frame_count/process_time:.1f} fps")
        print(f"Output saved to: {output_path}")
        
        return results
    
    def _detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        """Detect vehicles in image"""
        results = self.detector(image, conf=self.conf_threshold, device=self.device)
        
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
                
                # Skip small boxes
                if (x2 - x1) < 40 or (y2 - y1) < 40:
                    continue
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'class_name': self.vehicle_classes[class_id]
                })
        
        return detections
    
    def _clean_detections(self, detections: List[Dict]) -> List[Dict]:
        """Convert detections to Python types"""
        cleaned = []
        for det in detections:
            cleaned.append({
                'bbox': [int(x) for x in det['bbox']],
                'confidence': float(det['confidence']),
                'class_id': int(det['class_id']),
                'class_name': str(det['class_name']),
                'make': str(det.get('make', '')) if 'make' in det else '',
                'model': str(det.get('model', '')) if 'model' in det else '',
                'year': int(det.get('year', 0)) if 'year' in det else 0
            })
        return cleaned
    
    def _crop_car(self, image: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Crop car from image"""
        x1, y1, x2, y2 = bbox
        padding = 10  # Increased padding for better classification
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return image[y1:y2, x1:x2]
    
    def _draw_boxes(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on image (for single image)"""
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            
            # Choose color
            if class_name == 'car':
                color = (0, 255, 0)  # Green for cars
            elif class_name == 'bus':
                color = (0, 165, 255)  # Orange for buses
            elif class_name == 'truck':
                color = (0, 0, 255)  # Red for trucks
            else:
                color = (255, 0, 0)  # Blue for motorcycles
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}"
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add classification if available
            if 'make' in det and 'model' in det:
                cls_label = f"{det['make']} {det['model']}"
                if 'year' in det and det['year']:
                    cls_label += f" ({det['year']})"
                # Draw on the box
                cv2.putText(result, cls_label, (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 3)
        
        return result
    
    def _draw_tracked_objects(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw tracked objects with IDs and classification"""
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            track_id = det['track_id']
            
            # Color based on classification status
            if 'make' in det:
                color = (0, 255, 0)  # Green - classified
            else:
                color = (0, 255, 255)  # Yellow - unclassified
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label with track ID
            label = f"ID: {track_id}"
            
            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(result, (x1, y1 - label_height - 10),
                         (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add classification if available
            if 'make' in det and 'model' in det:
                cls_label = f"{det['make']} {det['model']}"
                if 'year' in det and det['year']:
                    cls_label += f" ({det['year']})"
                
                # Draw classification below the box
                cv2.putText(result, cls_label, (x1, y2 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
    
    def _add_video_overlay(self, image: np.ndarray, frame_count: int, detections: List[Dict], total_classified: int):
        """Add overlay text to video frame"""
        # Count classified vs unclassified
        classified_count = len([d for d in detections if 'make' in d])
        unclassified_count = len([d for d in detections if 'make' not in d])
        
        # Frame counter
        cv2.putText(image, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Stats
        cv2.putText(image, f"Tracks: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(image, f"Classified: {classified_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(image, f"Unclassified: {unclassified_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        cv2.putText(image, f"Total: {total_classified}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    """Simple menu for testing"""
    
    # === CONFIGURE YOUR PATHS HERE ===
    YOLO_MODEL_PATH = r"G:\Work Projects\AI & ML Projects\AI-ML-Projects\Car Recognition & Classification\Models\yolo11m.pt"
    CLASSIFICATION_MODEL_PATH = r"G:\Work Projects\AI & ML Projects\AI-ML-Projects\Car Recognition & Classification\Models\complete_model.pth"
    
    # First, check if files exist
    print("=" * 60)
    print("Checking model files...")
    print("=" * 60)
    print(f"YOLO model exists: {os.path.exists(YOLO_MODEL_PATH)}")
    print(f"Classification model exists: {os.path.exists(CLASSIFICATION_MODEL_PATH)}")
    
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"\n✗ ERROR: YOLO model not found at: {YOLO_MODEL_PATH}")
        print("Please check the path and try again.")
        return
    
    # Initialize pipeline
    try:
        pipeline = CarRecognitionPipeline(
            detection_model_path=YOLO_MODEL_PATH,
            classification_model_path=CLASSIFICATION_MODEL_PATH,
            conf_threshold=0.5,
            max_age=30,      # Keep tracks alive for 30 frames without detection
            min_hits=3,      # Require 3 detections before confirming track
            iou_threshold=0.3  # IOU threshold for matching
        )
    except Exception as e:
        print(f"✗ Error initializing pipeline: {e}")
        return
    
    print("\n" + "=" * 60)
    print("SORT-TRACKED Car Recognition Pipeline")
    print("=" * 60)
    print("- SORT tracker for stable object tracking")
    print("- Each car gets a unique track ID")
    print("- Classification happens once per track ID")
    print("- Persistent bounding boxes with track IDs")
    print(f"- Classification enabled: {pipeline.classifier is not None}")
    
    while True:
        print("\n" + "=" * 40)
        print("OPTIONS:")
        print("=" * 40)
        print("1. Process image")
        print("2. Process video with SORT tracking")
        print("3. Test classification on single image")
        print("4. Exit")
        
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == "1":
            img_path = input("Image path: ").strip()
            if os.path.exists(img_path):
                print("\nProcessing image...")
                result = pipeline.process_image(img_path, classify_cars=True)
                if 'output_dir' in result:
                    print(f"\n✓ Results saved to: {result['output_dir']}")
                    if 'classifications' in result and result['classifications']:
                        print(f"✓ Classified {len(result['classifications'])} cars")
                if 'error' in result:
                    print(f"\n✗ Error: {result['error']}")
            else:
                print("✗ File not found")
        
        elif choice == "2":
            vid_path = input("Video path: ").strip()
            if os.path.exists(vid_path):
                skip = input("Process every N frames (default 1 = every frame): ").strip()
                skip = int(skip) if skip.isdigit() and int(skip) > 0 else 1
                
                print(f"\n" + "="*50)
                print(f"Starting SORT-tracked video processing...")
                print(f"="*50)
                print(f"- Processing every {skip} frames")
                print("- SORT tracker assigns unique IDs to each car")
                print("- Each car classified once (on first appearance)")
                print("- Green boxes = classified cars with make/model/year")
                print("- Yellow boxes = unclassified cars (tracked but not classified)")
                print(f"- Classifier available: {pipeline.classifier is not None}")
                
                result = pipeline.process_video(vid_path, 
                                              classify_cars=True,
                                              process_every_n=skip)
                
                if 'output_video' in result:
                    print(f"\n✓ Video saved to: {result['output_video']}")
                    print(f"✓ Processing speed: {result.get('fps_processed', 0):.1f} fps")
                    print(f"✓ Total classifications: {result.get('total_classifications', 0)}")
                    print(f"✓ Total tracks: {result.get('total_tracks', 0)}")
                if 'error' in result:
                    print(f"\n✗ Error: {result['error']}")
            else:
                print("✗ File not found")
        
        elif choice == "3":
            # Test classification on single image
            if pipeline.classifier:
                test_img = input("Test image path: ").strip()
                if os.path.exists(test_img):
                    print(f"\nTesting classification on: {test_img}")
                    result = pipeline.classifier.predict_single_image(test_img)
                    print(f"Result: {result}")
                    if result.get('success', False):
                        print(f"✓ Success! Car is: {result['make']['name']} {result['model']['name']} ({result['year']['id']})")
                        print(f"  Confidence: Make={result['make']['confidence']:.3f}, Model={result['model']['confidence']:.3f}, Year={result['year']['confidence']:.3f}")
                    else:
                        print(f"✗ Failed: {result.get('error', 'Unknown error')}")
                else:
                    print("✗ File not found")
            else:
                print("✗ No classifier available")
        
        elif choice == "4":
            print("\nGoodbye!")
            break
        
        else:
            print("✗ Invalid choice")


if __name__ == "__main__":
    main()