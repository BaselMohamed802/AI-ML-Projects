"""
Filename: Licensence_Plate_Detector_OCR.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/19/2025

Description:
    This file is the main pipeline the license plate detection and Text extraction using OCR.
    The pipeline is as follows:
        1- Load the YOLOv11 model for license plate detection.
        2- Read the input image or video stream.
        3- Perform license plate detection using the YOLOv11 model.
        4- Extract the detected license plate regions.
        5- Apply OCR to extract text from the license plate regions.
        6- Display or save the results with detected license plates and extracted text.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import necessary libraries
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import string
import easyocr

# Load the SORT object tracker module
from sort.sort import *

# Mapping dictionaries for character conversion (UK License Plates)
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


# ---------- Main Class Pipeline ---------- #
class LicensePlateDetectorOCR:
    def __init__(self, 
                 license_plate_model_path,
                 car_yolo_model_path="yolo11m.pt",
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

        # Load the YOLOv11 model for license plate detection
        self.license_plate_model = YOLO(license_plate_model_path)

        # Load the YOLOv11 Car model for car detection
        self.car_detection_model = YOLO(car_yolo_model_path)
        self.vehicle_classes = vehicle_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Initialize SORT Tracker
        self.tracker = Sort(max_age=max_age_sort_track, 
                            min_hits=min_hits_sort_track, 
                            iou_threshold=iou_threshold)
        print(f"SORT tracker initialized (max_age={max_age_sort_track}, min_hits={min_hits_sort_track}, iou_threshold={iou_threshold})")

        # Initialize OCR reader
        self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))

        # Create/Add Output directory
        self.base_output_dir = Path("Yolov11-License Plate Recognition/outputs")
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Store results
        self.results = []
        
        print("============= Pipeline initialized successfully =============")

    # ========== Car Detection & Tracking ========== #
    def detect_vehicles(self, frame):
        """Detect vehicles in frame"""
        results = self.car_detection_model(
            frame, 
            conf=self.conf_threshold,
            verbose=False
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
                
                # Filter out small detections (Usually false positives)
                width = x2 - x1
                height = y2 - y1
                if width < 50 or height < 50:
                    continue
                
                detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections) if detections else np.empty((0, 5))
    
    def track_vehicles(self, detections):
        """
        Track vehicles using SORT.

        Args:
            detections (numpy.ndarray): Array of vehicle detections with shape (N, 5) where N is the number of detections. Each row contains the bounding box coordinates (x1, y1, x2, y2) and confidence of the detection.

        Returns:
            numpy.ndarray: Array of tracked vehicles with shape (N, 5) where N is the number of tracked vehicles. Each row contains the bounding box coordinates (x1, y1, x2, y2) and track ID of the vehicle.
        """
        if len(detections) == 0:
            return np.empty((0, 5))
        tracked = self.tracker.update(detections)
        
        # Ensure all values are properly converted to integers for drawing
        if len(tracked) > 0:
            # Convert to appropriate types
            tracked_ints = []
            for obj in tracked:
                x1, y1, x2, y2, track_id = obj
                tracked_ints.append([int(x1), int(y1), int(x2), int(y2), int(track_id)])
            return np.array(tracked_ints)
        return tracked
    
    # ========== Specific Car Tracking for License Plate Detection ========== #
    def track_car_of_interest(self, license_plate, tracked_vehicles):
        """
        Find which tracked vehicle contains the license plate.

        Args:
            license_plate (list): The bounding box coordinates of the license plate [x1, y1, x2, y2, confidence].
            tracked_vehicles (numpy.ndarray): Array of tracked vehicles with shape (N, 5) where N is the number of tracked vehicles.

        Returns:
            list: The tracked vehicle [x1, y1, x2, y2, track_id] that contains the license plate, or None if not found.
        """
        if len(tracked_vehicles) == 0:
            return None
            
        x1_lp, y1_lp, x2_lp, y2_lp, _ = license_plate
        
        for vehicle in tracked_vehicles:
            xcar1, ycar1, xcar2, ycar2, track_id = vehicle
            
            # Check if license plate is inside vehicle bounding box
            if (x1_lp >= xcar1 and y1_lp >= ycar1 and 
                x2_lp <= xcar2 and y2_lp <= ycar2):
                return vehicle
        
        return None
    
    # ========== License Plate Detection & Reading OCR ========== #
    def detect_license_plates_in_roi(self, frame, roi_bbox):
        """
        Detect license plates in a region of interest (ROI).

        Args:
            frame (numpy.ndarray): Input frame.
            roi_bbox (tuple): Bounding box coordinates of the ROI (x1, y1, x2, y2).

        Returns:
            list: List of detected license plates with shape (N, 5) where N is the number of detections. Each row contains the bounding box coordinates (x1, y1, x2, y2) and confidence of the detection.
            numpy.ndarray: Thresholded license plate crop for OCR (if license plate detected), else None.
        """
        x1, y1, x2, y2 = map(int, roi_bbox)
        
        # Ensure ROI is valid
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return [], None
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return [], None
        
        # Detect license plates
        results = self.license_plate_model(
            roi,
            conf=self.conf_threshold,
            verbose=False
        )
        
        license_plates = []
        license_plate_crop_thresh = None
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Convert to original frame coordinates
                lp_coords = [
                    int(rx1 + x1), int(ry1 + y1),
                    int(rx2 + x1), int(ry2 + y1),
                    float(conf)
                ]
                license_plates.append(lp_coords)
                
                # Get the license plate crop for the first detection only
                if license_plate_crop_thresh is None:
                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, [rx1, ry1, rx2, ry2])
                    lp_crop = roi[lp_y1:lp_y2, lp_x1:lp_x2]
                    
                    if lp_crop.size > 0:
                        license_plate_crop_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
                        license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)[1]
        
        return license_plates, license_plate_crop_thresh
    
    def check_license_plate_compliance(self, text):
        """
        Check if the license plate text complies with the required format.

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """
        if len(text) != 7:
            return False

        # Remove any spaces from the text
        text = text.replace(' ', '')
        
        if len(text) != 7:
            return False

        # Check each character position
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
           (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
           (text[2] in '0123456789' or text[2] in dict_char_to_int.keys()) and \
           (text[3] in '0123456789' or text[3] in dict_char_to_int.keys()) and \
           (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
           (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
           (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
            return True
        else:
            return False
        
    def format_license(self, text):
        """
        Format the license plate text by converting characters using the mapping dictionaries.

        Args:
            text (str): License plate text.

        Returns:
            str: Formatted license plate text.
        """
        license_plate_ = ''
        mapping = {
            0: dict_int_to_char, 
            1: dict_int_to_char, 
            2: dict_char_to_int, 
            3: dict_char_to_int,
            4: dict_int_to_char, 
            5: dict_int_to_char, 
            6: dict_int_to_char
        }
        
        text = text.replace(' ', '')
        
        if len(text) != 7:
            return text  # Return original if not 7 characters
            
        for j in range(7):
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

        return license_plate_
    
    def read_license_plate_ocr(self, license_plate_crop):
        """
        Read license plate text from a cropped image using OCR.

        Args:
            license_plate_crop (numpy.ndarray): Cropped license plate image.

        Returns:
            tuple: (license_plate_text, confidence_score) or (None, None) if not found
        """
        if license_plate_crop is None or license_plate_crop.size == 0:
            return None, None
            
        detections = self.reader.readtext(license_plate_crop)
        
        for detection in detections:
            bbox, text, score = detection
            text = text.upper().replace(' ', '')
            
            if self.check_license_plate_compliance(text):
                formatted_text = self.format_license(text)
                return formatted_text, score
            
        return None, None
    
    # ========== Writing Results ========== #
    def save_results(self, results, save_path=None):
        """
        Save detection results to a text file.

        Args:
            results (list): List of result dictionaries.
            save_path (str, optional): Path to save results. Defaults to None.
        """
        if save_path is None:
            save_path = self.base_output_dir / "detection_results.txt"
        
        with open(save_path, 'w') as f:
            f.write("License Plate Detection Results\n")
            f.write("=" * 50 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"Detection {i+1}:\n")
                f.write(f"  Frame: {result['frame']}\n")
                f.write(f"  Vehicle ID: {result['vehicle_id']}\n")
                f.write(f"  Vehicle BBox: {result['vehicle_bbox']}\n")
                f.write(f"  License Plate BBox: {result['lp_bbox']}\n")
                f.write(f"  License Plate Text: {result['lp_text']}\n")
                f.write(f"  OCR Confidence: {result['ocr_confidence']:.3f}\n")
                f.write(f"  LP Detection Confidence: {result['lp_confidence']:.3f}\n")
                f.write("-" * 40 + "\n")
        
        print(f"Results saved to: {save_path}")

    # ========== Process Single Image ========== #
    def process_image(self, image_path, output_path=None, visualize=True):
        """
        Process a single image for license plate detection.

        Args:
            image_path (str): Path to input image.
            output_path (str, optional): Path to save output image. Defaults to None.
            visualize (bool, optional): Whether to display the result. Defaults to True.

        Returns:
            dict: Detection results.
        """
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        # Detect vehicles
        vehicle_detections = self.detect_vehicles(frame)
        
        # Track vehicles (for single image, tracking will assign IDs)
        tracked_vehicles = self.track_vehicles(vehicle_detections)
        
        # Process each vehicle
        results = []
        annotated_frame = frame.copy()
        
        for vehicle in tracked_vehicles:
            xcar1, ycar1, xcar2, ycar2, track_id = map(int, vehicle)
            
            # Detect license plates in vehicle ROI
            license_plates, lp_crop = self.detect_license_plates_in_roi(
                frame, 
                (xcar1, ycar1, xcar2, ycar2)
            )
            
            if license_plates:
                # Use the first detected license plate
                lp = license_plates[0]
                x1_lp, y1_lp, x2_lp, y2_lp, lp_conf = lp
                
                # Read license plate text
                lp_text, ocr_conf = self.read_license_plate_ocr(lp_crop)
                
                # Store result
                result = {
                    'frame': 0,
                    'vehicle_id': int(track_id),
                    'vehicle_bbox': (int(xcar1), int(ycar1), int(xcar2), int(ycar2)),
                    'lp_bbox': (int(x1_lp), int(y1_lp), int(x2_lp), int(y2_lp)),
                    'lp_text': lp_text if lp_text else "NOT_FOUND",
                    'ocr_confidence': ocr_conf if ocr_conf else 0.0,
                    'lp_confidence': float(lp_conf)
                }
                results.append(result)
                
                # Draw vehicle box (green)
                cv2.rectangle(annotated_frame, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"V{int(track_id)}", 
                           (xcar1, ycar1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                # Draw license plate box (red)
                cv2.rectangle(annotated_frame, (x1_lp, y1_lp), (x2_lp, y2_lp), (0, 0, 255), 2)
                
                # Add license plate text
                if lp_text:
                    text_label = f"{lp_text} ({ocr_conf:.2f})"
                    cv2.putText(annotated_frame, text_label,
                               (x1_lp, y1_lp - 10), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (0, 0, 255), 2)
        
        # Save or display results
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            print(f"Output image saved to: {output_path}")
        
        if visualize:
            try:
                cv2.imshow('License Plate Detection', annotated_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                print("Display not available. Results saved to file.")
        
        # Save results to text file
        self.save_results(results)
        
        return results

    # ========== Process Input Video ========== #
    def process_video(self, video_path, output_path=None, max_frames=None, skip_frames=0, visualize=False):
        """
        Process a video for license plate detection.

        Args:
            video_path (str): Path to input video.
            output_path (str, optional): Path to save output video. Defaults to None.
            max_frames (int, optional): Maximum number of frames to process. Defaults to None.
            skip_frames (int, optional): Number of frames to skip between processing. Defaults to 0.
            visualize (bool, optional): Whether to display processing in real-time. Defaults to False.

        Returns:
            list: List of detection results.
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None
        
        frame_count = 0
        processed_count = 0
        results = []
        
        print("Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if specified
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                continue
            
            # Limit frames if specified
            if max_frames and processed_count >= max_frames:
                break
            
            # Detect vehicles
            vehicle_detections = self.detect_vehicles(frame)
            
            # Track vehicles
            tracked_vehicles = self.track_vehicles(vehicle_detections)
            
            # Process each tracked vehicle
            annotated_frame = frame.copy()
            for vehicle in tracked_vehicles:
                # Convert all coordinates to integers
                xcar1, ycar1, xcar2, ycar2, track_id = map(int, vehicle)
                
                # Detect license plates in vehicle ROI
                license_plates, lp_crop = self.detect_license_plates_in_roi(
                    frame, 
                    (xcar1, ycar1, xcar2, ycar2)
                )
                
                if license_plates:
                    # Use the license plate with highest confidence
                    lp = max(license_plates, key=lambda x: x[4])
                    x1_lp, y1_lp, x2_lp, y2_lp, lp_conf = lp
                    
                    # Read license plate text
                    lp_text, ocr_conf = self.read_license_plate_ocr(lp_crop)
                    
                    # Store result
                    result = {
                        'frame': frame_count,
                        'vehicle_id': int(track_id),
                        'vehicle_bbox': (int(xcar1), int(ycar1), int(xcar2), int(ycar2)),
                        'lp_bbox': (int(x1_lp), int(y1_lp), int(x2_lp), int(y2_lp)),
                        'lp_text': lp_text if lp_text else "NOT_FOUND",
                        'ocr_confidence': ocr_conf if ocr_conf else 0.0,
                        'lp_confidence': float(lp_conf)
                    }
                    results.append(result)
                    
                    # Draw on frame
                    # Draw vehicle box (green)
                    cv2.rectangle(annotated_frame, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"V{int(track_id)}", 
                               (xcar1, ycar1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 2)
                    
                    # Draw license plate box (red)
                    cv2.rectangle(annotated_frame, (x1_lp, y1_lp), (x2_lp, y2_lp), (0, 0, 255), 2)
                    
                    # Add license plate text
                    if lp_text:
                        text_label = f"{lp_text} ({ocr_conf:.2f})"
                        cv2.putText(annotated_frame, text_label,
                                   (x1_lp, y1_lp - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, (0, 0, 255), 2)
            
            # Write to output video
            if out:
                out.write(annotated_frame)
            
            # Display if requested
            if visualize:
                try:
                    cv2.imshow('License Plate Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing stopped by user")
                        break
                except:
                    visualize = False  # Disable visualization if not available
            
            processed_count += 1
            
            # Progress update
            if processed_count % 30 == 0:
                print(f"Processed {processed_count} frames...")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        if visualize:
            cv2.destroyAllWindows()
        
        print(f"Video processing complete! Processed {processed_count} frames.")
        
        # Save results
        self.save_results(results)
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    detector = LicensePlateDetectorOCR(
        license_plate_model_path=r"G:\Work Projects\AI & ML Projects\AI-ML-Projects\Yolov11-License Plate Recognition\Yolov11-License-Plate-Model\train5\weights\best.pt",  # Your trained license plate model
        car_yolo_model_path=r"G:\Work Projects\AI & ML Projects\AI-ML-Projects\Basic YOLOv11 Models\Object Detection\yolo11m.pt",    # Vehicle detection model
        conf_threshold=0.7,
        iou_threshold=0.5,
        max_age_sort_track=15,
        min_hits_sort_track=5,
        device='cuda'
    )
    
    # Process a single image
    # results = detector.process_image("test_image.jpg", "output_image.jpg")
    
    # Process a video
    results = detector.process_video(
        video_path=r"G:\Work Projects\AI & ML Projects\AI-ML-Projects\Car Recognition & Classification\Model Test Input\Cars Video 1.mp4",
        output_path="output_video.mp4",
        max_frames=None,
        skip_frames=1,
        visualize=False 
    )