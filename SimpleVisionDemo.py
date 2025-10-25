import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time


class MultiTaskVisionSystem:
    def __init__(self):
        # Initialize SMALL models for different tasks
        self.models = {
            'detection': YOLO('yolo11s.pt'),  # Object detection
            'segmentation': YOLO('yolo11s-seg.pt'),  # Segmentation
            'classification': YOLO('yolo11s-cls.pt'),  # Classification
            'pose': YOLO('yolo11s-pose.pt'),  # Pose estimation
            'tracking': YOLO('yolo11s.pt'),  # Tracking
            'obb': YOLO('yolo11s-obb.pt')  # Oriented bounding boxes
        }

        # Configuration for different tasks
        self.config = {
            'conf_threshold': 0.3,
            'iou_threshold': 0.3,
            'classes_to_count': [0, 2, 5, 7]  # Persons, cars, buses, trucks
        }

        # Initialize tracker
        self.track_history = defaultdict(lambda: [])
        self.fps = 0
        self.prev_time = 0

    def setup_camera(self, camera_id=0):
        """Setup camera"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            # Try other camera indices
            for i in range(1, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"Camera found at index {i}")
                    break

            if not self.cap.isOpened():
                print("Camera not found, using test video...")
                self.cap = cv2.VideoCapture('test_video.mp4')
                if not self.cap.isOpened():
                    raise ValueError("Failed to open camera or video")

        # Set parameters for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def calculate_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return self.fps

    def draw_detection_results(self, frame, results):
        """Visualize detection results"""
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])

                    # Draw bounding box
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label with class and confidence
                    label = f"{result.names[cls]}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def draw_segmentation_results(self, frame, results):
        """Visualize segmentation results"""
        for result in results:
            if result.masks is not None:
                # Use built-in visualization for speed
                annotated_frame = result.plot()
                return annotated_frame

        return frame

    def draw_pose_results(self, frame, results):
        """Visualize pose estimation results"""
        for result in results:
            if result.keypoints is not None:
                # Use built-in visualization for speed
                annotated_frame = result.plot()
                return annotated_frame

        return frame

    def draw_tracking_results(self, frame, results):
        """Visualize tracking results"""
        for result in results:
            if result.boxes is not None and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
                    x1, y1, x2, y2 = map(int, box)

                    # Draw bounding box with color by ID
                    color = self.get_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label with track ID
                    label = f"ID:{track_id} {result.names[cls]}:{conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Save track history
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    self.track_history[track_id].append(center)

                    # Draw track history (last 20 points)
                    if len(self.track_history[track_id]) > 20:
                        self.track_history[track_id].pop(0)

                    # Draw track line
                    if len(self.track_history[track_id]) > 1:
                        points = np.array(self.track_history[track_id], np.int32)
                        cv2.polylines(frame, [points], False, color, 2)

        return frame

    def draw_obb_results(self, frame, results):
        """Visualize oriented bounding box results"""
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                # Use built-in visualization for OBB
                annotated_frame = result.plot()
                return annotated_frame

        return frame

    def get_color(self, track_id):
        """Generate color based on track ID"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 0), (255, 128, 0), (0, 128, 255)
        ]
        return colors[track_id % len(colors)]

    def run_detection(self, frame):
        """Run object detection"""
        results = self.models['detection'](
            frame,
            conf=self.config['conf_threshold'],
            iou=self.config['iou_threshold'],
            verbose=False,
            imgsz=320
        )
        return self.draw_detection_results(frame, results)

    def run_segmentation(self, frame):
        """Run segmentation"""
        results = self.models['segmentation'](
            frame,
            conf=self.config['conf_threshold'],
            verbose=False,
            imgsz=320
        )
        return self.draw_segmentation_results(frame, results)

    def run_pose_estimation(self, frame):
        """Run pose estimation"""
        results = self.models['pose'](
            frame,
            conf=self.config['conf_threshold'],
            verbose=False,
            imgsz=320
        )
        return self.draw_pose_results(frame, results)

    def run_tracking(self, frame):
        """Run object tracking"""
        results = self.models['tracking'].track(
            frame,
            conf=self.config['conf_threshold'],
            iou=self.config['iou_threshold'],
            persist=True,
            verbose=False,
            imgsz=320,
            tracker="bytetrack.yaml"  # Use ByteTrack for better tracking
        )
        return self.draw_tracking_results(frame, results)

    def run_classification(self, frame):
        """Run scene classification"""
        results = self.models['classification'](
            frame,
            verbose=False,
            imgsz=224
        )

        # Get top predictions
        if results and hasattr(results[0], 'probs'):
            probs = results[0].probs
            if probs is not None:
                # Get top classes and probabilities
                top5_indices = probs.top5
                top5_probs = probs.top5conf

                # Display classification results
                for i, (cls_idx, conf) in enumerate(zip(top5_indices, top5_probs)):
                    if i >= 3:  # Show only top-3
                        break
                    class_name = results[0].names[cls_idx]
                    text = f"{i + 1}. {class_name}: {conf:.2f}"
                    cv2.putText(frame, text, (300, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def run_obb(self, frame):
        """Run oriented bounding box detection"""
        results = self.models['obb'](
            frame,
            conf=self.config['conf_threshold'],
            iou=self.config['iou_threshold'],
            verbose=False,
            imgsz=320
        )
        return self.draw_obb_results(frame, results)

    def run(self):
        """Main processing loop"""
        print("Starting Computer Vision System with SMALL models...")
        print("Control keys:")
        print("1 - Object Detection")
        print("2 - Segmentation")
        print("3 - Classification")
        print("4 - Object Tracking")
        print("5 - Pose Estimation")
        print("6 - Oriented Bounding Boxes (OBB)")
        print("q - Quit")

        current_mode = 'detection'
        self.prev_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to get frame")
                break

            # Calculate FPS
            fps = self.calculate_fps()

            # Process frame based on current mode
            try:
                if current_mode == 'detection':
                    processed_frame = self.run_detection(frame.copy())
                    mode_text = "Mode: Object Detection"

                elif current_mode == 'segmentation':
                    processed_frame = self.run_segmentation(frame.copy())
                    mode_text = "Mode: Segmentation"

                elif current_mode == 'classification':
                    processed_frame = self.run_classification(frame.copy())
                    mode_text = "Mode: Classification"

                elif current_mode == 'tracking':
                    processed_frame = self.run_tracking(frame.copy())
                    mode_text = "Mode: Object Tracking"

                elif current_mode == 'pose':
                    processed_frame = self.run_pose_estimation(frame.copy())
                    mode_text = "Mode: Pose Estimation"

                elif current_mode == 'obb':
                    processed_frame = self.run_obb(frame.copy())
                    mode_text = "Mode: Oriented BBoxes (OBB)"

                # Display current mode and FPS
                cv2.putText(processed_frame, mode_text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, "Model: yolo11s", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv2.imshow('Multi-Task Vision System (SMALL)', processed_frame)

            except Exception as e:
                print(f"Processing error: {e}")
                cv2.imshow('Multi-Task Vision System (SMALL)', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_mode = 'detection'
                print("Switched to: Object Detection")
            elif key == ord('2'):
                current_mode = 'segmentation'
                print("Switched to: Segmentation")
            elif key == ord('3'):
                current_mode = 'classification'
                print("Switched to: Classification")
            elif key == ord('4'):
                current_mode = 'tracking'
                print("Switched to: Object Tracking")
            elif key == ord('5'):
                current_mode = 'pose'
                print("Switched to: Pose Estimation")
            elif key == ord('6'):
                current_mode = 'obb'
                print("Switched to: Oriented Bounding Boxes")

        self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("System stopped")





def main():
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Using SMALL models for better performance")

    try:
        # Create and run system
        vision_system = MultiTaskVisionSystem()
        vision_system.setup_camera(0)
        vision_system.run()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()