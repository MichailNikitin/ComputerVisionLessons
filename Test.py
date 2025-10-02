import cv2
import numpy as np
import argparse
import os
import sys

def load_calibration(calib_file):
    """Load camera matrix and distortion coefficients."""
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")
    with np.load(calib_file) as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
    print("✅ Calibration loaded.")
    return camera_matrix, dist_coeffs

def undistort_and_show(image, camera_matrix, dist_coeffs):
    """
    Correct lens distortion and return side-by-side view: [original | undistorted]
    """
    h, w = image.shape[:2]

    # Compute optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # Undistort
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Optional: crop to ROI (uncomment if you want tight crop)
    # x, y, w_roi, h_roi = roi
    # undistorted = undistorted[y:y+h_roi, x:x+w_roi]
    # Resize back to original size for comparison (optional)
    # undistorted = cv2.resize(undistorted, (w, h))

    # Combine original and undistorted side by side
    combined = np.hstack((image, undistorted))
    return combined

def process_image(image_path, camera_matrix, dist_coeffs):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return

    result = undistort_and_show(image, camera_matrix, dist_coeffs)
    cv2.imshow("Calibration Result: Original | Undistorted", result)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_source, camera_matrix, dist_coeffs):
    try:
        src = int(video_source)
        cap = cv2.VideoCapture(src)
    except ValueError:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or camera disconnected.")
            break

        result = undistort_and_show(frame, camera_matrix, dist_coeffs)
        cv2.imshow("Calibration Result: Original | Undistorted", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize camera calibration result")
    parser.add_argument("--calibration", type=str, default="camera_calibration.npz",
                        help="Path to calibration .npz file (default: camera_calibration.npz)")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, default="0",
                        help="Path to video file or camera index (default: 0)")

    args = parser.parse_args()

    # Load calibration
    try:
        camera_matrix, dist_coeffs = load_calibration(args.calibration)
    except Exception as e:
        print(f"Error loading calibration: {e}")
        sys.exit(1)

    # Run
    if args.image:
        process_image(args.image, camera_matrix, dist_coeffs)
    else:
        process_video(args.video, camera_matrix, dist_coeffs)