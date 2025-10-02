import cv2
import numpy as np
import argparse
import os

def load_calibration(calib_file):
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")
    with np.load(calib_file) as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs'].ravel()  # Убедимся, что 1D
    print("✅ Calibration loaded.")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coeffs: {dist_coeffs}")
    return camera_matrix, dist_coeffs

def detect_aruco_with_pose(image, camera_matrix, dist_coeffs, aruco_dict_type, marker_size):
    """
    Detect markers on RAW image, estimate pose using calibration.
    DO NOT undistort the image before detection!
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # Detect on RAW image
    corners, ids, rejected = detector.detectMarkers(gray)

    output = image.copy()

    if ids is not None:
        # Estimate pose using calibration (this handles distortion internally)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeffs
        )

        for i in range(len(ids)):
            # Draw 3D axis (uses calibration to project correctly)
            cv2.drawFrameAxes(output, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size)

            # Display ID and distance
            c = corners[i][0]
            top_left = tuple(c[0].astype(int))
            dist = np.linalg.norm(tvecs[i][0])
            cv2.putText(output, f"ID:{ids[i][0]} {dist:.2f}m", top_left,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"Detected markers: {ids.flatten().tolist()}")
    else:
        print("No markers detected.")

    return output

def process_image(image_path, cam_mat, dist_coeff, dict_type, marker_size):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: cannot load {image_path}")
        return
    result = detect_aruco_with_pose(img, cam_mat, dist_coeff, dict_type, marker_size)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(src, cam_mat, dist_coeff, dict_type, marker_size):
    try:
        cap = cv2.VideoCapture(int(src))
    except:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"Cannot open source: {src}")
        return

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = detect_aruco_with_pose(frame, cam_mat, dist_coeff, dict_type, marker_size)
        cv2.imshow("ArUco + Calibration", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", default="camera_calibration.npz")
    parser.add_argument("--image", help="Input image")
    parser.add_argument("--video", default="1")
    parser.add_argument("--marker_size", type=float, default=0.104)
    parser.add_argument("--dict", default="DICT_5X5_250")

    args = parser.parse_args()

    # Load calibration
    try:
        cam_mat, dist_coeff = load_calibration(args.calibration)
    except Exception as e:
        print(f"Calibration error: {e}")
        exit(1)

    # Get dict
    if not hasattr(cv2.aruco, args.dict):
        print(f"Unknown dict: {args.dict}")
        exit(1)
    dict_type = getattr(cv2.aruco, args.dict)

    # Run
    if args.image:
        process_image(args.image, cam_mat, dist_coeff, dict_type, args.marker_size)
    else:
        process_video(args.video, cam_mat, dist_coeff, dict_type, args.marker_size)