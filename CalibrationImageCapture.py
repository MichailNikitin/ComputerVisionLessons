import cv2
import numpy as np
import os

def main():
    # === Settings ===
    CHESSBOARD_SIZE = (6, 4)        # (width, height) in inner corners
    SQUARE_SIZE = 0.035             # square size in meters (e.g., 25 mm)
    MIN_CAPTURES = 10               # minimum frames needed for calibration
    CAMERA_INDEX = 1                # default camera
    OUTPUT_DIR = "calibration_frames"  # folder to save captured frames

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare object points (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane
    frame_count = 0

    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return

    print("Controls:\n"
          "  'c' — capture frame (only if chessboard is detected)\n"
          "  'q' — quit and calibrate\n"
          f"Captured: 0 / {MIN_CAPTURES}\n"
          f"Frames will be saved to: '{OUTPUT_DIR}/'")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # Find chessboard corners
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if found:
            # Refine corner positions
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners_sub, found)
            cv2.putText(display, "Chessboard detected! Press 'c' to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Chessboard not found",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show capture count
        cv2.putText(display, f"Captured: {len(objpoints)} / {MIN_CAPTURES}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Real-Time Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and found:
            # Save frame to disk
            frame_filename = os.path.join(OUTPUT_DIR, f"frame_{frame_count:03d}.jpg")
            cv2.imwrite(frame_filename, frame)

            # Store points
            objpoints.append(objp.copy())
            imgpoints.append(corners_sub)
            frame_count += 1

            print(f"✅ Captured and saved: {frame_filename} (Total: {len(objpoints)})")

            if len(objpoints) >= MIN_CAPTURES:
                print("Enough frames collected for calibration!")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # === Calibration ===
    if len(objpoints) < MIN_CAPTURES:
        print(f"Insufficient frames ({len(objpoints)} < {MIN_CAPTURES}). Calibration aborted.")
        return

    print("Performing camera calibration...")
    h, w = gray.shape[:2]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    # Compute reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)

    print(f"\n✅ Calibration completed!")
    print(f"Mean reprojection error: {mean_error:.4f} pixels")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs.ravel()}")

    # Save calibration
    np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"\n💾 Calibration saved to 'camera_calibration.npz'")
    print(f"📸 Calibration frames saved in folder: '{OUTPUT_DIR}/'")

    # === Demo: Before vs After (using first saved frame) ===
    demo_path = os.path.join(OUTPUT_DIR, "frame_000.jpg")
    if os.path.exists(demo_path):
        test_img = cv2.imread(demo_path)
        h, w = test_img.shape[:2]

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, newcameramtx)

        combined = np.hstack((test_img, undistorted))
        cv2.imshow("Comparison: Original | Undistorted", combined)
        print("\nPress any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()