import cv2
import numpy as np
import yaml
import os
from typing import Dict, Any
from scipy.spatial.transform import Rotation as R


class ArucoDetector:
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.marker_length = float(self.config.get('marker_length', 0.05))
        aruco_dict_name = self.config.get('aruco_dict', 'DICT_5X5_50')
        self.camera_id = int(self.config.get('camera_id', 0))

        if not hasattr(cv2.aruco, aruco_dict_name):
            raise ValueError(f"Неизвестный словарь ArUco: {aruco_dict_name}")
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))

        self.detector_params = cv2.aruco.DetectorParameters()
        det_params_config = self.config.get('detector_params', {})
        for param_name, value in det_params_config.items():
            if hasattr(self.detector_params, param_name):
                setattr(self.detector_params, param_name, value)
            else:
                print(f"⚠️ Неизвестный параметр детектора: '{param_name}'")

        # 🔍 Автоматический путь к калибровке в той же папке, что и config.yaml
        config_dir = os.path.dirname(os.path.abspath(config_path))
        calibration_file = os.path.join(config_dir, "camera_calibration_good.npz")

        if not os.path.exists(calibration_file):
            raise FileNotFoundError(f"Файл калибровки не найден: {calibration_file}")

        with np.load(calibration_file) as data:  # ← ВАЖНО: 'data' после 'as'
            self.camera_matrix = data['camera_matrix'].astype(np.float32)
            self.dist_coeffs = data['dist_coeffs'].astype(np.float32)
            if self.dist_coeffs.ndim == 2:
                self.dist_coeffs = self.dist_coeffs.flatten()

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

        half = self.marker_length / 2
        self.marker_obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

        self.last_pose: Dict[int, Dict[str, Any]] = {}

    def detect(self, image: np.ndarray):
        if image is None or image.size == 0:
            return [], None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def estimate_pose(self, corners, ids):
        if ids is None or len(ids) == 0 or len(corners) != len(ids):
            return [], []

        rvecs, tvecs = [], []
        for corner in corners:
            img_points = corner.reshape((4, 2))
            success, rvec, tvec = cv2.solvePnP(
                self.marker_obj_points,
                img_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
            else:
                rvecs.append(np.zeros((3, 1), dtype=np.float32))
                tvecs.append(np.zeros((3, 1), dtype=np.float32))
        return rvecs, tvecs

    def get_pos(self, image: np.ndarray) -> Dict[int, Dict[str, Any]]:
        corners, ids = self.detect(image)
        rvecs, tvecs = self.estimate_pose(corners, ids)

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                marker_id = int(mid)
                rvec = rvecs[i].copy()
                tvec = tvecs[i].copy()
                rmat, _ = cv2.Rodrigues(rvec)
                rot = R.from_matrix(rmat)
                euler = rot.as_euler('xyz', degrees=True)

                self.last_pose[marker_id] = {
                    'position': tvec.flatten(),
                    'distance': float(np.linalg.norm(tvec)),
                    'euler_angles': euler.tolist(),
                    'rotation_vector': rvec.flatten().tolist(),
                    'valid': True
                }

        detected_ids = set(ids.flatten()) if ids is not None else set()
        for mid in self.last_pose:
            if mid not in detected_ids:
                self.last_pose[mid]['valid'] = False

        return self.last_pose.copy()

    def draw(self, image: np.ndarray) -> np.ndarray:
        corners, ids = self.detect(image)
        rvecs, tvecs = self.estimate_pose(corners, ids)
        output = image.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(output, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i].reshape((4, 2))
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))

                pose = self.last_pose.get(int(marker_id))
                if not pose:
                    continue

                dist = pose['distance']
                euler = pose['euler_angles']
                label = f"ID{marker_id}\nD={dist:.2f}m\nR={euler[0]:.0f}° P={euler[1]:.0f}° Y={euler[2]:.0f}°"
                y0 = cy - 60
                for j, line in enumerate(label.split('\n')):
                    cv2.putText(output, line, (cx - 30, y0 + j * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return output


# =============================
# ОСНОВНАЯ ПРОГРАММА
# =============================
if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\student\Desktop\artHzBan\config.yaml"

    try:
        detector = ArucoDetector(CONFIG_PATH)
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        exit(1)

    print(f"📷 Подключаюсь к камере {detector.camera_id}...")
    cap = cv2.VideoCapture(detector.camera_id)

    if not cap.isOpened():
        print(f"❌ Не удалось открыть камеру {detector.camera_id}")
        exit(1)

    print("✅ Запущено. Нажмите ESC для выхода.")
    frame_count = 0
    UPDATE_INTERVAL = 10  # обновлять консоль каждые 10 кадров

    # Для отслеживания изменений (опционально)
    last_known_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Ошибка чтения кадра")
            break

        all_poses = detector.get_pos(frame)
        output_frame = detector.draw(frame)

        # --- Обновляем консоль раз в N кадров ---
        frame_count += 1
        if frame_count % UPDATE_INTERVAL == 0:
            current_ids = set(all_poses.keys())
            if current_ids != last_known_ids:
                # Выводим заголовок только при изменении набора ID
                print("\n" + "="*50)
                print("Обнаружены маркеры:" if current_ids else "Нет маркеров")
                last_known_ids = current_ids

            if all_poses:
                for mid, data in all_poses.items():
                    status = "Виден" if data['valid'] else "Потерян (кэш)"
                    pos = data['position']
                    e = data['euler_angles']
                    dist = data['distance']
                    print(f"\n[ID {mid}] {status}")
                    print(f"  Расстояние: {dist:.3f} м")
                    print(f"  Углы (Roll, Pitch, Yaw): [{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}]°")
                    print(f"  Позиция (X, Y, Z): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            else:
                print("Нет данных о маркерах")

        # --- Показ видео ---
        cv2.imshow("Aruco Tracker", output_frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n⏹️  Завершено")
