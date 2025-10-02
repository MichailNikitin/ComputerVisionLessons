import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt


class ArucoDetector:
    def __init__(self, camera_matrix=None, dist_coeffs=None, marker_length=0.05):
        """
        Инициализация детектора ArUco маркеров

        Args:
            camera_matrix: Матрица камеры (3x3)
            dist_coeffs: Коэффициенты дисторсии
            marker_length: Физический размер маркера в метрах
        """
        self.marker_length = marker_length

        # Параметры по умолчанию (замените на калиброванные значения)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        # Создание словаря ArUco 5x5
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)

        # Параметры детектора
        self.parameters = aruco.DetectorParameters()

    def calibrate_camera(self, images, chessboard_size=(9, 6), square_size=0.025):
        """
        Калибровка камеры по шахматной доске

        Args:
            images: Список изображений шахматной доски
            chessboard_size: Размер шахматной доски (количество углов)
            square_size: Размер квадрата в метрах
        """
        # Критерии для субпиксельной точности
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Подготовка объектных точек
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        # Массивы для хранения точек
        objpoints = []  # 3D точки в реальном мире
        imgpoints = []  # 2D точки на изображении

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Поиск углов шахматной доски
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                objpoints.append(objp)

                # Уточнение позиции углов
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners_refined)

        # Калибровка камеры
        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        return ret

    def detect_markers(self, image):
        """
        Обнаружение и идентификация ArUco маркеров

        Args:
            image: Входное изображение (BGR)

        Returns:
            corners: Углы обнаруженных маркеров
            ids: ID маркеров
            rejected: Отвергнутые кандидаты
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Детекция маркеров
        corners, ids, rejected = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters)

        return corners, ids, rejected

    def estimate_pose(self, corners, ids):
        """
        Оценка положения маркеров в реальных координатах

        Args:
            corners: Углы маркеров
            ids: ID маркеров

        Returns:
            rvecs: Векторы поворота
            tvecs: Векторы перемещения
        """
        if ids is None or len(ids) == 0:
            return [], []

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

        return rvecs, tvecs

    def draw_detection(self, image, corners, ids, rvecs=None, tvecs=None):
        """
        Визуализация результатов детекции

        Args:
            image: Исходное изображение
            corners: Углы маркеров
            ids: ID маркеров
            rvecs: Векторы поворота
            tvecs: Векторы перемещения

        Returns:
            image: Изображение с визуализацией
        """
        # Рисуем обнаруженные маркеры
        image_with_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

        # Рисуем оси координат для оценки позы
        if rvecs is not None and tvecs is not None:
            for i in range(len(ids)):
                cv2.drawFrameAxes(image_with_markers, self.camera_matrix,
                                  self.dist_coeffs, rvecs[i], tvecs[i],
                                  self.marker_length * 0.5)

        return image_with_markers

    def get_marker_pose_info(self, rvecs, tvecs, ids):
        """
        Получение информации о положении маркеров

        Args:
            rvecs: Векторы поворота
            tvecs: Векторы перемещения
            ids: ID маркеров

        Returns:
            pose_info: Словарь с информацией о позе
        """
        pose_info = {}

        if ids is not None:
            for i, marker_id in enumerate(ids):
                # Преобразование вектора поворота в матрицу
                rmat, _ = cv2.Rodrigues(rvecs[i])

                # Положение маркера относительно камеры
                position = tvecs[i][0]
                distance = np.linalg.norm(position)

                pose_info[int(marker_id)] = {
                    'position': position,
                    'rotation_matrix': rmat,
                    'distance': distance,
                    'rotation_vector': rvecs[i][0],
                    'translation_vector': tvecs[i][0]
                }

        return pose_info


def main():
    # Инициализация детектора
    detector = ArucoDetector(marker_length=0.05)

    # Если камера не откалибрована, используем приближенные параметры
    if detector.camera_matrix is None:
        # Приближенная матрица камеры (замените на калиброванные значения)
        detector.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)

        detector.dist_coeffs = np.zeros((4, 1))

    # Захват видео с камеры
    cap = cv2.VideoCapture(0)

    print("Нажмите 'q' для выхода, 's' для сохранения изображения")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обнаружение маркеров
        corners, ids, rejected = detector.detect_markers(frame)

        # Оценка позы
        rvecs, tvecs = [], []
        if ids is not None:
            rvecs, tvecs = detector.estimate_pose(corners, ids)

        # Визуализация
        result_image = detector.draw_detection(frame, corners, ids, rvecs, tvecs)

        # Отображение информации
        if ids is not None:
            pose_info = detector.get_marker_pose_info(rvecs, tvecs, ids)
            for marker_id, info in pose_info.items():
                print(f"Маркер {marker_id}: Позиция {info['position']}, Дистанция: {info['distance']:.2f} м")

        cv2.imshow('ArUco Detection', result_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Сохранение изображения
            cv2.imwrite(f'aruco_detection_{cv2.getTickCount()}.jpg', result_image)
            print("Изображение сохранено")

    cap.release()
    cv2.destroyAllWindows()


def demo_static_image():
    """Демонстрация работы на статическом изображении"""
    # Создание тестового изображения с ArUco маркерами
    detector = ArucoDetector(marker_length=0.05)

    # Генерация изображения с маркерами
    marker_image = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Генерация маркеров
    for marker_id in [0, 1, 2, 3]:
        marker_img = aruco.generateImageMarker(detector.aruco_dict, marker_id, 200)

        # Размещение маркера на изображении
        x_offset = (marker_id % 2) * 320
        y_offset = (marker_id // 2) * 240
        marker_image[y_offset:y_offset + 200, x_offset:x_offset + 200] = \
            cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    # Обнаружение маркеров
    corners, ids, rejected = detector.detect_markers(marker_image)

    if ids is not None:
        print(f"Обнаружено маркеров: {len(ids)}")
        print(f"ID маркеров: {ids.flatten()}")

        # Визуализация
        result_image = detector.draw_detection(marker_image, corners, ids)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Обнаруженные ArUco маркеры')
        plt.axis('off')
        plt.show()
    else:
        print("Маркеры не обнаружены")


if __name__ == "__main__":
    # Запуск демо со статическим изображением
    # demo_static_image()

    # Запуск основной программы с видеопотоком
    main()