import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


class CameraCalibrator:
    def __init__(self, chessboard_size=(9, 6), square_size=0.025):
        """
        Инициализация калибратора камеры

        Args:
            chessboard_size: Количество внутренних углов шахматной доски (width, height)
            square_size: Размер квадрата в метрах
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        # Критерии для субпиксельной точности
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Подготовка объектных точек (3D точки в реальном мире)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        # Массивы для хранения точек
        self.objpoints = []  # 3D точки
        self.imgpoints = []  # 2D точки на изображении

        # Результаты калибровки
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_error = None

    def find_chessboard_corners(self, image_paths, visualize=True):
        """
        Поиск углов шахматной доски на изображениях

        Args:
            image_paths: Список путей к изображениям
            visualize: Визуализация процесса
        """
        successful_calibrations = 0

        for i, image_path in enumerate(image_paths):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Не удалось загрузить изображение: {image_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Поиск углов шахматной доски
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                successful_calibrations += 1

                # Уточнение позиции углов с субпиксельной точностью
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

                self.objpoints.append(self.objp)
                self.imgpoints.append(corners_refined)

                if visualize:
                    # Визуализация найденных углов
                    img_with_corners = cv2.drawChessboardCorners(img.copy(), self.chessboard_size, corners_refined, ret)

                    plt.figure(figsize=(10, 8))
                    plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
                    plt.title(f'Обнаружены углы шахматной доски ({i + 1}/{len(image_paths)})')
                    plt.axis('off')
                    plt.show()

                    print(f"Изображение {i + 1}: Успешно обнаружено {len(corners_refined)} углов")
            else:
                if visualize:
                    print(f"Изображение {i + 1}: Углы не обнаружены")

        print(f"Успешно обработано изображений: {successful_calibrations}/{len(image_paths)}")
        return successful_calibrations >= 5  # Минимум 5 изображений для калибровки

    def calibrate_camera(self):
        """
        Калибровка камеры по найденным точкам
        """
        if len(self.objpoints) < 5:
            print("Недостаточно изображений для калибровки (минимум 5)")
            return False

        # Калибровка камеры
        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints,
            self.imgpoints[0].shape[::-1],  # Размер изображения
            None, None,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

        self.calibration_error = ret
        print(f"Ошибка калибровки (RMS): {ret:.4f} пикселей")

        return ret < 1.0  # Хорошая калибровка имеет ошибку < 1 пикселя

    def analyze_calibration_quality(self, image_paths):
        """
        Анализ качества калибровки
        """
        if self.camera_matrix is None:
            print("Сначала выполните калибровку!")
            return

        reprojection_errors = []

        for i, image_path in enumerate(image_paths[:5]):  # Анализ первых 5 изображений
            img = cv2.imread(image_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret and i < len(self.objpoints):
                # Проецирование 3D точек обратно на изображение
                imgpoints_reprojected, _ = cv2.projectPoints(
                    self.objpoints[i], self.rvecs[i], self.tvecs[i],
                    self.camera_matrix, self.dist_coeffs
                )

                # Ошибка репроекции
                error = cv2.norm(self.imgpoints[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
                reprojection_errors.append(error)

                print(f"Изображение {i + 1}: Ошибка репроекции = {error:.4f} пикселей")

        print(f"Средняя ошибка репроекции: {np.mean(reprojection_errors):.4f} пикселей")

    def save_calibration(self, filename='camera_calibration.npz'):
        """
        Сохранение параметров калибровки
        """
        np.savez(filename,
                 camera_matrix=self.camera_matrix,
                 dist_coeffs=self.dist_coeffs,
                 calibration_error=self.calibration_error)
        print(f"Параметры калибровки сохранены в {filename}")

    def load_calibration(self, filename='camera_calibration.npz'):
        """
        Загрузка параметров калибровки
        """
        try:
            data = np.load(filename)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.calibration_error = data['calibration_error']
            print("Параметры калибровки загружены успешно")
            return True
        except FileNotFoundError:
            print(f"Файл {filename} не найден")
            return False

    def undistort_image(self, image):
        """
        Коррекция дисторсии на изображении
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("Сначала выполните калибровку!")
            return image

        h, w = image.shape[:2]

        # Оптимизация матрицы камеры для коррекции
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )

        # Коррекция дисторсии
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)

        # Обрезка изображения
        x, y, w, h = roi
        undistorted = undistorted[y:y + h, x:x + w]

        return undistorted

    def print_calibration_parameters(self):
        """
        Вывод параметров калибровки
        """
        if self.camera_matrix is None:
            print("Калибровка не выполнена")
            return

        print("\n=== ПАРАМЕТРЫ КАЛИБРОВКИ КАМЕРЫ ===")
        print(f"Матрица камеры:")
        print(f"fx (фокусное по X): {self.camera_matrix[0, 0]:.2f} пикселей")
        print(f"fy (фокусное по Y): {self.camera_matrix[1, 1]:.2f} пикселей")
        print(f"cx (главная точка X): {self.camera_matrix[0, 2]:.2f} пикселей")
        print(f"cy (главная точка Y): {self.camera_matrix[1, 2]:.2f} пикселей")

        print(f"\nКоэффициенты дисторсии:")
        if self.dist_coeffs is not None:
            for i, coeff in enumerate(self.dist_coeffs.flatten()):
                print(f"k{i + 1}: {coeff:.6f}")

        print(f"\nОшибка калибровки (RMS): {self.calibration_error:.4f} пикселей")


# Практический пример использования
def demonstrate_calibration():
    """
    Демонстрация процесса калибровки
    """
    # Создание калибратора
    calibrator = CameraCalibrator(chessboard_size=(9, 6), square_size=0.025)

    # Генерация тестовых изображений (в реальности используйте свои изображения)
    print("1. Создание тестовых изображений шахматной доски...")
    create_test_calibration_images()

    # Загрузка изображений для калибровки
    image_paths = glob.glob('calibration_images/*.jpg')

    if not image_paths:
        print("Не найдены изображения для калибровки!")
        return

    print(f"Найдено {len(image_paths)} изображений для калибровки")

    # Поиск углов шахматной доски
    print("\n2. Поиск углов шахматной доски...")
    success = calibrator.find_chessboard_corners(image_paths, visualize=True)

    if not success:
        print("Не удалось найти достаточно углов для калибровки")
        return

    # Калибровка камеры
    print("\n3. Калибровка камеры...")
    if calibrator.calibrate_camera():
        print("Калибровка выполнена успешно!")

        # Вывод параметров
        calibrator.print_calibration_parameters()

        # Сохранение параметров
        calibrator.save_calibration()

        # Демонстрация коррекции дисторсии
        print("\n4. Демонстрация коррекции дисторсии...")
        demonstrate_undistortion(calibrator, image_paths[0])

        # Использование для ArUco детекции
        print("\n5. Использование для ArUco детекции...")
        use_calibration_for_aruco(calibrator)
    else:
        print("Калибровка не удалась!")


def create_test_calibration_images():
    """
    Создание тестовых изображений для калибровки
    """
    import os
    os.makedirs('calibration_images', exist_ok=True)

    # Простой способ - в реальности снимайте настоящую шахматную доску!
    print("В реальном проекте замените эту функцию на съемку реальной шахматной доски")


def demonstrate_undistortion(calibrator, image_path):
    """
    Демонстрация коррекции дисторсии
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Не удалось загрузить изображение для демонстрации")
        return

    # Коррекция дисторсии
    undistorted = calibrator.undistort_image(img)

    # Визуализация
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Исходное изображение (с дисторсией)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    plt.title('Скорректированное изображение')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def use_calibration_for_aruco(calibrator):
    """
    Использование калибровки для ArUco детекции
    """
    from previous_code import ArucoDetector

    # Создание детектора с калиброванными параметрами
    detector = ArucoDetector(
        camera_matrix=calibrator.camera_matrix,
        dist_coeffs=calibrator.dist_coeffs,
        marker_length=0.05
    )

    print("Детектор ArUco инициализирован с калиброванными параметрами!")
    print("Теперь можно точно определять положение маркеров в 3D пространстве")


# Дополнительные утилиты
def estimate_focal_length(sensor_width, focal_length_mm, image_width_px):
    """
    Оценка фокусного расстояния в пикселях

    Args:
        sensor_width: Ширина сенсора в мм
        focal_length_mm: Фокусное расстояние в мм
        image_width_px: Ширина изображения в пикселях
    """
    focal_length_px = (focal_length_mm * image_width_px) / sensor_width
    return focal_length_px


def create_approximate_camera_matrix(image_width, image_height, fov_degrees=60):
    """
    Создание приближенной матрицы камеры

    Args:
        image_width: Ширина изображения
        image_height: Высота изображения
        fov_degrees: Угол обзора в градусах
    """
    # Преобразование угла обзора в фокусное расстояние
    fov_rad = np.radians(fov_degrees)
    focal_length = (image_width / 2) / np.tan(fov_rad / 2)

    camera_matrix = np.array([
        [focal_length, 0, image_width / 2],
        [0, focal_length, image_height / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    return camera_matrix


if __name__ == "__main__":
    # Демонстрация калибровки
    demonstrate_calibration()

    # Пример создания приближенной матрицы
    print("\n=== ПРИМЕР ПРИБЛИЖЕННОЙ МАТРИЦЫ ===")
    approx_matrix = create_approximate_camera_matrix(640, 480, 60)
    print("Приближенная матрица камеры для 640x480, FOV=60°:")
    print(approx_matrix)