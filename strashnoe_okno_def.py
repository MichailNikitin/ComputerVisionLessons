import numpy as np
import cv2


# --- Загрузка калибровочных данных ---
calibration_data = np.load("camera_calibration_good.npz")
K = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Параметры
h = 2.0                     # высота камеры над землёй (в метрах)
cell_size_m = 0.05          # размер одной клетки в метрах (5 см)

# Фокусные расстояния из матрицы камеры
fx = K[0, 0]
fy = K[1, 1]

# Масштаб: пиксель <-> метр
pixels_per_meter_x = fx / h
pixels_per_meter_y = fy / h

# Размер клетки в пикселях
px_per_cell_x = int(cell_size_m * pixels_per_meter_x)
px_per_cell_y = int(cell_size_m * pixels_per_meter_y)

print(f"Размер клетки: {px_per_cell_x}x{px_per_cell_y} пикселей")

def sliding_window_denoise(binary_mask, cell_w, cell_h, threshold=1):
    """
    Обрабатывает бинарную маску методом скользящего окна (по сетке),
    объединяя пиксели в клетки. Если в клетке >= threshold пикселей == 1,
    то вся клетка становится 1.

    Parameters:
        binary_mask: np.ndarray, dtype=uint8, значения 0 или 255 (или 0/1)
        cell_w, cell_h: размеры клетки в пикселях
        threshold: минимальное число активных пикселей для активации клетки

    Returns:
        cleaned_mask: np.ndarray той же формы, что и binary_mask
    """
    h_img, w_img = binary_mask.shape
    cleaned = np.zeros_like(binary_mask)

    # Нормализуем маску к 0/1
    mask_bin = (binary_mask > 0).astype(np.uint8)

    for y in range(0, h_img, cell_h):
        for x in range(0, w_img, cell_w):
            # Определяем границы текущей ячейки
            y_end = min(y + cell_h, h_img)
            x_end = min(x + cell_w, w_img)

            window = mask_bin[y:y_end, x:x_end]
            if np.sum(window) >= threshold:
                cleaned[y:y_end, x:x_end] = 255  # или 1, в зависимости от формата

    return cleaned

# --- Пример использования ---
if name == "__main__":
    # Загрузите ваше бинарное изображение (например, результат детекции)
    # binary_input = cv2.imread("binary_mask.png", cv2.IMREAD_GRAYSCALE)
    
    # ИЛИ создадим тестовую маску с шумом
    binary_input = np.zeros((480, 640), dtype=np.uint8)
    # Добавим несколько "пикселей-шумов"
    binary_input[100:102, 200:202] = 255
    binary_input[105, 205] = 255
    binary_input[300:310, 400:410] = 255  # крупный объект

    # Обработка
    cleaned_output = sliding_window_denoise(
        binary_input,
        cell_w=px_per_cell_x,
        cell_h=px_per_cell_y,
        threshold=1  # даже один пиксель активирует клетку
    )

    # Сохранение или отображение результата
    cv2.imshow("Input", binary_input)
    cv2.imshow("Cleaned", cleaned_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
