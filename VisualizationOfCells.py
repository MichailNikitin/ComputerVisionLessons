import cv2
import numpy as np

# Загрузка параметров калибровки
calibration_data = np.load("camera_calibration_good.npz")
K = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Загрузка изображения звезды
star_img = cv2.imread('star.png', cv2.IMREAD_UNCHANGED)  # с альфа-каналом
if star_img is None:
    print("Не удалось загрузить star.png")
    exit()

# Параметры
h = 2.0  # высота камеры
cell_size_m = 0.05  # размер клетки в метрах

# Инициализация камеры
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр.")
        break

    height, width = frame.shape[:2]
    fx, fy = K[0, 0], K[1, 1]

    # Масштаб: пиксель <-> метр
    pixels_per_meter_x = fx / h
    pixels_per_meter_y = fy / h

    # Размер клетки в пикселях
    px_per_cell_x = int(cell_size_m * pixels_per_meter_x)
    px_per_cell_y = int(cell_size_m * pixels_per_meter_y)

    if px_per_cell_x <= 0 or px_per_cell_y <= 0:
        print(f"Ошибка: размер клетки в пикселях недопустим: {px_per_cell_x}x{px_per_cell_y}")
        print(f"Проверь: fx={fx}, fy={fy}, h={h}, cell_size_m={cell_size_m}")
        break

    # Функция отрисовки сетки
    def draw_grid(img, cell_size_x, cell_size_y, color=(0, 255, 0), thickness=1):
        h, w = img.shape[:2]
        for x in range(0, w, cell_size_x):
            cv2.line(img, (x, 0), (x, h), color, thickness)
        for y in range(0, h, cell_size_y):
            cv2.line(img, (0, y), (w, y), color, thickness)

    # Рисуем сетку
    output = frame.copy()
    draw_grid(output, px_per_cell_x, px_per_cell_y)

    # Функция для наложения изображения (например, звезды)
    def overlay_image_alpha(img, overlay, pos):
        x, y = pos
        h, w = overlay.shape[:2]

        # Проверяем, чтобы не выйти за границы
        if x + w <= 0 or y + h <= 0 or x >= img.shape[1] or y >= img.shape[0]:
            return  # Всё изображение вне кадра

        # Обрезаем overlay и alpha, если он выходит за границы
        x1, y1 = max(0, x), max(0, y)
        x2 = min(img.shape[1], x + w)
        y2 = min(img.shape[0], y + h)

        # Размеры обрезанной области
        w_clip = x2 - x1
        h_clip = y2 - y1

        # Обрезаем overlay и alpha
        overlay_clip = overlay[:h_clip, :w_clip]
        if overlay_clip.shape[2] == 4:
            alpha = overlay_clip[:, :, 3] / 255.0
            roi = img[y1:y2, x1:x2]

            for c in range(3):
                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay_clip[:, :, c]
        else:
            img[y1:y2, x1:x2] = overlay_clip

    # Функция рисования фигур
    def draw_shape(img, x, y, shape='circle', color=(0, 0, 255)):
        center = (x, y)
        if shape == 'circle':
            cv2.circle(img, center, 10, color, -1)
        elif shape == 'square':
            pt1 = (x - 10, y - 10)
            pt2 = (x + 10, y + 10)
            cv2.rectangle(img, pt1, pt2, color, -1)
        elif shape == 'star':
            # Центрируем изображение звезды
            h, w = star_img.shape[:2]
            pos = (x - w // 2, y - h // 2)
            overlay_image_alpha(img, star_img, pos)

    # Пример: рисуем фигуры в определённых клетках
    draw_shape(output, 2 * px_per_cell_x + px_per_cell_x // 2,
               3 * px_per_cell_y + px_per_cell_y // 2, 'circle', (255, 0, 0))

    draw_shape(output, 4 * px_per_cell_x + px_per_cell_x // 2,
               1 * px_per_cell_y + px_per_cell_y // 2, 'star', (0, 255, 0))

    # Рисуем стрелку
    start = (2 * px_per_cell_x + px_per_cell_x // 2, 3 * px_per_cell_y + px_per_cell_y // 2)
    end = (4 * px_per_cell_x + px_per_cell_x // 2, 1 * px_per_cell_y + px_per_cell_y // 2)
    cv2.arrowedLine(output, start, end, (0, 0, 255), 2)

    # Отображение кадра
    cv2.imshow("Overlayed Frame", output)

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()