import cv2
import numpy as np


def nothing(*arg):
    pass


cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.namedWindow("settings", cv2.WINDOW_NORMAL)
cv2.namedWindow("original", cv2.WINDOW_NORMAL)

# Устанавливаем размеры окон
cv2.resizeWindow("result", 800, 600)
cv2.resizeWindow("original", 800, 600)
cv2.resizeWindow("settings", 800, 600)  # Увеличиваем окно настроек

# Создаем трекбары для цветового диапазона
cv2.createTrackbar('h1', 'settings', 0, 180, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
cv2.createTrackbar('h2', 'settings', 180, 180, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)

# Создаем трекбары для морфологических операций
cv2.createTrackbar('Erosion', 'settings', 0, 20, nothing)
cv2.createTrackbar('Dilation', 'settings', 0, 20, nothing)
cv2.createTrackbar('Kernel Size', 'settings', 3, 15, nothing)

# Создаем объект VideoCapture вне цикла
cap = cv2.VideoCapture(0)

while True:
    # Читаем кадр с камеры
    ret, img = cap.read()
    if not ret:
        print("Не удалось получить кадр с камеры")
        break

    # Сохраняем оригинальный размер для обработки
    h, w = img.shape[:2]

    # Увеличиваем размер изображения для отображения
    display_img = cv2.resize(img, (w, h))
    display_hsv = cv2.cvtColor(display_img, cv2.COLOR_BGR2HSV)

    # считываем значения бегунков цветового диапазона
    h1 = cv2.getTrackbarPos('h1', 'settings')
    s1 = cv2.getTrackbarPos('s1', 'settings')
    v1 = cv2.getTrackbarPos('v1', 'settings')
    h2 = cv2.getTrackbarPos('h2', 'settings')
    s2 = cv2.getTrackbarPos('s2', 'settings')
    v2 = cv2.getTrackbarPos('v2', 'settings')

    # считываем значения бегунков морфологических операций
    erosion_iter = cv2.getTrackbarPos('Erosion', 'settings')
    dilation_iter = cv2.getTrackbarPos('Dilation', 'settings')
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'settings')

    # Гарантируем, что размер ядра нечетный и не меньше 1
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    # Бинаризация по цветовому диапазону
    img_bin = cv2.inRange(display_hsv, h_min, h_max)

    # Применяем морфологические операции
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Эрозия (уменьшение белых областей)
    if erosion_iter > 0:
        img_bin = cv2.erode(img_bin, kernel, iterations=erosion_iter)

    # Дилатация (увеличение белых областей)
    if dilation_iter > 0:
        img_bin = cv2.dilate(img_bin, kernel, iterations=dilation_iter)

    # Создаем цветное изображение с выделенными контурами
    result_colored = display_img.copy()
    result_colored[img_bin == 255] = [0, 255, 0]  # Зеленый цвет для выделенных областей

    # Отображаем информацию о параметрах на изображении
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result_colored, f'Erosion: {erosion_iter}', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(result_colored, f'Dilation: {dilation_iter}', (10, 60), font, 0.7, (255, 255, 255), 2)
    cv2.putText(result_colored, f'Kernel: {kernel_size}x{kernel_size}', (10, 90), font, 0.7, (255, 255, 255), 2)

    # Показываем изображения
    cv2.imshow('result', result_colored)
    cv2.imshow('original', display_img)
    cv2.imshow('binary', img_bin)  # Добавляем окно с бинарным изображением

    ch = cv2.waitKey(5)
    if ch == 27:  # ESC
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()