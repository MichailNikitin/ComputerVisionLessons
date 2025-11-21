#Нынешняя калибровка для определения коричневого щебня: min(21;56;0) max(45;255;193)
import cv2
import numpy as np

if __name__ == '__main__':
    def nothing(*arg):
        pass

cv2.namedWindow("result")  # создаем главное окно
cv2.namedWindow("settings")  # создаем окно настроек

cap = cv2.VideoCapture(1)
# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('h_min', 'settings', 0, 255, nothing)
cv2.createTrackbar('s_min', 'settings', 68, 255, nothing)
cv2.createTrackbar('v_min', 'settings', 0, 255, nothing)
cv2.createTrackbar('h_max', 'settings', 255, 255, nothing)
cv2.createTrackbar('s_max', 'settings', 255, 255, nothing)
cv2.createTrackbar('v_max', 'settings', 189, 255, nothing)
crange = [0, 0, 0, 0, 0, 0]

while True:
    flag, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # считываем значения бегунков
    h1 = cv2.getTrackbarPos('h_min', 'settings')
    s1 = cv2.getTrackbarPos('s_min', 'settings')
    v1 = cv2.getTrackbarPos('v_min', 'settings')
    h2 = cv2.getTrackbarPos('h_max', 'settings')
    s2 = cv2.getTrackbarPos('s_max', 'settings')
    v2 = cv2.getTrackbarPos('v_max', 'settings')

    # формируем начальный и конечный цвет фильтра
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    # накладываем фильтр на кадр в модели HSV
    thresh = cv2.inRange(hsv, h_min, h_max)
    col = cv2.bitwise_and(hsv, hsv, mask=thresh)
    res = cv2.cvtColor(col, cv2.COLOR_HSV2BGR, col)

    cv2.imshow('result', thresh)
    cv2.imshow('mask', res)

    ch = cv2.waitKey(5)
    if ch == 27:
        break

cap.release()
cv2.destroyAllWindows()