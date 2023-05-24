import cv2 as cv
import numpy as np

# Определяем камеру с помощью OpenCV
vid = cv.VideoCapture(0)

# Проверяем, успешно ли открыта камера
if not vid.isOpened():
    print("Не удалось открыть камеру")
    exit()

# Определяем начальные координаты и размеры прямоугольника
x, y, w, h = 0, 0, 0, 0

# Переменные для отслеживания движения
previous_frame = None #хранит предыдущий кадр для сравнения с текущим
motion_threshold = 1000 # пороговое значение площади контура, выше которого считается, что объект движется.

while True:
    # Берем видео с камеры
    ret, frame = vid.read()

    # Преобразуем изображение в цветовое пространство HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Определяем нижний и верхний цветовые пороги для фильтра
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([10, 255, 255])

    # Применяем цветовой фильтр к изображению
    mask = cv.inRange(hsv, lower_color, upper_color)

    # Находим контуры объектов на изображении
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Обновляем предыдущий кадр для отслеживания движения
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21, 21), 0)
    if previous_frame is None:
        previous_frame = gray
        continue

    # Вычисляем абсолютное различие между текущим и предыдущим кадром
    frame_delta = cv.absdiff(previous_frame, gray)

    # Применяем порог для выделения движения
    thresh = cv.threshold(frame_delta, 25, 255, cv.THRESH_BINARY)[1]

    # Расширяем контуры для устранения шума
    thresh = cv.dilate(thresh, None, iterations=2)

    # Находим контуры движущихся объектов
    motion_contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Выбираем контур с наибольшей площадью и подходящим размером
    filtered_motion_contours = []
    for contour in motion_contours:
        contour_area = cv.contourArea(contour)
        if contour_area > motion_threshold:
            filtered_motion_contours.append(contour)

    if len(filtered_motion_contours) > 0:
        max_motion_contour = max(filtered_motion_contours, key=cv.contourArea)
        epsilon = 0.02 * cv.arcLength(max_motion_contour, True)
        approx = cv.approxPolyDP(max_motion_contour, epsilon, True)
        x, y, w, h = cv.boundingRect(approx)

        # Фильтруем прямоугольник по размерам
        if w > 30 and h > 30:
            # Рисуем красный квадрат на изображении для выделения движущегося объекта
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Обновляем предыдущий кадр
    previous_frame = gray

    # Отображаем изображение
    cv.imshow('frame', frame)

    # Проверяем, была ли нажата клавиша 'q'(закрывает программу)
    if cv.waitKey(1) & 0xFF in (ord('q'), 27):
        break

    # Проверяем, закрыто ли окно пользователем(закрывает программу)
    if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) < 1:
        break

# После цикла освобождаем камеру
vid.release()

# Уничтожаем все окна
cv.destroyAllWindows()


#добавить исчезновение прямоугольника с экрана в случае отсутствия движения
