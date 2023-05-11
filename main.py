#в данной версии мы используем алгоритм выделения обьекта с помощью цветового фильтра
#(это делается для того, чтобы избежать ошибок если обьект большой и контрастный)
#opencv - библиотека для обработки видеопотока с камеры и отображения на экране прямоугольника вокруг объекта определенного цвета.

import cv2 as cv
import numpy as np

# определяем камеру с помощью opencv
vid = cv.VideoCapture(0)

# определяем нижний и верхний цветовые пороги для фильтра
lower_color = np.array([0, 50, 50])
upper_color = np.array([10, 255, 255])

# определяем начальные координаты и размеры прямоугольника
x, y, w, h = 0, 0, 0, 0 

while(True):
    # берем видео с камеры
    ret, frame = vid.read()

    # преобразуем изображение в цветовое пространство HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # применяем цветовой фильтр к изображению
    mask = cv.inRange(hsv, lower_color, upper_color)

    # находим контуры объектов на изображении
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # выбираем контур с наибольшей площадью
    if len(contours) > 0:
        max_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(max_contour)

    # рисуем прямоугольник на изображении
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # отображаем
    cv.imshow('frame', frame)

    # проверяем, была ли нажата клавиша 'q'
    if cv.waitKey(1) & 0xFF == ord('q') or cv.waitKey(1) & 0xFF == 27:
        break

    # проверяем, закрыта ли окно пользователем
    if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) < 1:
        break

# после цикла освобождаем камеру
vid.release()

# уничтожаем все окна
cv.destroyAllWindows()
