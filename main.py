import time

import cv2
import numpy as np


def task_1():
    img = cv2.imread('images/variant-6.png')  # читаем изображение
    w, h = img.shape[:2]  # определяем его размеры
    w_new, h_new = map(lambda coord: coord * 2, [w, h])  # определяем новые размеры растянутого в 2 раза изображение
    img_new = cv2.resize(img, (w_new, h_new))  # растягиваем изображение в 2 раза
    cv2.imshow('New scaled image', img_new)  # показываем растянутое изображение


def task_2():
    cap = cv2.VideoCapture(0)  # подключаемся к камере по её id (обычно равно нулю)
    down_points = (640, 480)  # определяем размер выводимого изображения
    r_count, l_count, i = 0, 0, 0  # заводим счётчик попадания метки в правую и левую половину экрана и счётчик кадров
    while True:  # выводим в бесконечном цикле изображения, имитируя видео с камеры
        ret, frame = cap.read()  # читаем фрейм с камеры во frame
        if not ret:  # и проверяем, True ли ret, т.е. правильно ли прочитался фрейм
            break

        # подгоняем изображение с камеры до определённого выше размера down_points
        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)

        # получаем метку
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:  # если найдены похожие контуры, ...
            c = max(contours, key=cv2.contourArea)  # ... то получаем максимальный по площади контур из всех возможных
            x, y, w, h = cv2.boundingRect(c)  # определяем его размеры
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # очерчиваем голубой прямоугольник
            # располагаем слева и справа сверху соответствующие блоки текста со счётчиками
            cv2.putText(frame, f'Слева: {l_count}', (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Справа: {r_count}', (down_points[0] - 90, 20), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if i % 10 == 0:  # каждые 10 кадров...
                # ... проверяем, где находится метка и увеличиваем соответствующий счётчик в зависимости от того справа
                # она или слева на фрейме, считая от середины
                if x + (w // 2) > down_points[0] // 2:
                    r_count += 1
                elif x + (w // 2) < down_points[0] // 2:
                    l_count += 1

        # показываем фрейм со всеми добавленными на него объектами
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # завершить цикл, если нажата Q
            break

        time.sleep(0.1)
        i += 1  # увеличиваем счётчик кадров

    cap.release()  # отключаемся от камеры


def extra_task():
    cap = cv2.VideoCapture(0)  # подключаемся к камере по её id (обычно равно нулю)
    down_points = (640, 480)  # определяем размер выводимого изображения
    r_count, l_count, i = 0, 0, 0  # заводим счётчик попадания метки в правую и левую половину экрана и счётчик кадров
    fly = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)  # читаем муху из файла с флагом, не изменяющим изображение

    while True:  # выводим в бесконечном цикле изображения, имитируя видео с камеры
        ret, frame = cap.read()  # читаем фрейм с камеры во frame
        if not ret:  # и проверяем, True ли ret, т.е. правильно ли прочитался фрейм
            break

        # подгоняем изображение с камеры до определённого выше размера down_points
        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)

        # получаем метку
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:  # если найдены похожие контуры, ...
            c = max(contours, key=cv2.contourArea)  # ... то получаем максимальный по площади контур из всех возможных
            x, y, w, h = cv2.boundingRect(c)  # определяем его размеры

            # определяем размеры мухи по размерам контура
            new_fly = cv2.resize(fly, (w if x + w < down_points[0] else down_points[0] - x,
                                       h if y + h < down_points[1] else down_points[1] - y))
            alpha_channel, fly_colors = new_fly[:, :, 3] / 255, new_fly[:, :, :3]  # извлекаем альфа-канал и цвета мухи
            # находим альфа-маску, делающую изображение прозрачным
            alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
            h_fly, w_fly = new_fly.shape[:2]  # изменяем форму с (h, y, 4) на (h, y)
            background_subsection = frame[y:(y + h_fly), x:(x + w_fly)]  # выделяем прямоугольник мухи из фрейма камеры
            # накладываем муху с применением маски на прямоугольник из фрейма
            composite = background_subsection * (1 - alpha_mask) + fly_colors * alpha_mask
            frame[y:(y + h_fly), x:(x + w_fly)] = composite  # вставляем полученное наложение в фрейм

            # располагаем слева и справа сверху соответствующие блоки текста со счётчиками
            cv2.putText(frame, f'Слева: {l_count}', (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Справа: {r_count}', (down_points[0] - 90, 20), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if i % 10 == 0:  # каждые 10 кадров...
                # ... проверяем, где находится метка и увеличиваем соответствующий счётчик в зависимости от того справа
                # она или слева на фрейме, считая от середины
                if x + (w // 2) > down_points[0] // 2:
                    r_count += 1
                elif x + (w // 2) < down_points[0] // 2:
                    l_count += 1

        # показываем фрейм со всеми добавленными на него объектами
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # завершить цикл, если нажата Q
            break

        time.sleep(0.1)
        i += 1  # увеличиваем счётчик кадров

    cap.release()  # отключаемся от камеры


if __name__ == '__main__':
    task_1()
    # task_2()
    # extra_task()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
