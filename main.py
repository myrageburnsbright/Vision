import cv2
from paddleocr import PaddleOCR
import paddle
import numpy as np
cls = True
ocr = PaddleOCR(cls=cls, lang='ru', use_gpu=True)

cap = cv2.VideoCapture("dd2.mp4")
boxs = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1)
    if key == 32:  # По нажатию 's' — распознать текст
        result = ocr.ocr(frame, cls=cls)
        lines = []
        if not result[0]:
            print("")
        else:
            new_boxs = []
            for line in result[0]:
                text = line[1][0]
                lines.append((line[0][0][1], text))  # Y-координата + текст

                new_boxs.append(line[0])
        # Преобразуем их в целые числа
                
                
                # Рисуем рамку вокруг текста
                
            boxs = new_boxs
            # Сортировка по Y (по вертикали)
            lines.sort(key=lambda x: x[0])

            # Объединение в строку
            full_text = ' '.join([text for _, text in lines])
            print(full_text)
        
    if key == 13:
        boxs = []
    for box in boxs:
        points = [(int(point[0]), int(point[1])) for point in box]
        cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Video", frame)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()