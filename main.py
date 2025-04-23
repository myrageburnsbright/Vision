import cv2
import os
import tempfile
import threading
import time
from paddleocr import PaddleOCR
import numpy as np
from moviepy import VideoFileClip
import pygame

cls = True # angles
ocr = PaddleOCR(cls=cls, lang='ru', use_gpu=True)

video_path = "la.mp4"
audio_path = None

temp_file = False

# === Извлечение аудио во временный .wav файл ===
if not audio_path:
    clip = VideoFileClip(video_path)
    if not clip.audio:
        raise ValueError("❌ В видео нет аудио-дорожки!")

    temp_audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")

    print(temp_audio_path)
    clip.audio.write_audiofile(temp_audio_path, logger=None)
    temp_file = True
    audio_path = temp_audio_path

def play_audio():
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

# pylint: disable=no-member
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_duration = 1 / fps
start_time = time.time()

audio_thread = threading.Thread(target=play_audio)
#audio_thread.daemon = True
audio_thread.start()

boxs = []

def recognize_text(frame):
    global boxs
    # Преобразуем кадр в черно-белое изображение для лучшего контраста
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Применение бинарного порога для улучшения контраста
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    result = ocr.ocr(thresh, cls=cls)
    lines = []
    if not result[0]:
        print("❌ Не удалось распознать текст!")
    else:
        new_boxs = []
        for line in result[0]:
            text = line[1][0]
            lines.append((line[0][0][1], text))  # Y-координата + текст

            new_boxs.append(line[0])
            
        boxs = new_boxs
        # Сортировка по Y (по вертикали)
        lines.sort(key=lambda x: x[0])

        # Объединение в строку
        full_text = ' '.join([text for _, text in lines])
        print(full_text)

pause = False
ret, frame = None, None

pause_time = time.time()
with open("result.txt", "r+", encoding="utf-8") as f:
    while cap.isOpened():
        if not pause:
            ret, frame = cap.read()

        if not ret:
            break
        
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        time_sec = frame_num / fps

        # pylint: disable=no-member
        key = cv2.waitKey(1)
        if key == 32:
            recognize_text(frame=frame)
            
        if key == 13:
            boxs = []
        for box in boxs:
            points = [(int(point[0]), int(point[1])) for point in box]
            cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("Video", frame)
        if key == ord('q'):
            break
        
        if key == ord('p'):
            if pause:
                pygame.mixer.music.unpause()
                start_time = start_time + (time.time() - pause_time)
            else:
                pygame.mixer.music.pause()
                pause_time = time.time()
            pause = not pause

        elapsed = time.time() - start_time
        expected = frame_num * frame_duration
        delay = expected - elapsed
        if delay > 0:
            time.sleep(delay)

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()

    pygame.mixer.music.unload()
        
    if temp_file:
        os.remove(audio_path)