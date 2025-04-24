import cv2
import os, json
import tempfile
import threading
import time
from paddleocr import PaddleOCR
import numpy as np
from moviepy import VideoFileClip
import pygame

from google.cloud import vision

#main settings
video_path = "va1.mp4"
audio_path = None
cls = True # angles

path = os.path.dirname(os.path.abspath(__file__))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(path, "keys.json")

client = vision.ImageAnnotatorClient()

if __name__ == "__main__":
    ocr = PaddleOCR(cls=cls, lang='ru', use_gpu=True)
    temp_file = False

    # === Извлечение аудио во временный .wav файл ===
    if not audio_path:
        clip = VideoFileClip(video_path)
        if not clip.audio:
            raise ValueError("❌ В видео нет аудио-дорожки!")

        temp_audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
        
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
    audio_thread.start()

    boxs = []

    def recognize_google(frame):
        global boxs
        _, buffer = cv2.imencode(".jpg", frame)
        content = buffer.tobytes()
        image_context = vision.ImageContext(language_hints=["ru"])
        image = vision.Image(content=content)

        # Распознавание текста (OCR)
        try:
            new_boxs = []
            response = client.text_detection(image=image, image_context=image_context)
            texts = response.text_annotations
            for text in texts[1:]:  # texts[0] — это весь текст, не нужно рисовать рамки для всего текста
                vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                new_boxs.append(vertices)
            boxs = new_boxs
            if texts:
                print("Распознанный текст:")
                print(texts[0].description)
                return texts[0].description
            else:
                print("Текст не найден.")
        except Exception as e:
            print("EXECPTION: ", e)
            
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
            return full_text

    pause = False
    ret, frame_original, frame = None, None, None
    
    pause_time = time.time()
    with open("result.txt", "w", encoding="utf-8") as f:
        while cap.isOpened():
            if not pause:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_original = frame.copy()
            
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            time_sec = frame_num / fps

            # pylint: disable=no-member
            key = cv2.waitKey(1)
                            
            if key == 13:
                boxs = []
                if pause:
                    frame = frame_original.copy()

            if key == 32:
                boxs = []
                text = recognize_text(frame=frame)
                #text = recognize_google(frame=frame_original)
                if text:
                    f.write(text + '\n\n')
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
            for box in boxs:
                points = [(int(point[0]), int(point[1])) for point in box]
                
                cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

            cv2.imshow("Video", frame)

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