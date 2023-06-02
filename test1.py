import cv2
import mediapipe as mp

import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
import time


cap = cv2.VideoCapture(0)
start_time = time.time()
num_frames = 0
target_fps = 10
delay = 1.0 / target_fps
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        height, width, channels = image.shape
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
            image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
            lanmarksX = []
            lanmarksY = []
            for lanmark in hand_landmarks.landmark:
               lanmarksX.append(lanmark.x)
               lanmarksY.append(lanmark.y)
            min_X = int(min(lanmarksX)*width)
            max_X = int(max(lanmarksX)*width)
            min_Y = int(min(lanmarksY)*height)
            max_Y = int(max(lanmarksY)*height)
            imgCrop = image[min_Y-40:max_Y+40,min_X-40:max_X+40]
            if len(imgCrop) >0 and len(imgCrop[0])>0:
                cv2.imshow('MediaPipe Hands', imgCrop)
                if cv2.waitKey(1)  == ord('k'):
                   cv2.imwrite(f"1.jpg", imgCrop)
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if num_frames == 500:
           break
        if cv2.waitKey(1)  == ord('q'):
            break
cap.release()