import pickle
import speech_recognition as sr
from gtts import gTTS
import playsound
import pyttsx3
import time
import os
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
word = ""
predicted_character=""
count_same_frame = 0
prediction_probability=""
probability_value=""
labels_dict = {0: 'A', 1: 'B', 2: 'C',3: 'D', 4: 'E', 5: 'F',6: 'G', 7: 'H', 8: 'I',9: 'J', 10: 'K', 11: 'L',12: 'M', 13: 'N', 14: 'O',15: 'P', 16: 'Q', 17: 'R',18: 'S', 19: 'T', 20: 'U',21: 'V', 22: 'W', 23: 'X',24: 'Y',25: 'Z', 26:'_', 27:'DEL'}

def speak(text):
    engine = pyttsx3.init()
    # Thiết lập các tùy chỉnh giọng nói
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.setProperty('voice', 'vietnamese_m1')
    engine.say(text)
    engine.runAndWait()

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cropped_image = frame[0:400, 0:400]
    # resized_frame = cv2.resize(cropped_image, (200,200))
    # reshaped_frame = (np.array(resized_frame)).reshape((1, 200, 200, 3))
    # frame_for_model = reshaped_frame/255.0
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Lấy bàn tay đầu tiên

    # Vẽ điểm dấu tay cho bàn tay đầu tiên
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        old_predicted_character = predicted_character
        print(predicted_character)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        # if predicted_character == 'SPACE':
        #     predicted_character = '_'
        	
        if old_predicted_character == predicted_character:
            count_same_frame += 1
        else:
            count_same_frame = 0
                
        if count_same_frame > 50:
            # speak(predicted_character)
            if predicted_character == 'DEL':
                word =  word[:-1]
            else:
                word = word + predicted_character
            count_same_frame = 0
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
    cv2.putText(blackboard, f"Predict: {predicted_character}", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
    cv2.putText(blackboard, word, (30, 300), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    res = np.hstack((frame, blackboard))
    cv2.imshow('frame', res)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()