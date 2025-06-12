import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import mediapipe as mp
import csv
import datetime

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

csv_filename = "/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/Music Recommendation - Synthetic Verifier/face_images.csv"

fieldnames = ["image_name", "0", "1", "2", "3", "4", "5", "6", "7", "8"]

INPUT_MODE = 0
cap = cv.VideoCapture(0)

images_folder_root = "/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/Music Recommendation - Synthetic Verifier/face_images"
list_coords_wrt_center = []

with mp_holistic.Holistic(
       static_image_mode=False,
       model_complexity=1,
       smooth_landmarks=True,
       min_detection_confidence=0.5,
       min_tracking_confidence=0.5) as holistic:

        # for ep in range(num_frames*total_episodes):
        while True:
            face_coords = {"0": None, "1": None, "2": None, "3": None, "4": None, "5": None, "6": None, "7": None, "8": None}

            success, img = cap.read()

            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)

            if (success) and (img is not None):
                now = datetime.datetime.now()
                pose_landmarks = results.pose_landmarks
                  
                if pose_landmarks:
                # mp_drawing.draw_landmarks(img2, pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    for key in face_coords:
                        t = pose_landmarks.landmark[int(key)]
                        face_coords[key] = [t.x, t.y, t.z, t.visibility]
                    
                    # N_M = []

                    for key in face_coords:
                        x, y, z, vis = face_coords[key]
                        # N_M.append([x, y, z, vis])

                        x *= img.shape[1]
                        x = int(x)
                        
                        y *= img.shape[0]
                        y = int(y)

                        # cv.circle(img, (x, y), 3, (255, 0, 0), 3)
                    
                    face_coords['image_name'] = str(now)
                    list_coords_wrt_center.append(face_coords)

                    # cv.imwrite("/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/Music Recommendation - Synthetic Verifier/face_images/" + str(now) + ".jpg", img)
                
                    cv.imshow("img", img)
                
                if False or (cv.waitKey(1) & 0xFF == ord('q')):
                    with open(csv_filename, mode='a', newline='\n') as file:
                            writer = csv.DictWriter(file, fieldnames=fieldnames)
                            writer.writerows(list_coords_wrt_center)
                    break

cap.release()
cv.destroyAllWindows()
