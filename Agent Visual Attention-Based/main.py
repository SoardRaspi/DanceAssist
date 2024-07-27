"""
This model will try to understand the attention of the user.
The model will try to understand what musical notes or the type of music that attracts the user's attention.

This model will know what grabs the user's attention in terms of music and try to generate new music to bring back the
user's attentions

Two approaches:
1. First take the input of the parameters when the user is looking at the screen.
2. Directly take data when the user is trying to perform the motion.

Inputs: 1. The eye images of the user, re-shaped
        2. The face landmark points in the form of a GCN
Output: Spectrogram of generated music...

----------------------------------------------------------------------

The ultimate task is to know which music composition best suits the user. We do this based on the feedback of the user
for the attention of the user.
"""

# from numpy import loadtxt
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
#
# dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
#
# X = dataset[:, 0:8]
# Y = dataset[:, 8]
#
# model = Sequential()
# model.add(Dense(12, input_shape=(8,), activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(X, Y, epochs=150, batch_size=10)
#
# _, accuracy = model.evaluate(X, Y)
#
# # make class predictions with the model
# predictions = (model.predict(X) > 0.5).astype(int)
# # summarize the first 5 cases
# for i in range(5):
#     print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))

import keras
import torch
from keras import layers
from keras import models

import mediapipe as mp
import numpy as np
import cv2 as cv

from threading import Event, Thread
from music_generator import play_tick
import queue

# from Music_Relevance_Attention import MRA

import MRA_agent

# face
# 0: nose

# right leg
# 24: right hip
# 26: right knee
# 28: right ankle
# 30: right heel
# 32: right foot index

# left leg
# 23: left hip
# 25: left knee
# 27: left ankle
# 29: left heel
# 31: left foot index

# right arm
# 12: right shoulder
# 14: right elbow
# 16: right wrist
# 18: right pinky
# 20: right index

# left arm
# 11: left shoulder
# 13: left elbow
# 15: left wrist
# 17: left pinky
# 19: left index

# Movement metrics:

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
# mp_face_mesh = mp.solutions.face_mesh

cap = cv.VideoCapture(0)

# right_eye_main_dict = {33: None, 246: None, 161: None, 160: None, 159: None, 158: None, 157: None,
#                        173: None, 133: None, 155: None, 154: None, 153: None, 145: None, 144: None,
#                        163: None, 7: None}
#
# left_eye_main_dict = {263: None, 466: None, 388: None, 387: None, 386: None, 385: None, 384: None,
#                       398: None, 362: None, 382: None, 381: None, 380: None, 374: None, 373: None,
#                       390: None, 249: None}
#
# r_lobe_coords = {130: None, 247: None, 30: None, 29: None, 27: None, 28: None, 56: None, 190: None, 243: None,
#                  25: None,
#                  110: None, 24: None, 23: None, 22: None, 26: None, 112: None}
#
# l_lobe_coords = {463: None, 414: None, 286: None, 258: None, 257: None, 259: None, 260: None, 467: None, 359: None,
#                  255: None, 339: None, 254: None, 253: None, 252: None, 256: None, 341: None}
#
# percents_re = []
# percents_le = []
# re_points = []
# le_points = []

mra_agent = MRA_agent.Agent()

main_face = [i for i in range(9)]

window_size = 60
ST_GCN_input = [[], [], [], [], [], [], [], [], []]

music_one_hot = [0 for i in range(10)]
music_one_hot[0] = 1

t_2_event = Event()
def t_2_func(queue):
    # while not t_2_event.is_set():
    while True:
        frequency, duration, bpm = queue.get()

        try:
            if frequency < -1:
                play_tick(20, duration, 60 / 60)
            else:
                # play_tick(frequency, duration, 60 / bpm)
                play_tick(frequency, duration, 60 / 60)
        except:
            print("error in playing music:", frequency, duration, bpm)

speed_queue = queue.Queue()
Thread(target=t_2_func, args=(speed_queue,)).start()

while True:
    # regions_re = {1: [33, 246, 7], 2: [246, 7, 163, 161], 3: [161, 163, 144, 160], 4: [160, 144, 145, 159],
    #               5: [159, 145, 153, 158], 6: [158, 153, 154, 157], 7: [157, 154, 155, 173], 8: [173, 155, 133]}
    # regions_le = {1: [263, 466, 249], 2: [466, 249, 390, 388], 3: [388, 390, 373, 387], 4: [387, 373, 374, 386],
    #               5: [386, 374, 380, 385], 6: [385, 380, 381, 384], 7: [384, 381, 382, 398], 8: [398, 382, 362]}
    # right_upper_lobe = {1: [247, 246, 33, 130], 2: [247, 30, 161, 246], 3: [30, 29, 160, 161], 4: [29, 27, 159, 160],
    #                     5: [27, 28, 158, 159], 6: [28, 56, 157, 158], 7: [56, 190, 173, 157], 8: [190, 243, 133, 173]}
    # right_lower_lobe = {1: [130, 33, 7, 25], 2: [7, 163, 110, 25], 3: [163, 144, 24, 110], 4: [144, 145, 23, 24],
    #                     5: [145, 153, 22, 23], 6: [153, 154, 26, 22], 7: [154, 155, 112, 26], 8: [155, 133, 243, 112]}
    # left_upper_lobe = {1: [359, 263, 466, 467], 2: [467, 466, 388, 260], 3: [260, 388, 387, 259],
    #                    4: [259, 387, 386, 257],
    #                    5: [257, 386, 385, 258], 6: [258, 385, 384, 286], 7: [286, 384, 398, 414],
    #                    8: [414, 398, 362, 463]}
    # left_lower_lobe = {1: [359, 263, 249, 255], 2: [255, 249, 390, 339], 3: [339, 390, 373, 254],
    #                    4: [254, 373, 374, 253],
    #                    5: [253, 374, 380, 252], 6: [252, 380, 381, 256], 7: [256, 381, 382, 341],
    #                    8: [341, 382, 362, 463]}

    success, img = cap.read()

    if not success:
        print("error opening camera")
        break

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    ### 32 imp body landmarks ###
    cv.imshow("Original", img)

    landmarks_dict = {}

    frequency = 200 + music_one_hot.index(1)*100

    if window_size == 60:
        speed_queue.put([frequency, 1, 0])

    if results.pose_landmarks:
        # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            # print("inner_data:", id, lm)
            # print("inner_data types:", type(id), type(lm))
            id_temp = int(id)

            if id_temp in main_face:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)

                landmarks_dict[id_temp] = (cx, cy, lm.z)

                ST_GCN_input[id_temp].append([lm.x, lm.y, lm.z])

        # print(landmarks_dict)

    cv.imshow("Landmarks", img)
    window_size -= 1

    if window_size == 0:
        mra_agent.train_short_memory(ST_GCN_input)

        music_one_hot = torch.tensor(music_one_hot).to(torch.float)
        ST_GCN_input = torch.tensor(ST_GCN_input)
        # print("ST_GCN_input:", ST_GCN_input)
        # print(ST_GCN_input)
        # print(ST_GCN_input.shape)

        frequency_index = mra_agent.get_music(ST_GCN_input, music_one_hot)
        print("freq_index:", frequency_index)
        music_one_hot = [0 for i in range(10)]
        music_one_hot[frequency_index] = 1

        ST_GCN_input = [[], [], [], [], [], [], [], [], []]

        window_size = 60

    # ### MP Face Landmarks
    # percents_re = []
    # percents_le = []
    #
    # img_for_eye = img.copy()
    #
    # flag_re = False
    # flag_le = False
    # flag_n = False
    #
    # right_eye = None
    # left_eye = None
    #
    # focus_points = [None, None, None, None, None]
    # #                ri, ro, li, lo, n
    # face_points = [None, None, None]
    # #              lm,  c, rm
    # bounds_right = [None, None, None, None]
    # #                 ur,   lr,   rc,   lc
    # bounds_left = [None, None, None, None]
    # #                 ur,   lr,   rc,   lc
    #
    # right_eye_main = []
    # left_eye_main = []
    #
    # with mp_face_mesh.FaceMesh(
    #         static_image_mode=True,
    #         min_detection_confidence=0.5) as face_mesh:
    #     if True:
    #         # Convert the BGR image to RGB before processing.
    #         results_2 = face_mesh.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    #
    #         # Print and draw face mesh landmarks on the image.
    #         if not results_2.multi_face_landmarks:
    #             continue
    #
    #         # print(len(results_2.multi_face_landmarks))
    #
    #         # annotated_image = img.copy()
    #         for face_landmarks in results_2.multi_face_landmarks:
    #             # mpDraw.draw_landmarks(
    #             #     image=img,
    #             #     landmark_list=face_landmarks,
    #             #     connections=mp_face_mesh.FACEMESH_CONTOURS,
    #             #     landmark_drawing_spec=drawing_spec,
    #             #     connection_drawing_spec=drawing_spec)
    #
    #             for _, data_point in enumerate(face_landmarks.landmark):
    #                 # print("data inner:", data_point.x, data_point.y)
    #                 h, w, c = img.shape
    #                 cx, cy = int(data_point.x * w), int(data_point.y * h)
    #
    #                 if int(_) == 168:
    #                     re_points.append([cx, cy])
    #                     le_points.append([cx, cy])
    #                     glabella = [cx, cy]
    #                 if int(_) == 133:
    #                     re_points.append([cx, cy])
    #                 if int(_) == 362:
    #                     le_points.append([cx, cy])
    #                 if int(_) == 195:
    #                     p_195 = [cx, cy]
    #                 if int(_) == 94:
    #                     p_94 = [cx, cy]
    #                 if int(_) == 4:
    #                     nose = [cx, cy]
    #
    #                 if int(_) == 223:
    #                     bounds_right[0] = [cx, cy]
    #                 if int(_) == 230:
    #                     bounds_right[1] = [cx, cy]
    #                 if int(_) == 35:
    #                     bounds_right[2] = [cx, cy]
    #                     focus_points[1] = [cx, cy]
    #                 if int(_) == 245:
    #                     bounds_right[3] = [cx, cy]
    #                     focus_points[0] = [cx, cy]
    #
    #                 if int(_) == 443:
    #                     bounds_left[0] = [cx, cy]
    #                 if int(_) == 450:
    #                     bounds_left[1] = [cx, cy]
    #                 if int(_) == 465:
    #                     bounds_left[2] = [cx, cy]
    #                     focus_points[2] = [cx, cy]
    #                 if int(_) == 265:
    #                     bounds_left[3] = [cx, cy]
    #                     focus_points[3] = [cx, cy]
    #
    #                 if int(_) == 0:
    #                     focus_points[4] = [cx, cy]
    #
    #                 if int(_) in right_eye_main_dict.keys():
    #                     right_eye_main_dict[int(_)] = [cx, cy]
    #                 if int(_) in left_eye_main_dict.keys():
    #                     left_eye_main_dict[int(_)] = [cx, cy]
    #
    #                 if int(_) in [130, 247, 30, 29, 27, 28, 56, 190, 243, 25, 110, 24, 23, 22, 26, 112]:
    #                     r_lobe_coords[int(_)] = [cx, cy]
    #
    #                 if int(_) in [463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341]:
    #                     l_lobe_coords[int(_)] = [cx, cy]
    #
    #         for cx, cy in bounds_right:
    #             cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED)
    #
    #         for cx, cy in bounds_left:
    #             cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)
    #
    # for key in right_eye_main_dict:
    #     right_eye_main.append(right_eye_main_dict[key])
    #
    # for key in left_eye_main_dict:
    #     left_eye_main.append(left_eye_main_dict[key])
    #
    # right_eye_main = np.array(right_eye_main, np.int32)
    # right_eye_main = right_eye_main.reshape((-1, 1, 2))
    #
    # left_eye_main = np.array(left_eye_main, np.int32)
    # left_eye_main = left_eye_main.reshape((-1, 1, 2))
    #
    # isClosed = True
    # color = (255, 0, 0)
    # thickness = 2
    #
    # img = cv.polylines(img, [right_eye_main], isClosed, color, thickness)
    # img = cv.polylines(img, [left_eye_main], isClosed, color, thickness)
    #
    # if bounds_right[0] is not None:
    #     flag_re = True
    #
    # if bounds_left[0] is not None:
    #     flag_le = True
    #
    # if flag_re is True:
    #     right_eye = img_for_eye[bounds_right[0][1]:bounds_right[1][1], bounds_right[2][0]:bounds_right[3][0]]
    #
    # if flag_le is True:
    #     left_eye = img_for_eye[bounds_left[0][1]:bounds_left[1][1], bounds_left[2][0]:bounds_left[3][0]]
    #
    # for key in regions_re.keys():
    #     temp = regions_re[key]
    #     temp_coords = []
    #
    #     for index in temp:
    #         temp_coords.append(right_eye_main_dict[index])
    #
    #     regions_re[key] = temp_coords
    #
    # for key in regions_le.keys():
    #     temp = regions_le[key]
    #     temp_coords = []
    #
    #     for index in temp:
    #         temp_coords.append(left_eye_main_dict[index])
    #
    #     regions_le[key] = temp_coords
    #
    # for key in right_upper_lobe.keys():
    #     temp = right_upper_lobe[key]
    #     temp_coords = []
    #
    #     print("second error:", key, temp)
    #
    #     for index in temp:
    #         print("second error:", index, r_lobe_coords)
    #
    #         if index in r_lobe_coords.keys():
    #             temp_coords.append(r_lobe_coords[index])
    #
    #         else:
    #             temp_coords.append(right_eye_main_dict[index])
    #
    #     right_upper_lobe[key] = temp_coords
    #
    # for key in right_lower_lobe.keys():
    #     temp = right_lower_lobe[key]
    #     temp_coords = []
    #
    #     for index in temp:
    #         if index in r_lobe_coords.keys():
    #             temp_coords.append(r_lobe_coords[index])
    #
    #         else:
    #             temp_coords.append(right_eye_main_dict[index])
    #
    #     right_lower_lobe[key] = temp_coords
    #
    # for key in left_upper_lobe.keys():
    #     temp = left_upper_lobe[key]
    #     temp_coords = []
    #
    #     for index in temp:
    #         if index in l_lobe_coords.keys():
    #             temp_coords.append(l_lobe_coords[index])
    #
    #         else:
    #             temp_coords.append(left_eye_main_dict[index])
    #
    #     left_upper_lobe[key] = temp_coords
    #
    # for key in left_lower_lobe.keys():
    #     temp = left_lower_lobe[key]
    #     temp_coords = []
    #
    #     for index in temp:
    #         if index in l_lobe_coords.keys():
    #             temp_coords.append(l_lobe_coords[index])
    #
    #         else:
    #             temp_coords.append(left_eye_main_dict[index])
    #
    #     left_lower_lobe[key] = temp_coords
    #
    # print("regions_re:", regions_re)
    # print("regions_le:", regions_le)
    # print("right_upper_lobe:", right_upper_lobe)
    # print("right_lower_lobe:", right_lower_lobe)
    # print("left_upper_lobe:", left_upper_lobe)
    # print("left_lower_lobe:", left_lower_lobe)
    #
    # rect_right = cv.boundingRect(right_eye_main)
    # x, y, w, h = rect_right
    # cropped_re = img[y:y + h, x:x + w].copy()
    # right_eye_main = right_eye_main - right_eye_main.min(axis=0)
    # mask_re = np.zeros(cropped_re.shape[:2], np.uint8)
    # cv.drawContours(mask_re, [right_eye_main], -1, (255, 255, 255), -1, cv.LINE_AA)
    # dst_re = cv.bitwise_and(cropped_re, cropped_re, mask=mask_re)
    # bg = np.ones_like(cropped_re, np.uint8) * 255
    # cv.bitwise_not(bg, bg, mask=mask_re)
    # dst_re = bg + dst_re
    #
    # for key in regions_re.keys():
    #     temp = regions_re[key]
    #     temp_t = []
    #
    #     for i in range(len(temp)):
    #         temp_t.append([temp[i][0] - x, temp[i][1] - y])
    #
    #     regions_re[key] = temp_t
    #
    # for key in right_upper_lobe.keys():
    #     temp = right_upper_lobe[key]
    #     temp_t = []
    #
    #     for i in range(len(temp)):
    #         temp_t.append([temp[i][0] - x, temp[i][1] - y])
    #
    #     right_upper_lobe[key] = temp_t
    #
    # for key in right_lower_lobe.keys():
    #     temp = right_lower_lobe[key]
    #     temp_t = []
    #
    #     for i in range(len(temp)):
    #         temp_t.append([temp[i][0] - x, temp[i][1] - y])
    #
    #     right_lower_lobe[key] = temp_t
    #
    # for id in landmarks_dict.keys():
    #     cx, cy = landmarks_dict[id][0:2]
    #     cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
    #
    # rect_left = cv.boundingRect(left_eye_main)
    # x, y, w, h = rect_left
    # cropped_le = img[y:y + h, x:x + w].copy()
    # left_eye_main = left_eye_main - left_eye_main.min(axis=0)
    # mask_le = np.zeros(cropped_le.shape[:2], np.uint8)
    # cv.drawContours(mask_le, [left_eye_main], -1, (255, 255, 255), -1, cv.LINE_AA)
    # dst_le = cv.bitwise_and(cropped_le, cropped_le, mask=mask_le)
    # bg = np.ones_like(cropped_le, np.uint8) * 255
    # cv.bitwise_not(bg, bg, mask=mask_le)
    # dst_le = bg + dst_le
    #
    # for key in regions_le.keys():
    #     temp = regions_le[key]
    #     temp_t = []
    #
    #     for i in range(len(temp)):
    #         temp_t.append([temp[i][0] - x, temp[i][1] - y])
    #
    #     regions_le[key] = temp_t
    #
    # for key in left_upper_lobe.keys():
    #     temp = left_upper_lobe[key]
    #     temp_t = []
    #
    #     for i in range(len(temp)):
    #         temp_t.append([temp[i][0] - x, temp[i][1] - y])
    #
    #     left_upper_lobe[key] = temp_t
    #
    # for key in left_lower_lobe.keys():
    #     temp = left_lower_lobe[key]
    #     temp_t = []
    #
    #     for i in range(len(temp)):
    #         temp_t.append([temp[i][0] - x, temp[i][1] - y])
    #
    #     left_lower_lobe[key] = temp_t
    #
    # kernel = np.array([[0, -1, 0], [-1, 6.5, -1], [0, -1, 0]])
    #
    # thresh = 0
    #
    # dst_re = cv.cvtColor(dst_re, cv.COLOR_BGR2GRAY)
    # # dst_re = cv.threshold(dst_re, thresh, 255, cv.THRESH_BINARY)[1]
    # dst_re = cv.filter2D(dst_re, -1, kernel)
    # dst_re = cv.medianBlur(dst_re, 5)
    # # dst_re = cv.threshold(dst_re, thresh, 255, cv.THRESH_BINARY)[1]
    # # dst_re = cv.Canny(image=dst_re, threshold1=50, threshold2=200)
    #
    # dst_le = cv.cvtColor(dst_le, cv.COLOR_BGR2GRAY)
    # dst_le = cv.filter2D(dst_le, -1, kernel)
    # dst_le = cv.medianBlur(dst_le, 5)
    # # dst_le = cv.threshold(dst_le, thresh, 255, cv.THRESH_BINARY)[1]
    # # dst_le = cv.Sobel(src=dst_le, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
    # # dst_le = cv.Canny(image=dst_le, threshold1=50, threshold2=200)
    #
    # ar_re = []
    # ar_le = []
    #
    # for key in regions_re.keys():
    #     # mask_re = np.zeros_like(dst_re)
    #     #
    #     list_coords = np.array(regions_re[key])
    #     # # print("list_coords_re:", list_coords)
    #     # cv.fillPoly(mask_re, [list_coords], 255)
    #     # mask_re = cv.threshold(mask_re, 200, 255, cv.THRESH_BINARY)[1]
    #     # masked_image = cv.bitwise_and(dst_re, mask_re)
    #     masked_image = dst_re
    #
    #     # black_pixel_count_re = 0
    #     # white_pixel_count_re = 0
    #
    #     pixel_count = 0
    #     values = 0
    #
    #     for y in range(masked_image.shape[0]):
    #         for x in range(masked_image.shape[1]):
    #             if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
    #                 pixel_count += 1
    #                 values += masked_image[y, x]
    #
    #                 # if masked_image[y, x] < 20:
    #                 #     black_pixel_count_re += 1
    #                 #
    #                 # else:
    #                 #     white_pixel_count_re += 1
    #
    #     # percents_re.append(round(black_pixel_count_re / (black_pixel_count_re + white_pixel_count_re), 2))
    #     percents_re.append(round((values / pixel_count), 2))
    #     text_temp = ":" + str(pixel_count) + ":"
    #     ar_re.append(text_temp)
    #
    # for key in regions_le.keys():
    #     # mask_le = np.zeros_like(dst_le)
    #     #
    #     list_coords = np.array(regions_le[key])
    #     # # print("list_coords_le:", list_coords)
    #     # cv.fillPoly(mask_le, [list_coords], 255)
    #     # # cv.drawContours(mask_le, [list_coords], -1, 255, 1, thickness=cv.LINE_AA)
    #     # # mask_le = cv.threshold(mask_le, 200, 255, cv.THRESH_BINARY)[1]
    #     # masked_image = cv.bitwise_and(dst_le, mask_le)
    #     masked_image = dst_le
    #
    #     # black_pixel_count_le = 0
    #     # white_pixel_count_le = 0
    #
    #     pixel_count = 0
    #     values = 0
    #
    #     for y in range(masked_image.shape[0]):
    #         for x in range(masked_image.shape[1]):
    #             if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
    #                 pixel_count += 1
    #                 values += masked_image[y, x]
    #
    #                 # if masked_image[y, x] < 20:
    #                 #     black_pixel_count_le += 1
    #                 #
    #                 # else:
    #                 #     white_pixel_count_le += 1
    #
    #     # percents_le.append(round(black_pixel_count_le / (black_pixel_count_le + white_pixel_count_le), 2))
    #     percents_le.append(round((values / pixel_count), 2))
    #     text_temp = ":" + str(pixel_count) + ":"
    #     ar_le.append(text_temp)
    #
    # index = 0
    # for key in right_upper_lobe.keys():
    #     list_coords = np.array(right_upper_lobe[key])
    #     masked_image = dst_re
    #     pixel_count = 0
    #
    #     for y in range(masked_image.shape[0]):
    #         for x in range(masked_image.shape[1]):
    #             if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
    #                 pixel_count += 1
    #
    #     ar_re[index] = str(pixel_count) + ar_re[index]
    #     index += 1
    #
    # index = 0
    # for key in right_lower_lobe.keys():
    #     list_coords = np.array(right_lower_lobe[key])
    #     masked_image = dst_re
    #     pixel_count = 0
    #
    #     for y in range(masked_image.shape[0]):
    #         for x in range(masked_image.shape[1]):
    #             if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
    #                 pixel_count += 1
    #
    #     ar_re[index] = ar_re[index] + str(pixel_count)
    #     index += 1
    #
    # index = 0
    # for key in left_upper_lobe.keys():
    #     list_coords = np.array(left_upper_lobe[key])
    #     masked_image = dst_re
    #     pixel_count = 0
    #
    #     for y in range(masked_image.shape[0]):
    #         for x in range(masked_image.shape[1]):
    #             if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
    #                 pixel_count += 1
    #
    #     ar_le[index] = str(pixel_count) + ar_le[index]
    #     index += 1
    #
    # index = 0
    # for key in left_lower_lobe.keys():
    #     list_coords = np.array(left_lower_lobe[key])
    #     masked_image = dst_re
    #     pixel_count = 0
    #
    #     for y in range(masked_image.shape[0]):
    #         for x in range(masked_image.shape[1]):
    #             if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
    #                 pixel_count += 1
    #
    #     ar_le[index] = ar_le[index] + str(pixel_count)
    #     index += 1
    #
    # print("percents_re:", percents_re)
    # print("percents_le:", percents_le)
    #
    # # contours, _ = cv.findContours(dst_le, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    # # contour_image = np.zeros_like(dst_le)
    # # cv.drawContours(contour_image, contours, -1, (255, 255, 255), 2)  # Draw all contours with thickness 2
    # # # print("_:", _)
    #
    # cv.imshow("right_eye cropped", dst_re)
    # cv.imshow("left_eye cropped", dst_le)
    # # cv.imshow("left_eye contoured", contour_image)
    #
    # cv.imshow("Face detailed", img)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
