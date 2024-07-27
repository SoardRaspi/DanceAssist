import torch

from Instructor_PyGame import changer, main_function

import time
import pygame
import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import csv
import DAUM1 as model_v1
from cubical import dtw_score
from music_generator import *
from threading import Event, Thread
from PIL import Image

import threading
import queue
import multiprocessing as multip

import playsound
import librosa

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Music_Generator')))

from Music_Generator import music_agent

filename = "right_arm_data.csv"
fields = ["sa", "ea", "wa"]
right_arm_data = []

model_v1.filename = "movement_correlation.csv"
model_v1.print_function()

# Load the audio file
path = "musics"
audio_files = os.listdir(path)

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

right_arm_sa = []
right_arm_ea = []
right_arm_wa = []

window_size = 10
window = window_size

# window_increment = 10
window_increment = window

matrix_shown = []

matrix_shown_full = []
matrix_actual_full = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)

    for _, lines in enumerate(csvreader):
        if _ > 0:
            sa, ea, wa = lines
            right_arm_sa.append(float(sa))
            right_arm_ea.append(float(ea))
            right_arm_wa.append(float(wa))

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
stability_rest_down = [0, 0, 0]
stabiltiy_rest_up = [0, 0, 0]

#______Camera Init_____
cap = cv.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


# cap = cv.VideoCapture(0)

pTime = 0

#______Camera Init Done_____

pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = int(0.8 * SCREEN_WIDTH)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Instructor")

head = pygame.Rect((375, 135, 50, 50))
body = pygame.Rect((350, 185, 100, 100))
limb1_length = 50
limb2_length = 50
limb3_length = 25

joints = {"rs": (350, 185), "re": (), "rw": (), "r_ft": (), "ls": (450, 185), "le": (), "lw": (), "l_ft": (),
          "rh": (375, 285), "rk": (), "ra": (), "r_tt": (), "lh": (425, 285), "lk": (), "la": (), "l_tt": ()}
angles = {"rs": 0, "re": 0, "rw": 0, "ls": 0, "le": 0, "lw": 0,
          "rh": 0, "rk": 0, "ra": 0, "lh": 0, "lk": 0, "la": 0}

joints_2 = {"rs": (350, 185), "re": (), "rw": (), "r_ft": (), "ls": (450, 185), "le": (), "lw": (), "l_ft": (),
            "rh": (375, 285), "rk": (), "ra": (), "r_tt": (), "lh": (425, 285), "lk": (), "la": (), "l_tt": ()}
angles_2 = {"rs": 0, "re": 0, "rw": 0, "ls": 0, "le": 0, "lw": 0,
            "rh": 0, "rk": 0, "ra": 0, "lh": 0, "lk": 0, "la": 0}

c_obj = changer(joints_dir=joints, joints_dir_2=joints_2, angles_dir=angles, angles_dir_2=angles_2,
                l1_l=75, l2_l=75, l3_l=25,
                l1_l_2=75, l2_l_2=75, l3_l_2=25,
                screen=screen, head=head, body=body)

motion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         rs, ls, re, le, rw, lw, rh, lh, rk, lk, ra, la
#         12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27

# Task 1: Make a simple imitator ---------- DONE
# Task 2: To make the patient align with the initial position

def calculate_angle(p1, p2, p3):
    # ang1 = np.arctan2(*p1[::-1])
    # ang2 = np.arctan2(*p2[::-1]
    # return np.rad2deg((ang1 - ang2) % (2 * np.pi))

    # Calculate the vectors between the points
    vector1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    vector2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # Calculate the dot product
    dot_product = np.dot(vector1, vector2)

    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the angle in radians
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))

    # Convert radians to degrees
    angle_deg = np.degrees(angle_rad)

    if p3[0] < p1[0]:
        return -angle_deg

    return angle_deg

def pose_similarity(a1, a2):
    # a1 = [.25*a1[0], .5*a1[1], 1*a1[2]]
    # a2 = [.25*a2[0], .5*a2[1], 1*a2[2]]

    a1 = a1 / np.linalg.norm(a1)
    a2 = a2 / np.linalg.norm(a2)

    # diff = [a1[0]-a2[0], a1[1]-a2[1], a1[2]-a2[2]]

    dot_product = np.dot(a1, a2)
    norm_v1 = np.linalg.norm(a1)
    norm_v2 = np.linalg.norm(a2)
    # norm_diff = np.linalg.norm(diff)

    # sum_diff = np.absolute(diff[0]) + np.absolute(diff[1]) + np.absolute(diff[2])

    # return np.round(dot_product / (norm_v1 * norm_v2), decimals=12)
    return np.round(np.mean((a1 - a2) ** 2), decimals=12)
    # return np.round(sum_diff / norm_diff, decimals=12)

flag_rest_down = False
flag_rest_up = False

# delay_start = 5
# for t in range(1, delay_start + 1):
#     print("t:", t)
#     time.sleep(1)

window_i = 1

dtw_score_printing = -1

frequency = 1000  # Frequency of the tick sound in Hz
duration = 0.1  # Duration of the tick sound in seconds
bpm = 120  # Beats per minute
interval = 60 / bpm  # Interval between ticks in seconds

bpms = [100, 105, 110, 115, 120, 125, 130, 135, 140]
freqs = [i*10 for i in range(2, 21)]

storage_location = "correlation_images_3"
music_params_csv = "correlation_images_3/music_params_1.csv"

music_params_list = [["freq", "bpm"]]
# freq, bpm

t_2_event = Event()
# def t_2_func(frequency, duration, interval, queue):
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


def map_value(value, from_min, from_max, to_min, to_max):
    value = max(0, min(1, value))
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min

# def video_input_and_facial_landmarks(frames_queue, window_size=1, motion_seq_queue=None, camera_index=0):
#     cap = cv.VideoCapture(camera_index)
#     frames_window_item = []
#     matrix_actual_perform = []
#     counter_inner = 0
#
#     # matrix_shown_perform = motion_seq_queue
#
#     # for perform_row in matrix_shown_perform:
#     #     # print("perform_row", perform_row)
#     #
#     #     screen.fill((0, 0, 0))
#     #     pygame.draw.rect(screen, (255, 255, 255), head, 3)
#     #     pygame.draw.rect(screen, (255, 255, 255), body, 3)
#     #     motion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     #
#     #     # # # # #
#     #
#     #     pygame.display.update()
#
#     run = True
#     while run:
#         motion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#         success, img = cap.read()
#         imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#
#         if not success:
#             print('failed to capture frame')
#             break
#
#         if counter_inner == window_size:
#             if not frames_queue.full():
#                 frames_queue.put([frames_window_item[0], matrix_actual_perform[0]])
#                 print([frames_window_item[0], matrix_actual_perform[0]])
#
#                 frames_window_item = []
#                 matrix_actual_perform = []
#
#             counter_inner = 0
#
#         results = pose.process(imgRGB)
#         cv.imshow("Image", img)
#
#         k = cv.waitKey(1)
#         if k == ord('q'):
#             break
#
#         if results.pose_landmarks:
#             mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
#             joints_dict = {}
#
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 # print("inner_data:", id, lm)
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
#
#                 id_temp = int(id)
#
#                 joints_dict[id_temp] = (cx, cy)
#
#             if (12 in joints_dict) and (14 in joints_dict):
#                 if 24 in joints_dict:
#                     angle_temp = calculate_angle(joints_dict[14], joints_dict[12], joints_dict[24])
#
#                 angle_temp = calculate_angle(joints_dict[14], joints_dict[12],
#                                              (joints_dict[12][0], joints_dict[12][1] + 1))
#                 motion[0] = angle_temp
#
#             if (14 in joints_dict) and (16 in joints_dict):
#                 angle_temp = calculate_angle(joints_dict[16], joints_dict[14],
#                                              (joints_dict[14][0], joints_dict[14][1] + 1))
#                 motion[2] = angle_temp
#
#             if (16 in joints_dict) and (20 in joints_dict):
#                 angle_temp = calculate_angle(joints_dict[20], joints_dict[16],
#                                              (joints_dict[16][0], joints_dict[16][1] + 1))
#                 motion[4] = angle_temp
#
#             right_arm_data.append([motion[0], motion[2], motion[4]])
#
#             if (11 in joints_dict) and (13 in joints_dict):
#                 if 23 in joints_dict:
#                     angle_temp = calculate_angle(joints_dict[13], joints_dict[12], joints_dict[23])
#
#                 angle_temp = calculate_angle(joints_dict[13], joints_dict[11],
#                                              (joints_dict[11][0], joints_dict[11][1] + 1))
#                 motion[1] = angle_temp
#
#             if (13 in joints_dict) and (15 in joints_dict):
#                 angle_temp = calculate_angle(joints_dict[15], joints_dict[13],
#                                              (joints_dict[13][0], joints_dict[13][1] + 1))
#                 motion[3] = angle_temp
#
#             if (15 in joints_dict) and (19 in joints_dict):
#                 angle_temp = calculate_angle(joints_dict[19], joints_dict[15],
#                                              (joints_dict[15][0], joints_dict[15][1] + 1))
#                 motion[5] = angle_temp
#
#         cTime = time.time()
#         # fps = 1 / (cTime - pTime)
#         pTime = cTime
#
#         sa_temp = np.ceil(((motion[0] + 180) * 256) / 360).astype(np.uint8)
#         ea_temp = np.ceil(((motion[2] + 180) * 256) / 360).astype(np.uint8)
#         wa_temp = np.ceil(((motion[4] + 180) * 256) / 360).astype(np.uint8)
#         denom_temp = np.sqrt((sa_temp ** 2) + (ea_temp ** 2) + (wa_temp ** 2))
#
#         t = [sa_temp / denom_temp, ea_temp / denom_temp, wa_temp / denom_temp]
#         matrix_actual_perform.append(t)
#
#         # cv.putText(img, str(dtw_score__), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
#         cv.imshow("Image", img)
#         frames_window_item.append(motion)
#
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
#
#         counter_inner += 1
#
#     cap.release()
#     cv.destroyAllWindows()

# def main_function(queue_video_main_function):
if True:
    start_curr_window = 0
    end_curr_window = start_curr_window + window_size
    motion_seq_length = len(right_arm_sa)
    run = True

    # try:
    if True:
        agent = music_agent.Agent()
        speed_queue = queue.Queue()

        threading.Thread(target=t_2_func, args=(speed_queue,)).start()

        print("inside main after threading")

        window_num = 0
        last_music_played = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # while end_curr_window <= motion_seq_length:
        while run:
            print("#########################################")
            print("Window Number: ", window_num)
            print("#########################################")

            matrix_shown_perform = []
            matrix_actual_perform = []
            matrix_actual = []
            matrix_shown_compare = []
            matrix_shown = []

            for index in range(start_curr_window, end_curr_window):
                index_inner = index % motion_seq_length
                print("index:", index_inner, start_curr_window, end_curr_window)

                sa_temp = np.ceil(((right_arm_sa[index_inner] + 180) * 256) / 360).astype(np.uint8)
                ea_temp = np.ceil(((right_arm_ea[index_inner] + 180) * 256) / 360).astype(np.uint8)
                wa_temp = np.ceil(((right_arm_wa[index_inner] + 180) * 256) / 360).astype(np.uint8)
                denom_temp = np.sqrt((sa_temp ** 2) + (ea_temp ** 2) + (wa_temp ** 2))

                t = [sa_temp / denom_temp, ea_temp / denom_temp, wa_temp / denom_temp]
                matrix_shown_compare.append(t)
                matrix_shown.append(torch.tensor([sa_temp, ea_temp, wa_temp]))
                matrix_shown_perform.append([right_arm_sa[index_inner],
                                             right_arm_ea[index_inner],
                                             right_arm_wa[index_inner]])

                # matrix_shown_full.append([sa_temp, ea_temp, wa_temp])

            ############### PROCESSING PARAMETERS ###############
            # get old state
            # print("matrix_shown right after:", matrix_shown)
            # matrix_shown = agent.get_state(matrix_shown)

            matrix_shown_full = [torch.stack(matrix_shown)]
            # print("matrix_shown_full 1:", matrix_shown_full)

            matrix_shown_full = torch.stack(matrix_shown_full)
            # print("matrix_shown_full 2:", matrix_shown_full)

            motion_seq_shown = matrix_shown_full
            # motion_seq_shown = matrix_shown

            # print("motion_seq_shown:", motion_seq_shown)
            print("matrix_shown_perform:", matrix_shown_perform)

            # get move
            print("motion_seq_shown:", motion_seq_shown)
            frequency = agent.get_music(motion_seq_shown)
            print("checkpoint 1:", frequency)
            frequency = np.argmax(frequency)

            # frequency = map_value(frequency, 0, 1, 20, 2000)
            frequency = 200 + 220*frequency
            print("checkpoint 1_2:", frequency)

            bpm = 0

            speed_queue.put([frequency, 1, bpm])

            # if done:
            #     # train long memory, plot result
            #     game.reset()
            #     agent.n_games += 1
            #     agent.train_long_memory()
            #
            #     if score > record:
            #         record = score
            #         agent.model.save()
            #
            #     print('Game', agent.n_games, 'Score', score, 'Record:', record)
            #
            #     plot_scores.append(score)
            #     total_score += score
            #     mean_score = total_score / agent.n_games
            #     plot_mean_scores.append(mean_score)
            #     plot(plot_scores, plot_mean_scores)
            ############## PROCESSING PARAMETERS END ###############

            for row_i, perform_row in enumerate(matrix_shown_perform):
                print("row_i, perform_row", row_i, perform_row)

                screen.fill((0, 0, 0))
                pygame.draw.rect(screen, (255, 255, 255), head, 3)
                pygame.draw.rect(screen, (255, 255, 255), body, 3)

                # motion_done_for_current_shown, matrix_actual_perform_row = queue_video_main_function.get()
                # matrix_actual_perform.append(matrix_actual_perform_row)
                #
                # print("motion_done_for_current_shown:", [motion_done_for_current_shown, matrix_actual_perform_row])
                #
                # # motion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                # motion = motion_done_for_current_shown

                motion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                success, img = cap.read()
                imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                if not success:
                    print('failed to capture frame')
                    break

                # if counter_inner == window_size:
                #     if not frames_queue.full():
                #         frames_queue.put([frames_window_item[0], matrix_actual_perform[0]])
                #         print([frames_window_item[0], matrix_actual_perform[0]])
                #
                #         frames_window_item = []
                #         matrix_actual_perform = []
                #
                #     counter_inner = 0

                results = pose.process(imgRGB)
                # cv.imshow("Image", img)

                k = cv.waitKey(1)
                if k == ord('q'):
                    break

                if results.pose_landmarks:
                    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                    joints_dict = {}

                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        h, w, c = img.shape
                        # print("inner_data:", id, lm)
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)

                        id_temp = int(id)

                        joints_dict[id_temp] = (cx, cy)

                    if (12 in joints_dict) and (14 in joints_dict):
                        if 24 in joints_dict:
                            angle_temp = calculate_angle(joints_dict[14], joints_dict[12], joints_dict[24])

                        angle_temp = calculate_angle(joints_dict[14], joints_dict[12],
                                                     (joints_dict[12][0], joints_dict[12][1] + 1))
                        motion[0] = angle_temp

                    if (14 in joints_dict) and (16 in joints_dict):
                        angle_temp = calculate_angle(joints_dict[16], joints_dict[14],
                                                     (joints_dict[14][0], joints_dict[14][1] + 1))
                        motion[2] = angle_temp

                    if (16 in joints_dict) and (20 in joints_dict):
                        angle_temp = calculate_angle(joints_dict[20], joints_dict[16],
                                                     (joints_dict[16][0], joints_dict[16][1] + 1))
                        motion[4] = angle_temp

                    right_arm_data.append([motion[0], motion[2], motion[4]])

                    if (11 in joints_dict) and (13 in joints_dict):
                        if 23 in joints_dict:
                            angle_temp = calculate_angle(joints_dict[13], joints_dict[12], joints_dict[23])

                        angle_temp = calculate_angle(joints_dict[13], joints_dict[11],
                                                     (joints_dict[11][0], joints_dict[11][1] + 1))
                        motion[1] = angle_temp

                    if (13 in joints_dict) and (15 in joints_dict):
                        angle_temp = calculate_angle(joints_dict[15], joints_dict[13],
                                                     (joints_dict[13][0], joints_dict[13][1] + 1))
                        motion[3] = angle_temp

                    if (15 in joints_dict) and (19 in joints_dict):
                        angle_temp = calculate_angle(joints_dict[19], joints_dict[15],
                                                     (joints_dict[15][0], joints_dict[15][1] + 1))
                        motion[5] = angle_temp

                cTime = time.time()
                # fps = 1 / (cTime - pTime)
                pTime = cTime

                sa_temp = np.ceil(((motion[0] + 180) * 256) / 360).astype(np.uint8)
                ea_temp = np.ceil(((motion[2] + 180) * 256) / 360).astype(np.uint8)
                wa_temp = np.ceil(((motion[4] + 180) * 256) / 360).astype(np.uint8)
                denom_temp = np.sqrt((sa_temp ** 2) + (ea_temp ** 2) + (wa_temp ** 2))

                t = [sa_temp / denom_temp, ea_temp / denom_temp, wa_temp / denom_temp]
                matrix_actual_perform.append(t)
                matrix_actual.append(torch.tensor([sa_temp, ea_temp, wa_temp]))

                # cv.putText(img, str(dtw_score__), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
                cv.imshow("Image", img)
                # frames_window_item.append(motion)

                # counter_inner += 1

                # # # # #
                color_polygon = (0, 0, 0)

                c_obj.draw()
                c_obj.change(motion[0], motion[1], motion[2], motion[3], motion[4], motion[5],
                             motion[6], motion[7], motion[8], motion[9], motion[10], motion[11],
                             perform_row[0], 0,
                             perform_row[1], 0,
                             perform_row[2], 0,
                             0, 0, 0, 0, 0, 0,
                             True, color_polygon)

                pygame.display.update()

            # # # # # # # # # #

            # perform move and get new state
            # reward, done, score = game.play_step(final_move)
            # frequency = freqs[freq_index - 1]
            # print("Enter BPM: ")
            # bpm = int(input())
            bpm = 0
            # frequency = 0

            music_test = [torch.tensor(frequency), torch.tensor(bpm)]
            music_test = torch.stack(music_test)

            # DTW_pred = model.use_model(motion_seq_shown, music_test)
            matrix_shown_compare = np.array(matrix_shown_compare)
            matrix_actual_perform = np.array(matrix_actual_perform)

            print("for dtw_score", matrix_shown_compare, matrix_actual_perform)

            dtw_score__ = dtw_score(matrix_shown_compare / 256, matrix_actual_perform / 256)
            reward = 1000 - abs(1000 * dtw_score__)

            matrix_actual_full = [torch.stack(matrix_actual)]
            matrix_actual_full = torch.stack(matrix_actual_full)
            motion_seq_done = matrix_actual_full

            # train short memory
            agent.train_short_memory(motion_seq_shown, frequency, reward, motion_seq_done)

            # remember
            agent.remember(motion_seq_shown, frequency, reward)

            start_curr_window += window_increment
            end_curr_window += window_increment

            window_num += 1

            print("iteration end:", dtw_score__, reward)

            if cv.waitKey(1) & 0xFF == ord('q'):
                pygame.quit()
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    run = False

    # except Exception as e:
    #     print("Exception in main:", e)

    pygame.quit()


# if __name__ == "__main__":
#     queue_video = multip.Queue(maxsize=20)
#
#     p1 = multip.Process(target=video_input_and_facial_landmarks, args=(queue_video,))
#     p2 = multip.Process(target=main_function, args=(queue_video,))
#
#     # Start both processes
#     p1.start()
#     p2.start()
#
#     # Wait for both processes to complete
#     p1.join()
#     p2.join()
