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

import G1_from_DAUM1 as model

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

matrix_shown = []
matrix_actual = []

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

length_angles_motion = len(right_arm_sa)
length_angles_motion_counter = 0

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

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


cap = cv.VideoCapture(0)

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

        if frequency == -1:
            break

        # play_tick(frequency, duration, 60 / bpm)
        play_tick(frequency, duration, 60 / 60)

run = True
try:
    # t_2 = Thread(target=t_2_func, args=(frequency, duration, interval,))
    # t_2.start()

    speed_queue = queue.Queue()
    threading.Thread(target=t_2_func, args=(speed_queue,)).start()

    while run:
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0

        # print("checkpoint 1")
        agent = music_agent.Agent()
        print("window:", window)
        # print("checkpoint 2")

        if window == 0:
            window = window_size

            # # print("checkpoint 3")
            print("temp music params:", [frequency, bpm])
            # # print("checkpoint 4")
            speed_queue.put([frequency, duration, bpm])

            matrix_shown_padded = []
            padding = 3

            for iii in range((window - padding) // 2):
                matrix_shown_padded.append(matrix_shown[padding + iii])

            for iii in range(padding):
                matrix_shown_padded.append(matrix_shown[((window - padding) // 2) + padding])

            for iii in range(window - padding - ((window - padding) // 2)):
                matrix_shown_padded.append(matrix_shown[((window - padding) // 2) + padding + iii])

            # model_v1.calc_correlation(matrix_shown, matrix_actual)
            # correlation_matrix_return = model_v1.calc_correlation(matrix_shown, matrix_actual)
            correlation_matrix_return = model_v1.calc_correlation(matrix_shown, matrix_shown_padded)
            # print("correlation_matrix_return ", window_i)

            data_min = correlation_matrix_return.min()
            data_max = correlation_matrix_return.max()
            data_normalized = (correlation_matrix_return - data_min) / (data_max - data_min)

            grayscale_data = (data_normalized * 255).astype(np.uint8)

            # print(grayscale_data)

            # # Save the grayscale image using Pillow
            # image = Image.fromarray(grayscale_data, mode='L')  # 'L' mode is for grayscale
            # image_name = storage_location + "/corr_matrix_" + str(window_i) + ".png"
            # image.save(image_name)

            matrix_shown = np.array(matrix_shown)
            # print(matrix_shown_full)

            image = Image.fromarray(matrix_shown, mode='L')  # 'L' mode is for grayscale
            # image_name = storage_location + "/shown_seq_" + str(window_i) + ".png"
            image_name = storage_location + "/part_shown_seq_" + str(1) + ".png"
            image.save(image_name)

            matrix_actual = np.array(matrix_actual)

            image = Image.fromarray(matrix_actual, mode='L')  # 'L' mode is for grayscale
            # image_name = storage_location + "/done_seq_" + str(window_i) + ".png"
            image_name = storage_location + "/part_done_seq_" + str(1) + ".png"
            image.save(image_name)

            music_params_list.append([frequency, bpm])

            dtw_score__ = dtw_score(matrix_shown / 256, matrix_actual / 256)
            print("dtw score: ", window_i, dtw_score__, matrix_shown)
            dtw_score_printing = dtw_score__

            window_i += 1

            # freq_index = np.random.random_integers(1, len(freqs))
            # bpm_index = np.random.random_integers(1, len(bpms))

            # matrix_test_full = []
            # for row in matrix_shown:
            #     sa_temp = np.ceil(((row[0] + 180) * 256) / 360).astype(np.uint8)
            #     ea_temp = np.ceil(((row[1] + 180) * 256) / 360).astype(np.uint8)
            #     wa_temp = np.ceil(((row[2] + 180) * 256) / 360).astype(np.uint8)
            #     denom_temp = np.sqrt((sa_temp ** 2) + (ea_temp ** 2) + (wa_temp ** 2))
            #
            #     matrix_test_full.append(torch.tensor([sa_temp, ea_temp, wa_temp]))
            #
            # matrix_test_full = [torch.stack(matrix_test_full)]
            # matrix_test_full = torch.stack(matrix_test_full)
            #
            # print("matrix_test_full:", matrix_test_full)

            print("matrix_test_full:", matrix_shown)

            matrix_shown_test = []
            for row in matrix_shown:
                matrix_shown_test.append(torch.tensor(row))
            matrix_shown = matrix_shown_test
            del matrix_shown_test

            ############### PROCESSING PARAMETERS ###############
            # get old state
            matrix_shown_full = [torch.stack(matrix_shown)]
            print("checkpoint 11:", matrix_shown_full)

            matrix_shown_full = torch.stack(matrix_shown_full)
            print("checkpoint 12:", matrix_shown_full)

            motion_seq_shown = matrix_shown_full
            print("checkpoint 13:", motion_seq_shown)

            # get move
            print("checkpoint 1:", motion_seq_shown)
            frequency = agent.get_music(motion_seq_shown)
            print("checkpoint 2:", "just after agent.get_music()")

            # perform move and get new state
            # reward, done, score = game.play_step(final_move)
            # frequency = freqs[freq_index - 1]
            # print("Enter BPM: ")
            # bpm = int(input())
            bpm = 0

            print("checkpoint 3:", frequency)

            music_test = [torch.tensor(frequency), torch.tensor(bpm)]
            music_test = torch.stack(music_test)

            # DTW_pred = model.use_model(motion_seq_shown, music_test)
            reward = dtw_score__

            # train short memory
            # agent.train_short_memory(motion_seq_shown, frequency, reward)

            # remember
            agent.remember(motion_seq_shown, frequency, reward)

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
            ############### PROCESSING PARAMETERS END ###############

            # print("temp music params:", [frequency, bpm])
            #
            # speed_queue.put([frequency, duration, bpm])

            # matrix_shown_full = []
            # matrix_actual_full = []

            matrix_shown = []
            matrix_actual = []

            # # print("checkpoint 3")
            print("temp music params 2:", [frequency, bpm])
            # # print("checkpoint 4")
            speed_queue.put([frequency, duration, bpm])

        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), head, 3)
        pygame.draw.rect(screen, (255, 255, 255), body, 3)

        motion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # print("checkpoint 5")

        # time.sleep(2)

        success, img = cap.read()
        # print("checkpoint 6")

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # print("checkpoint 7")

        results = pose.process(imgRGB)
        # print("checkpoint 8")
        # print(results.pose_landmarks)

        # if flag_rest_down is False:
        #     cv.putText(img, "lower your hands in a comfortable position.", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        #     cv.putText(img, "hold up your hands in a comfortable position.", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

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
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # pose_sim = pose_similarity([motion[0], motion[2], motion[4]],
        #                            [right_arm_sa[length_angles_motion_counter],
        #                             right_arm_ea[length_angles_motion_counter],
        #                             right_arm_wa[length_angles_motion_counter]])
        #
        # # cv.putText(img, str(np.round_([motion[0], motion[2], motion[4]], decimals=3)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        # cv.putText(img, str(pose_sim), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        cv.putText(img, str(dtw_score_printing), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        cv.imshow("Image", img)

        # print("checkpoint 9")
        # # cv.waitKey(1)
        #
        # color_polygon = (0, 127, 0)
        color_polygon = (0, 0, 0)
        # if pose_sim > 0.0015:
        #     color_polygon = (127, 0, 0)

        c_obj.draw()
        c_obj.change(motion[0], motion[1], motion[2], motion[3], motion[4], motion[5],
                     motion[6], motion[7], motion[8], motion[9], motion[10], motion[11],
                     right_arm_sa[length_angles_motion_counter], 0,
                     right_arm_ea[length_angles_motion_counter], 0,
                     right_arm_wa[length_angles_motion_counter], 0,
                     0, 0, 0, 0, 0, 0,
                     True, color_polygon)

        #
        sa_temp = np.ceil(((right_arm_sa[length_angles_motion_counter] + 180) * 256) / 360).astype(np.uint8)
        ea_temp = np.ceil(((right_arm_ea[length_angles_motion_counter] + 180) * 256) / 360).astype(np.uint8)
        wa_temp = np.ceil(((right_arm_wa[length_angles_motion_counter] + 180) * 256) / 360).astype(np.uint8)
        denom_temp = np.sqrt((sa_temp**2) + (ea_temp**2) + (wa_temp**2))

        t = [sa_temp / denom_temp, ea_temp / denom_temp, wa_temp / denom_temp]
        matrix_shown.append(t)

        matrix_shown_full.append([sa_temp, ea_temp, wa_temp])

        #
        sa_temp = np.ceil(((motion[0] + 180) * 256) / 360).astype(np.uint8)
        ea_temp = np.ceil(((motion[2] + 180) * 256) / 360).astype(np.uint8)
        wa_temp = np.ceil(((motion[4] + 180) * 256) / 360).astype(np.uint8)
        denom_temp = np.sqrt((sa_temp ** 2) + (ea_temp ** 2) + (wa_temp ** 2))

        t = [sa_temp / denom_temp, ea_temp / denom_temp, wa_temp / denom_temp]
        matrix_actual.append(t)

        matrix_actual_full.append([sa_temp, ea_temp, wa_temp])

        window -= 1
        # print("checkpoint 10")
        print("10:", matrix_shown_full)
        print("10:", matrix_actual_full)

        # audio_file = path + "/" + audio_files[length_angles_motion_counter]
        # y, sr = librosa.load(audio_file)
        # stft = librosa.stft(y)
        # stft_db = librosa.amplitude_to_db(np.abs(stft))

        # c_obj_defined.draw()
        # c_obj_defined.change(right_arm_sa[length_angles_motion_counter], 0,
        #                      right_arm_ea[length_angles_motion_counter], 0,
        #                      right_arm_wa[length_angles_motion_counter], 0,
        #                      0, 0, 0, 0, 0, 0)

        length_angles_motion_counter += 1
        length_angles_motion_counter = length_angles_motion_counter % len(right_arm_sa)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        pygame.display.update()

    # t_2_event.set()
    speed_queue.put([-1, -1, -1])
except:
    pass

pygame.quit()
cap.release()
cv.destroyAllWindows()

# # with open(filename, 'w') as csvfile:
# #     csvwriter = csv.writer(csvfile)
# #     csvwriter.writerow(fields)
# #     csvwriter.writerows(right_arm_data)
#
# matrix_shown_full = np.array(matrix_shown_full)
# matrix_shown_full = matrix_shown_full[:10 * (len(matrix_shown_full) // 10)]
# # print(matrix_shown_full)
#
# image = Image.fromarray(matrix_shown_full, mode='L')  # 'L' mode is for grayscale
# # image_name = storage_location + "/shown_seq_" + str(window_i) + ".png"
# image_name = storage_location + "/shown_seq_" + str(1) + ".png"
# image.save(image_name)
#
# matrix_actual_full = np.array(matrix_actual_full)
# matrix_actual_full = matrix_actual_full[:10 * (len(matrix_actual_full) // 10)]
#
# image = Image.fromarray(matrix_actual_full, mode='L')  # 'L' mode is for grayscale
# # image_name = storage_location + "/done_seq_" + str(window_i) + ".png"
# image_name = storage_location + "/done_seq_" + str(1) + ".png"
# image.save(image_name)
#
# with open(music_params_csv, 'w') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     # csvwriter.writerow(music_params_list)
#     csvwriter.writerows(music_params_list)
#
# # TRAINING and USING the model
#
# model.train()
#
# something = [
#     [1.6466530916060216, -6.197230574440126, -14.995079129175995],
#     [1.889484790107918, -5.495088463517644, -14.15341258785141],
#     [2.234225826449654, -4.932087428264866, -13.799485396019389],
#     [2.5244316425008577, -16.05760840828249, -35.60453398043311],
#     [2.988632455229505, -14.15341258785141, -29.604450746004908],
#     [2.2070614927153462, -12.933780353202234, -24.128402930267857],
#     [2.0157895227202656, -10.388857815469619, -20.55604521958346],
#     [2.067103216935307, -13.873702685485192, -32.31961650818018],
#     [2.862405226111779, -15.697792517861332, -27.474431626277134],
#     [4.377067977053934, -16.04426686320363, -27.149681697783173]]
# matrix_test_full = []
#
# for row in something:
#     sa_temp = np.ceil(((row[0] + 180) * 256) / 360).astype(np.uint8)
#     ea_temp = np.ceil(((row[1] + 180) * 256) / 360).astype(np.uint8)
#     wa_temp = np.ceil(((row[2] + 180) * 256) / 360).astype(np.uint8)
#     denom_temp = np.sqrt((sa_temp ** 2) + (ea_temp ** 2) + (wa_temp ** 2))
#
#     matrix_test_full.append(torch.tensor([sa_temp, ea_temp, wa_temp]))
#
# matrix_test_full = [torch.stack(matrix_test_full)]
# matrix_test_full = torch.stack(matrix_test_full)
# # print(matrix_test_full)
# # print(matrix_test_full.size())
#
# while True:
#     print("Enter frequency: ")
#     freq = int(input())
#     # print("Enter BPM: ")
#     # bpm = int(input())
#     bpm = 0
#
#     music_test = [torch.tensor(freq), torch.tensor(bpm)]
#     music_test = torch.stack(music_test)
#
#     DTW_pred = model.use_model(matrix_test_full, music_test)
#
#     print(DTW_pred)
