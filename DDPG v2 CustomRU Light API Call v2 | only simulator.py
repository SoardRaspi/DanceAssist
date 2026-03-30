import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import queue
import time
import requests
import threading
import math
import sys
import json

import queue
import time

# import tkinter as tk
from PIL import Image
from itertools import count

# from DanceGenerator import dance_comp_Dyuthi
# from DanceGenerator.DyuthiVRNN import modelv1
from DanceGenerator.Spectros import spectro_combined, combined_music_pydub

from flask import Flask, request, jsonify

# import DanceGenerator.FaceKps_EyeCroppedNet.eye_mode_1
# from DanceGenerator.FaceKps_EyeCroppedNet.eye_mode_1 import eye_mode_1

from DanceGenerator.FaceKps_EyeCroppedNet.actor import actor_model_RU

# from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
# from PyQt5.QtGui import QMovie
# from PyQt5.QtCore import QTimer

from music_recommendation_synthetic_verifier import random_selection

# 0 -> normal mode - take video input
# 1 -> testing mode - take face keypoints and reward from tester
RUN_MODE = 1

# import pygame

import mediapipe as mp
# import imageio

from st_gcn_st_gcn.net.st_gcn import ST_GCN_Model

region__kps_idx__map = random_selection.get_region__kps_idx__map()
physical_music_pref_gaussian_2d_map_heatmap = random_selection.get_physical_music_pref_gaussian_2d_map_heatmap()
random_music_pref_ring_radii = random_selection.get_random_music_pref_ring_radii()
random_music_pref_mean_dict = random_selection.get_random_music_pref_mean_dict()

itoo_music = [item[1] for item in sorted([(v, k) for k, v in random_music_pref_mean_dict.items()])[::-1]]
# itoo_music is important to get the user's preference of music choice
# print("region__kps_idx__map:", region__kps_idx__map)
print("itoo_music:", itoo_music)
print("random_music_pref_ring_radii:", random_music_pref_ring_radii)

# for key in random_music_pref_ring_radii:
#     min_radius_temp, max_radius_temp = random_music_pref_ring_radii[key][0]
#     width = abs(int(max_radius_temp - min_radius_temp)) // 2
#     print(int(min_radius_temp), abs(int(max_radius_temp - min_radius_temp)), width)

# USER PROPERTY TO REGULATE THE DELAY BETWEEN HEAD MOVES
head_mov_delay_time = 0.5

# import concurrent.futures

# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils

# mp_face_mesh = mp.solutions.face_mesh
# mp_pose = mp.solutions.pose

# eye modes
# eye_mov_mode: 0 -> to disable the branch
#               1 -> to consider just the values of the position of the iris in the eye, will require some iris processing using MediaPipe
#               2 -> to consider the whole eye-cropped frame with affine transformation as pre-processing and then feature extraction (too heavy and unstable for DDPG)
#               3 -> to consider only the eye input
block_duration = 1

INPUT_MODE = 0
num_steps_in_batch = 60 * block_duration  # 60

# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

block_size = 10 * block_duration  # 10
num_music_options = 7
total_episodes = 10

app = Flask(__name__)

# Create a Queue for handling music change requests
music_queue = queue.Queue()

cap = cv.VideoCapture(0)

chosen_music_array = []
music_index_iter_array = []

# @app.route('/clean_main', methods=['POST'])
def video_capture_and_stuff(path_to_json_root = "/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DDPG with new changes", preference_plotting=None):
    # app = QApplication(sys.argv)

    ht__hidden_dimension = 16
    num_music_options = 7
    num_RU_blocks = 5  # 6

    block_time_started = False

    block_start_time = time.time()
    music_times__per_block = []

    # window = GifPlayer("/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/animation_from.gif")
    # window.show()

    for ep_inner in range(total_episodes):
        with open(os.path.join(path_to_json_root, f"keypoints_data_{ep_inner + 1}.json"), 'w') as json_file:
            pass
    
    # ht_0 = np.zeros(ht__hidden_dimension,)
    ht_0 = torch.zeros(ht__hidden_dimension)

    # music_probs__t_0 = np.random.uniform(0, 1, (1, num_RU_blocks, num_music_options))
    music_probs__t_0 = torch.rand(1, num_RU_blocks, num_music_options)

    music_index_iter = combined_music_pydub.vanilla_music_selector(music_probs__t_0)

    face_9_keypoints = []
    T_N_M = []

    green_sig_init_once = False
    flag_generated_motion_once = False
    flag_movement_demo = False
    flag_sig_movement_demo = False
    sig_movement_demo = None
    flag_do_demo_warn = False
    flag_take_usr_input_rep_demo = False
    flag_take_usr_input_rep = False
    flag_user_rep_done = False
    flag_shown_GIF_loop_once = False

    pose_seq_similarity_score = None

    episode_counter = 0

    steps_batch_counter = 0

    # model_actor_local = EyeVidPre_and_ViViT.Model(5)
    music_folder = "Musics"  # Folder where music files are stored
    music_files = [os.path.join(music_folder, f) for f in os.listdir(music_folder) if f.endswith('.mp3')]
    music_index_iter = np.random.randint(0, 7)
    chosen_music = random_selection.tester_get_chosen_music()

    motion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # count = 1

    right_eye_video = []
    left_eye_video = []

    # prev_head_movement = None
    prev_head_movement = torch.tensor(np.zeros((1, 2, 15, 9, 1)), requires_grad=True)

    curr_head_movement = None

    BRIGHTNESS_THRESHOLD = 100  # You can adjust this value based on testing

    reward_avg = 0
    avgs_unit = []

    gen_seq_runs = 5

    re = None
    le = None
    p_195 = None
    p_94 = None
    nose = None
    glabella = None

    num = None
    denom = None

    re_points = []
    le_points = []

    count = 1

    plain_raw_video = [[], [], [], []]  # [[], []] = [(right-eye), (left-eye)]

    # if RUN_MODE == 0:
    #     cap = cv.VideoCapture(0)
    # elif RUN_MODE == 1:
    #     # tester code
    #     cap = None
    # cap = cv.VideoCapture(0)

    # num_frames = 32
    num_frames = num_steps_in_batch

    # ### Threading objects start
    # frame_queue = queue.Queue(maxsize=5)  # Limit the queue size to avoid overfilling
    #
    # # Create and start the numbered threads
    # num_threads = 5
    # threads = [NumberedThread(frame_queue, i) for i in range(num_threads)]
    # for thread in threads:
    #     thread.start()
    #
    frame_arr = []
    # ### Threading objects end

    prev_state = None
    action = None
    reward = None
    curr_state = None

    flag_red_sig = True
    flag_green_sig = False

    try:
        print("Before for loop in try, finally condition...")
        window_num = 1
        # vid_start_time = time.time()

        # #Add music playing part here
        # current_music_index = (window_num % 5)
        # current_music_file = music_files[current_music_index]
        # print("current_music_file:", current_music_file)
        # music_queue.put(music_files[current_music_index])

        start_time_total = time.time()
        red_signal = time.time()
        VRNN_pose_input = []
        pose_seq_user_rep = []
        VRNN_seq_gen = None

        max_indices = music_probs__t_0.argmax(dim=-1)  # Shape: [1, 6]
        one_hot = F.one_hot(max_indices, num_classes=7)  # Shape: [1, 6, 7]
        one_hot = one_hot.float()
        music_one_hot_vectors = one_hot.argmax(dim=-1)
        
        if RUN_MODE == 1:
            nose_point_start_time = None
            nose_point_start_flag = False

            API_input = {"input": []}

            face_coords = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}
            rand_kps_1 = random_selection.tester_call_function(music_index_iter, region__kps_idx__map)

            # print("rand_kps_1, music_index_iter:", rand_kps_1, music_index_iter)

            frame_counter = 0

            for k in rand_kps_1.keys():
                face_coords[int(k)] = rand_kps_1[k]
            
            for _ in range(total_episodes):
                # for ep in range(num_frames):
                while frame_counter < num_frames:
                    success, img = cap.read()
                    # success, img = False, None

                    if block_time_started is False:
                        block_start_time = time.time()
                        block_time_started = True
                    
                    if nose_point_start_flag is False:
                        nose_point_start_flag = True
                        nose_point_start_time = time.time()
                    
                    if time.time() - nose_point_start_time >= head_mov_delay_time:
                        face_coords = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}
                        rand_kps_1 = random_selection.tester_call_function(music_index_iter, region__kps_idx__map)

                        # print("rand_kps_1:", rand_kps_1)

                        if success and (img is not None):
                            # print("inside getting frames")
                        # if False:
                            # plot the keypoints
                            image_x = int(img.shape[1] / 2)
                            image_y = int(img.shape[0] / 2)

                            gh, gw, _ = physical_music_pref_gaussian_2d_map_heatmap.shape

                            top = image_y - gh // 2
                            left = image_x - gw // 2

                            top = max(0, top)
                            left = max(0, left)
                            bottom = top + gh
                            right = left + gw

                            # Ensure overlay doesn't exceed frame size
                            if bottom > img.shape[0] or right > img.shape[1]:
                                continue

                            roi = img[top:bottom, left:right]

                            # print("shapes for gaussian overlay:", roi.shape, physical_music_pref_gaussian_2d_map_heatmap.shape, img.shape)

                            # Blend heatmap over region of interest
                            blended = cv.addWeighted(roi, 0.5, physical_music_pref_gaussian_2d_map_heatmap, 0.5, 0)
                            img[top:bottom, left:right] = blended

                            overlay = img.copy()
                            alpha = 0.4

                            print("---overlay points---")
                            for key in random_music_pref_ring_radii:
                                min_radius_temp, max_radius_temp = min(random_music_pref_ring_radii[key][0]), max(random_music_pref_ring_radii[key][0])
                                
                                # cv.circle(img, (image_x, image_y), int(min_radius_temp), (0, 0, 255), int(max_radius_temp - min_radius_temp))
                                # overlay = img.copy()
                                cv.circle(overlay, (image_x, image_y), int(min_radius_temp), (0, 0, 255), int(max_radius_temp - min_radius_temp))
                                c = int((max_radius_temp + min_radius_temp) // 2)
                                cv.putText(overlay, str(key), (image_x + c, image_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                                print("key:", key, min_radius_temp, max_radius_temp, c)
                                # cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                            
                            for key in random_music_pref_ring_radii:
                                min_radius_temp, max_radius_temp = min(random_music_pref_ring_radii[key][0]), max(random_music_pref_ring_radii[key][0])
                                c = int((max_radius_temp + min_radius_temp) // 2)
                                cv.putText(overlay, str(key), (image_x + c, image_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                                # cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                            print("---overlay points---")
                            
                            face_coord_9pointer = rand_kps_1

                            for key in face_coord_9pointer:
                                x, y, _z, __aa = face_coord_9pointer[key]
                                # x, y = face_coord_9pointer[key]

                                x *= img.shape[1]
                                x = int(x)

                                y *= img.shape[0]
                                y = int(y)

                                face_coord_9pointer[key] = (x, y)

                                # cv.circle(img, (x, y), 3, (128, 128, 128), 3)
                                cv.circle(img, (x, y), 3, (255, 255, 255), 3)
                            
                            # print("face_coord_9pointer:", face_coord_9pointer["7"])

                            line_connecting_color = (255, 255, 255)
                            cv.line(overlay, face_coord_9pointer["7"], face_coord_9pointer["3"], line_connecting_color, 3)
                            cv.line(overlay, face_coord_9pointer["3"], face_coord_9pointer["2"], line_connecting_color, 3)
                            cv.line(overlay, face_coord_9pointer["2"], face_coord_9pointer["1"], line_connecting_color, 3)

                            cv.line(overlay, face_coord_9pointer["1"], face_coord_9pointer["0"], line_connecting_color, 3)

                            cv.line(overlay, face_coord_9pointer["8"], face_coord_9pointer["6"], line_connecting_color, 3)
                            cv.line(overlay, face_coord_9pointer["6"], face_coord_9pointer["5"], line_connecting_color, 3)
                            cv.line(overlay, face_coord_9pointer["5"], face_coord_9pointer["4"], line_connecting_color, 3)

                            cv.line(overlay, face_coord_9pointer["4"], face_coord_9pointer["0"], line_connecting_color, 3)

                            cv.putText(img, str(chosen_music) + ", " + str(music_index_iter) + ", " + str(reward), (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                            # print("img shape", img.shape)

                            cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                            cv.imshow("img", img)
                        
                        print(rand_kps_1)
                        for k in rand_kps_1.keys():
                            face_coords[int(k)] = rand_kps_1[k]

                        # API_input[frame_counter] = face_coords
                        face_coords_temp = [v for k, v in face_coords.items()]
                        # print("face_coords_temp:", face_coords_temp)
                        API_input["input"].append(face_coords_temp)
                        
                        nose_point_start_flag = False
                        nose_point_start_time = time.time()

                        frame_counter += 1
                        img = None
                        print("frame_counter:", frame_counter)
                        print("music details:", chosen_music, music_index_iter)
                        
                    # if flag_red_sig is True:
                    if flag_user_rep_done is False:
                        # if (time.time() - green_signal) < 5:
                        if steps_batch_counter < num_steps_in_batch:
                                # if success:
                                if True:
                                    # cv.putText(img2, f"Frames left {num_steps_in_batch - steps_batch_counter}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
                                    steps_batch_counter += 1
                                    # cv.putText(img2, str(5 - (int(time.time() - green_signal))), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
                                    # print("flag_green_sig is still False: ", 5 - (int(time.time() - green_signal)), time.time(), green_signal)
                                
                                else:
                                    raise Exception("Error in initial video feed")
                            
                        else:
                            flag_user_rep_done = True
                            # print("flag_user_rep_done is now True")

                    right_eye_main = []
                    left_eye_main = []

                    try:
                        if (frame_counter % block_size) == 0:
                            decided_music_index = music_one_hot_vectors[0][(frame_counter // block_size) % len(music_files)]
                            music_file_temp = music_files[decided_music_index]
                            # print("music_file_temp in music queue:", music_file_temp)
                            
                            # music_queue.put(music_file_temp)

                            block_end_time = time.time()
                            music_times__per_block.append(block_end_time - block_start_time)
                            block_start_time = time.time()

                            block_time_started = False
                    except Exception as music_playing_error:
                        print("error in sending music through queue:", music_playing_error)
                            
                    N_M = []

                    for key in face_coords:
                        try:
                            # print("test point 1s")
                            
                            # x, y, z, vis = face_coords[key]
                            x, y = face_coords[key][0], face_coords[key][1]

                            # print(x, y, z, vis)
                            # N_M.append([x, y, z, vis])
                            N_M.append([x, y])

                            # x *= img2.shape[1]
                            x *= 640
                            x = int(x)
                            
                            # y *= img2.shape[0]
                            y *= 480
                            y = int(y)

                            # cv.circle(img2, (x, y), 3, (255, 0, 0), 3)
                        except Exception as e_test_point_1s:
                            print("e_test_point_1s:", e_test_point_1s)
                            print("error at test point 1s:", face_coords)
                            break
                    
                    # print("N_M:", N_M)
                    
                    T_N_M.append(N_M)

                    if flag_user_rep_done is True:
                        # music_queue.put(None)

                        flag_user_rep_done = False
                        steps_batch_counter = 0

                        face_9_keypoints = []
                        reward = 0

                        music_times__per_block = []

                        if False or (cv.waitKey(1) & 0xFF == ord('q')):
                            break
                        # frame_counter += 1
                
                reward = 0
                
                prev_head_movement = curr_head_movement

                response = None

                # music_index_iter = 0

                music_probs__t_0 = torch.rand(1, num_RU_blocks, num_music_options)
                music_index_iter = combined_music_pydub.vanilla_music_selector(music_probs__t_0)

                frame_counter = 0
                API_input = {"input" : []}

    except Exception as e_video_input_threading:
        print("Error in video input threading...:", e_video_input_threading)
        pass

video_capture_and_stuff()