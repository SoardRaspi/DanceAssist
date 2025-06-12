import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
from keras import layers, models, Input
from keras._tf_keras.keras.activations import relu, softmax, tanh

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import queue
import time
import threading
import math
import sys
import json

import tkinter as tk
from PIL import Image, ImageTk, ImageSequence
from itertools import count

from transformers import VivitImageProcessor, VivitForVideoClassification, \
    AutoImageProcessor, MobileNetV1Model
from DanceGenerator import dance_comp_Dyuthi
from DanceGenerator.DyuthiVRNN import modelv1
from DanceGenerator.Spectros import spectro_combined, combined_music_pydub

# import DanceGenerator.FaceKps_EyeCroppedNet.eye_mode_1
# from DanceGenerator.FaceKps_EyeCroppedNet.eye_mode_1 import eye_mode_1

from DanceGenerator.FaceKps_EyeCroppedNet.actor import actor_model_RU

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QTimer

from music_recommendation_synthetic_verifier import random_selection

# 0 -> normal mode - take video input
# 1 -> testing mode - take face keypoints and reward from tester
RUN_MODE = 1

# import pygame

import mediapipe as mp
import imageio

from st_gcn_st_gcn.net.st_gcn import ST_GCN_Model

# USER PROPERTY TO REGULATE THE DELAY BETWEEN HEAD MOVES
head_mov_delay_time = 2.0

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
num_steps_in_batch = 60 * block_duration

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

block_size = 10 * block_duration
num_music_options = 7

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

try:
    dance_gen_model_path = "/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DanceGenerator/models/vrnn_trial_2__only_top__epoch14.pth"
    dance_gen_model_path = "/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DanceGenerator/models/vrnn_trial_2__only_top__epoch19 right dataset.pth"

    model_seq_gen = modelv1.get_model(dance_gen_model_path)
    print("dance generator model:", type(model_seq_gen))
except:
    model_seq_gen = None
    raise Exception("Error initializing the Dance Generator model")

# pygame.mixer.init()

# Global variable to track the current music
# current_music_file = None
# music_change_event = threading.Event()

# cap = cv.VideoCapture(0)

from CriticNetwork import EyeVidPre_and_ViViT_tf, EyeVidPre_and_ViViT
# from CriticNetwork.EyeVidPre_and_ViViT import KMeans_Image
from CriticNetwork import EyeCropping


# GIF player
class GifPlayer(QWidget):
    def __init__(self, gif_path):
        super().__init__()
        self.setWindowTitle("GIF Player")
        self.setGeometry(100, 100, 400, 300)

        # Create layout and QLabel for displaying GIF
        self.layout = QVBoxLayout()
        self.label = QLabel(self)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        # Load and start GIF
        self.loop_counter = 0
        self.max_loops = 3  # ✅ Set the number of times the GIF should play
        self.update_gif(gif_path)

    def update_gif(self, gif_path):
        """ Update the GIF being displayed and play it exactly 3 times """
        self.loop_counter = 0  # Reset loop count when updating GIF
        self.movie = QMovie(gif_path)
        self.label.setMovie(self.movie)

        # ✅ Connect the finished signal to a custom slot
        self.movie.finished.connect(self.restart_gif)

        self.movie.start()

    def restart_gif(self):
        """ Restart the GIF manually until it reaches 3 loops """
        if self.loop_counter < self.max_loops - 1:
            self.loop_counter += 1
            self.movie.start()  # Restart the GIF
        else:
            print("GIF finished playing 3 times.")  # Stop after 3 loops

# model_frozen = EyeVidPre_and_ViViT.Model()
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# model_frozen.to(device)

# def get_actor(right_eye_video_gray, left_eye_video_gray):
#     # frames_right = []
#     # frames_left = []
#     #
#     # print("reached inside PreProcessandFreezedOutput")
#     #
#     # for i, frame in enumerate(right_eye_video_gray):
#     #     print("frame shape:", frame.shape)
#     #
#     #     normal = frame
#     #     otsu_threshold, otsu_binarized = cv.threshold(
#     #         frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
#     #     )
#     #     kmeans = KMeans_Image(frame)
#     #
#     #     normal = np.expand_dims(normal, axis=-1)
#     #     otsu_binarized = np.expand_dims(otsu_binarized, axis=-1)
#     #     kmeans = np.expand_dims(kmeans, axis=-1)
#     #
#     #     normal = np.concatenate((normal, otsu_binarized), axis=-1)
#     #     normal = np.concatenate((normal, kmeans), axis=-1)
#     #
#     #     frames_right.append(normal)
#     #
#     # for i, frame in enumerate(left_eye_video_gray):
#     #     normal = frame
#     #     otsu_threshold, otsu_binarized = cv.threshold(
#     #         frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
#     #     )
#     #     kmeans = KMeans_Image(frame)
#     #
#     #     normal = np.expand_dims(normal, axis=-1)
#     #     otsu_binarized = np.expand_dims(otsu_binarized, axis=-1)
#     #     kmeans = np.expand_dims(kmeans, axis=-1)
#     #
#     #     normal = np.concatenate((normal, otsu_binarized), axis=-1)
#     #     normal = np.concatenate((normal, kmeans), axis=-1)
#     #
#     #     frames_left.append(normal)
#     #
#     # frames_right = np.array(frames_right)
#     # frames_left = np.array(frames_left)
# 
#     output_from_model, softmaxed_layer = EyeVidPre_and_ViViT.PreProcessandFreezedOutput(
#         right_eye_video_gray, left_eye_video_gray)
#     # model = EyeVidPre_and_ViViT.Model()
# 
#     final_frequency = torch.argmax(softmaxed_layer, dim=-1)
# 
#     # return output_from_model
#     return final_frequency
#     # return model
# 
# def get_critic(right_eye_video_gray_, left_eye_video_gray_, final_layer_from_model):
#     output_from_model, softmaxed_layer = EyeVidPre_and_ViViT.PreProcessandFreezedOutput(
#         right_eye_video_gray_, left_eye_video_gray_)
# 
#     # output_from_model = np.concatenate((output_from_model, final_layer_from_model), axis=-1)
# 
#     # Both are passed through separate layer before concatenating
#     concat = layers.Concatenate()([output_from_model, final_layer_from_model])
#     print("length after concat:", output_from_model.shape, final_layer_from_model.shape, concat.shape)
# 
#     out = layers.Dense(256, activation="relu")(concat)
#     out = layers.Dense(256, activation="relu")(out)
#     outputs = layers.Dense(1)(out)
# 
#     return outputs
# 
# 
# class Actor(nn.Module):
#     def __init__(self):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(3137 * 768, 128)
#         self.fc2 = nn.Linear(256, 128)
#         self.softmax = nn.Softmax()

log_file = "DDPG v2 log.txt"

lower_bound = 0
upper_bound = 5

# Create a Queue for handling music change requests
music_queue = queue.Queue()

# image_processor_1 = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
# model_1 = MobileNetV1Model.from_pretrained("google/mobilenet_v1_1.0_224")

# Load GIF
root = tk.Tk()
root.title("GIF Player")

def tf_to_torch(tf_tensor):
    return torch.tensor(tf_tensor.numpy(), requires_grad=True)

def torch_to_tf(torch_tensor):
    return tf.convert_to_tensor(torch_tensor.detach().numpy())

def play_gif(frames, label, frame_idx, play_count, loop_count=1):
    # global frame_idx, play_count
    
    if play_count >= loop_count:
        root.destroy()  # Close the Tkinter window after playing 10 times
        return
    
    frame = frames[frame_idx]
    label.config(image=frame)
    frame_idx = (frame_idx + 1) % len(frames)
    
    if frame_idx == 0:
        play_count += 1
    
    root.after(100, play_gif)  # Adjust timing if needed

def get_actor_extra_RNN():
    # num_frames = 32
    num_frames = num_steps_in_batch

    # shape_single_frame = raw_video_input[0][0].shape

    # print("num_frames, shape_single_frame:", num_frames, shape_single_frame)

    print("\nStart of shapes in get_actor_extra_RNN---")

    right_eye_seq = layers.Input(shape=(num_frames, 32, 
                                        128, 3))
    left_eye_seq = layers.Input(shape=(num_frames, 32, 
                                       128, 3))
    
    print(f"\n shape of right_eye_seq: {right_eye_seq.shape}")
    
    # TimeDistributed CNN for spatial feature extraction
    x = layers.TimeDistributed(
        tf.keras.Sequential([
            layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.Flatten()
        ]),
        name="TimeDistributed_CNN"
    )(right_eye_seq)  # Shape: (batch_size, sequence_length, features)
    print(f"\n shape of x 1: {x.shape}")

    # 3-layer GRU for temporal sequence modeling
    x = layers.GRU(64, return_sequences=True, activation="tanh", name="GRU_Layer_1")(x)
    print(f"\n shape of x 2: {x.shape}")

    x = layers.GRU(64, return_sequences=True, activation="tanh", name="GRU_Layer_2")(x)
    print(f"\n shape of x 3: {x.shape}")

    x_right_eye = layers.GRU(64, return_sequences=False, activation="tanh", name="GRU_Layer_3")(x)
    print(f"\n shape of x_right_eye: {x_right_eye.shape}")

    # # TimeDistributed CNN for spatial feature extraction
    # x = layers.TimeDistributed(
    #     tf.keras.Sequential([
    #         layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    #         layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    #         layers.Flatten()
    #     ]),
    #     name="TimeDistributed_CNN"
    # )(left_eye_seq)  # Shape: (batch_size, sequence_length, features)

    # # 3-layer GRU for temporal sequence modeling
    # x = layers.GRU(64, return_sequences=True, activation="tanh", name="GRU_Layer_1")(x)
    # x = layers.GRU(64, return_sequences=True, activation="tanh", name="GRU_Layer_2")(x)
    # x_left_eye = layers.GRU(64, return_sequences=False, activation="tanh", name="GRU_Layer_3")(x)

    dense_right_eye = layers.Dense(6, activation="softmax")(x_right_eye)
    print(f"\n shape of dense_right_eye: {dense_right_eye.shape}")

    model = keras.Model(inputs=right_eye_seq, outputs=dense_right_eye)
    model.summary()

    return model

class KerasWrapped_ST_GCN(layers.Layer):
    def __init__(self, torch_model):
        super(KerasWrapped_ST_GCN, self).__init__()

        self.torch_model = torch_model
    
    def call(self, inputs):
        numpy_inputs = tf.experimental.numpy.array(inputs)
        torch_inputs = torch.tensor(numpy_inputs, dtype=torch.float32)
        torch_outputs = self.torch_model(torch_inputs)

        # torch_inputs = keras.ops.convert_to_tensor(inputs)
        # torch_outputs = self.torch_model(torch_inputs)

        # return keras.ops.convert_to_tensor(torch_outputs)
        return tf.convert_to_tensor(torch_outputs.detach().cpu().numpy())

def get_actor_extra_ST_GCN():
    ST_GCN = ST_GCN_Model(in_channels=4, num_class=6, graph_args={"layout": 'mediapipe_face_9'}, edge_importance_weighting=False)
    # input for model of shape (1, 4, 60, 9, 1) --> batch_size, features (x, y, z, visibility), num_frames, num_keypoints, num_persons

    ST_GCN_Keras_wrapped = layers.TorchModuleWrapper(ST_GCN)
    # ST_GCN_Keras_wrapped = KerasWrapped_ST_GCN(ST_GCN)

    head_movement_input = layers.Input(shape=(1, 4, num_steps_in_batch, 9, 1))
    print("shape of head_movement_input:", type(head_movement_input), head_movement_input)

    x = ST_GCN_Keras_wrapped(head_movement_input)

    out_ST_GCN = softmax(x)

    actor_ST_GCN_model = keras.Model(head_movement_input, out_ST_GCN)
    return actor_ST_GCN_model

class get_actor_extra_ST_GCN_torch(nn.Module):
    def __init__(self):
        super(get_actor_extra_ST_GCN_torch, self).__init__()
        self.st_gcn = ST_GCN_Model(
            in_channels=4, 
            num_class=6, 
            graph_args={"layout": 'mediapipe_face_9'}, 
            edge_importance_weighting=False
        )
        self.softmax = nn.Softmax(dim=1)  # Softmax along class dimension

    def forward(self, x):
        print("Input shape:", x.shape)

        x = self.st_gcn(x)  # Pass through ST-GCN
        x = self.softmax(x)  # Apply softmax

        return x

# def get_actor_extra():
#     # model = nn.Sequential(
#     #     nn.Linear(3137 * 768, 128),
#     #     nn.ReLU(),
#     #     nn.Flatten(),
#     #     nn.Linear(256, 128),
#     #     nn.Softmax()
#     # )

#     '''model = EyeVidPre_and_ViViT.Actor_PreProcessandFreezedOutput_MobileNetV1_Model_Tf()'''

#     # model = models.Sequential([
#     #     layers.Dense(128, input_shape=(3137 * 768,)),
#     #     layers.Activation(relu),
#     #     layers.Flatten(),
#     #     layers.Dense(128),
#     #     layers.Activation(softmax)
#     # ])

#     print("\nStart of shapes in get_actor_extra---")

#     right_eye = layers.Input(shape=(1024, 7, 7))
#     left_eye = layers.Input(shape=(1024, 7, 7))

#     dense_layer_1 = layers.Dense(128, input_shape=(2*50176,))
#     dense_layer_1_conc = layers.Dense(128)
#     dense_layer_2 = layers.Dense(64)
#     dense_layer_3 = layers.Dense(6)
#     softmax = layers.Softmax()
#     sigmoid = layers.Activation('sigmoid')

#     last_state_concat = layers.Concatenate(axis=-1)([right_eye, left_eye])
#     print("shape of last_state_concat in get_actor_extra:", last_state_concat.shape)
#     print(f"\nshape of last_state_concat in get_actor_extra: {last_state_concat.shape}")

#     # concat_layer_flatten = tf.reshape(last_state_concat, (tf.shape(last_state_concat)[0], -1))
#     concat_layer_flatten = layers.Reshape((-1,))(last_state_concat)
#     print("shape of concat_layer_flatten in get_actor_extra:", concat_layer_flatten.shape)
#     print(f"\nshape of concat_layer_flatten in get_actor_extra: {concat_layer_flatten.shape}")

#     output_after_dense = dense_layer_1(concat_layer_flatten)
#     print(f"\nshape of output_after_dense in get_actor_extra after dense_layer_1: {output_after_dense.shape}")

#     # output_after_dense = tf.reshape(output_after_dense, (-1,))
#     output_after_dense = layers.Reshape((-1,))(output_after_dense)
#     print(f"\nshape of output_after_dense in get_actor_extra after reshaping 1: {output_after_dense.shape}")

#     # output_after_dense = tf.expand_dims(output_after_dense, axis=0)
#     output_after_dense = layers.Reshape((1, -1))(output_after_dense)
#     print(f"\nshape of output_after_dense in get_actor_extra after reshaping 2: {output_after_dense.shape}")

#     output_after_dense = dense_layer_1_conc(output_after_dense)
#     print(f"\nshape of output_after_dense in get_actor_extra after dense_layer_1_conc: {output_after_dense.shape}")

#     output_after_dense = dense_layer_2(output_after_dense)
#     print(f"\nshape of output_after_dense in get_actor_extra after dense_layer_2: {output_after_dense.shape}")

#     output_after_dense = dense_layer_3(output_after_dense)
#     print(f"\nshape of output_after_dense in get_actor_extra after dense_layer_3: {output_after_dense.shape}")

#     final_output = softmax(output_after_dense)
#     print(f"\nshape of final_output in get_actor_extra: {final_output.shape}")
#     print("\nEnd of shapes in get_actor_extra---")

#     model = keras.Model([right_eye, left_eye], final_output)
#     return model

class get_critic_extra_ST_GCN_torch(nn.Module):
    def __init__(self):
        super(get_critic_extra_ST_GCN_torch, self).__init__()

        # ST-GCN Model
        self.st_gcn = ST_GCN_Model(
            in_channels=4, 
            num_class=6, 
            graph_args={"layout": 'mediapipe_face_9'}, 
            edge_importance_weighting=False
        )

        # Dense layers
        self.dense_layer_parallel_1 = nn.Linear(6, 16)  # Input shape: (batch_size, 1, 6)
        self.dense_layer_parallel_2 = nn.Linear(16, 32)

        self.conv_layer_parallel_1 = nn.Conv1d(in_channels=7, out_channels=16, kernel_size=3, padding=1) # Input shape: (batch_size, 6, 7)
        self.conv_layer_parallel_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.dense_layer_comb_1 = nn.Linear(32 + 6, 32)  # Concatenated size (6 from softmax output)
        self.dense_layer_comb_2 = nn.Linear(32, 8)
        self.dense_layer_final = nn.Linear(8, 1)

        self.softmax = nn.Softmax(dim=1)  # Softmax on class dimension

    def forward(self, x):
        """
        head_movement_input: (batch_size, 4, 60, 9, 1)

        # action: (batch_size, 1, 6)
        action: (batch_size, 6, 7) -> 7 is the one-hot vector
        """

        head_movement_input, action = x

        print("type of action in get_critic_extra_ST_GCN_torch model:", type(action), len(action), type(action[0]))
        if not isinstance(action, torch.Tensor):
            action = torch.stack(action)
        print("action shape initial in get_critic_extra_ST_GCN_torch:", action.shape)

        if action.ndim == 2:
            pass
        elif action.ndim == 3:
            # Input: x of shape (batch_size, 6, 7)
            # Convert one-hot to indices (argmax along feature dim)
            x_indices = action.argmax(dim=2)  # Shape: (batch_size, 6)

            embedding = nn.Embedding(num_embeddings=7, embedding_dim=16)
            x_embed = embedding(x_indices)  # Shape: (batch_size, 6, 16)
            action = x_embed
        else:
            raise ValueError("Error in the action input to get_critic_extra_ST_GCN_torch model")

        print("\nInput Shapes:")
        print("Head movement input:", head_movement_input.shape)
        print("Action input:", action.shape)

        # Pass through ST-GCN
        x = self.st_gcn(head_movement_input)
        out_st_gcn = self.softmax(x)

        # # Process action input through parallel dense layers
        # action_inner = F.relu(self.dense_layer_parallel_1(action))
        # action_inner = F.relu(self.dense_layer_parallel_2(action_inner))

        # Process action input through parallel conv1d layers
        action_inner = action.permute(0, 2, 1)
        # action_inner = F.relu(self.conv_layer_parallel_1(action_inner))
        action_inner = F.relu(self.conv_layer_parallel_2(action_inner))
        action_inner = nn.AdaptiveAvgPool1d(1)(action_inner)  # (batch_size, 32)
        action_inner = action_inner.squeeze(-1)

        print("\nShape after action_dense layers:", action_inner.shape)
        print("\nShape after st_gcn layer:", out_st_gcn.shape)

        # Concatenate action and out_st_gcn
        comb = torch.cat((action_inner, out_st_gcn), dim=1)

        print("\nShape after concatenation:", comb.shape)

        # Flatten (reshape to remove extra dimensions)
        comb = comb.view(comb.shape[0], -1)

        print("\nShape after reshaping:", comb.shape)

        # Apply final dense layers
        comb = F.relu(self.dense_layer_comb_1(comb))
        comb = F.relu(self.dense_layer_comb_2(comb))
        final_output = self.dense_layer_final(comb)

        print("\nFinal Output Shape:", final_output.shape)

        return final_output

def get_critic_extra_ST_GCN():
    ST_GCN = ST_GCN_Model(in_channels=4, num_class=6, graph_args={"layout": 'mediapipe_face_9'}, edge_importance_weighting=False)
    # input for model of shape (1, 4, 60, 9, 1) --> batch_size, features (x, y, z, visibility), num_frames, num_keypoints, num_persons

    dense_layer_parallel_1 = keras.layers.Dense(16)
    dense_layer_parallel_2 = keras.layers.Dense(32)
    dense_layer_comb_1 = keras.layers.Dense(32)
    dense_layer_comb_2 = keras.layers.Dense(8)
    dense_layer_final = keras.layers.Dense(1)

    ST_GCN_Keras_wrapped = layers.TorchModuleWrapper(ST_GCN)
    # ST_GCN_Keras_wrapped = KerasWrapped_ST_GCN(ST_GCN)

    head_movement_input = layers.Input(shape=(1, 4, num_steps_in_batch, 9, 1))
    action = layers.Input(shape=(1, 6))

    x = ST_GCN_Keras_wrapped(head_movement_input)

    out_ST_GCN = softmax(x)

    # Process action input through parallel dense layers
    action_inner = dense_layer_parallel_1(action)
    print(f"\nshape of action_inner in get_critic_extra after dense_layer_parallel_1: {action_inner.shape}")

    action_inner = dense_layer_parallel_2(action_inner)
    print("shape of action_inner:", action_inner.shape)
    print(f"\nshape of action_inner in get_critic_extra after dense_layer_parallel_2: {action_inner.shape}")

    # Combine action and output_after_dense
    # comb = tf.concat([action, output_after_dense], axis=0)
    # comb = layers.Concatenate(axis=0)([action_inner, output_after_dense])
    comb = layers.Concatenate(axis=1)([action_inner, out_ST_GCN])

    print("shape of comb:", comb.shape)
    print(f"\nshape of comb in get_critic_extra after action_inner, output_after_dense concat: {comb.shape}")

    # comb = tf.reshape(comb, (tf.shape(comb)[0], -1))
    comb = layers.Reshape((-1,))(comb)
    print("shape of comb 2:", comb.shape)
    print(f"\nshape of comb in get_critic_extra after comb reshape: {comb.shape}")

    # Apply the combined dense layers
    comb = dense_layer_comb_1(comb)
    print(f"\nshape of comb in get_critic_extra after dense_layer_comb_1: {comb.shape}")

    comb = dense_layer_comb_2(comb)
    print("shape of comb 3:", comb.shape)
    print(f"\nshape of comb in get_critic_extra after dense_layer_comb_2: {comb.shape}")

    # Final output
    final_output = dense_layer_final(comb)
    print("shape of final_output:", final_output.shape)
    print(f"\nshape of final_output in get_critic_extra for final output: {final_output.shape}")

    actor_ST_GCN_model = keras.Model([head_movement_input, action], final_output)
    return actor_ST_GCN_model

# def get_critic_extra():
#     # class CustomModel(nn.Module):
#     #     def __init__(self):
#     #         super(CustomModel, self).__init__()
#     #         self.layer1 = nn.Sequential(
#     #             nn.Linear(3137 * 768, 128),
#     #             nn.ReLU(),
#     #             nn.Flatten(),
#     #             nn.Linear(256, 128)
#     #         )
#     #         self.layer2 = nn.Sequential(
#     #             nn.Linear(256, 256),
#     #             nn.ReLU(),
#     #             nn.Linear(256, 256),
#     #             nn.ReLU(),
#     #             nn.Linear(256, 1),
#     #             nn.Tanh()
#     #         )
#     #
#     #     def forward(self, x1, x2):
#     #         x1 = self.layer1(x1)
#     #         x = torch.cat((x1, x2), dim=1)
#     #         x = self.layer2(x)
#     #         return x
#     #
#     # model = CustomModel()
#     # return model

#     # Input layers

#     # input1 = Input(shape=(3137 * 768,))
#     # input2 = Input(shape=(256,))
#     #
#     # # Layer 1
#     # x1 = layers.Dense(128)(input1)
#     # x1 = layers.Activation(relu)(x1)
#     # x1 = layers.Flatten()(x1)
#     # x1 = layers.Dense(128)(x1)
#     #
#     # # Concatenation
#     # x = layers.Concatenate()([x1, input2])
#     #
#     # # Layer 2
#     # x = layers.Dense(256)(x)
#     # x = layers.Activation(relu)(x)
#     # x = layers.Dense(256)(x)
#     # x = layers.Activation(relu)(x)
#     # x = layers.Dense(1)(x)
#     # x = layers.Activation(tanh)(x)
#     #
#     # # Model
#     # model = models.Model(inputs=[input1, input2], outputs=x)

#     '''model = EyeVidPre_and_ViViT.Critic_PreProcessandFreezedOutput_MobileNetV1_Model_Tf()'''

#     print("Inside critic model definition")

#     right_eye = layers.Input(shape=(1024, 7, 7))  # 1024 feature map images of size 7x7
#     left_eye = layers.Input(shape=(1024, 7, 7))  # 1024 feature map images of size 7x7
#     action = layers.Input(shape=(1, 6))

#     print("critic model, init done...")

#     # dense_layer_1 = keras.layers.Dense(128, input_shape=(50176,))
#     dense_layer_1 = keras.layers.Dense(128, input_shape=(100352,))
#     # dense_layer_1 = keras.layers.Dense(64, input_shape=(50176,))

#     dense_layer_1_conc = keras.layers.Dense(128)
#     dense_layer_2 = keras.layers.Dense(64)
#     dense_layer_3 = keras.layers.Dense(32)
#     dense_layer_parallel_1 = keras.layers.Dense(16)
#     dense_layer_parallel_2 = keras.layers.Dense(32)
#     dense_layer_comb_1 = keras.layers.Dense(32)
#     dense_layer_comb_2 = keras.layers.Dense(8)
#     dense_layer_final = keras.layers.Dense(1)

#     print("\nStart of shapes in get_critic_extra---")

#     print("critic model, layer definitions done...")

#     # Concatenation of the last hidden states
#     # last_state_concat = layers.Concatenate(axis=-3)([right_eye, left_eye])
#     last_state_concat = layers.Concatenate(axis=-1)([right_eye, left_eye])

#     print("shape of last_state_concat:", last_state_concat.shape)
#     print(f"\nshape of last_state_concat in get_critic_extra: {last_state_concat.shape}")

#     # Flatten the concatenated states
#     # concat_layer_flatten = tf.reshape(last_state_concat, (tf.shape(last_state_concat)[0], -1))
#     concat_layer_flatten = layers.Reshape((-1,))(last_state_concat)
#     print("shape of concat_layer_flatten:", concat_layer_flatten.shape)  
#     print(f"\nshape of concat_layer_flatten in get_critic_extra: {concat_layer_flatten.shape}")

#     # Pass through the first dense layer
#     output_after_dense = dense_layer_1(concat_layer_flatten)
#     print(f"\nshape of output_after_dense in get_critic_extra after dense_layer_1: {output_after_dense.shape}")
    
#     # output_after_dense = tf.reshape(output_after_dense, (-1,))
#     output_after_dense = layers.Reshape((-1,))(output_after_dense)
#     print("shape of output_after_dense:", output_after_dense.shape)
#     print(f"\nshape of output_after_dense in get_critic_extra after dense_layer_1 and reshape: {output_after_dense.shape}")

#     output_after_dense = dense_layer_1_conc(output_after_dense)
#     print(f"\nshape of output_after_dense in get_critic_extra after dense_layer_1_conc: {output_after_dense.shape}")

#     # Pass through the additional dense layers
#     output_after_dense = dense_layer_2(output_after_dense)
#     print(f"\nshape of output_after_dense in get_critic_extra after dense_layer_2: {output_after_dense.shape}")

#     output_after_dense = dense_layer_3(output_after_dense)
#     print(f"\nshape of output_after_dense in get_critic_extra after dense_layer_3: {output_after_dense.shape}")

#     # output_after_dense = tf.reshape(output_after_dense, (1, -1))
#     output_after_dense = layers.Reshape((1, -1))(output_after_dense)
#     print("shape of output_after_dense 2:", output_after_dense.shape)
#     print(f"\nshape of output_after_dense in get_critic_extra after dense_layer_3 and reshape: {output_after_dense.shape}")

#     # Parallel action processing layer in the model
#     print(f"\nshape of action in get_critic_extra for parallel: {action.shape}")

#     # Process action input through parallel dense layers
#     action_inner = dense_layer_parallel_1(action)
#     print(f"\nshape of action_inner in get_critic_extra after dense_layer_parallel_1: {action_inner.shape}")

#     action_inner = dense_layer_parallel_2(action_inner)
#     print("shape of action_inner:", action_inner.shape)
#     print(f"\nshape of action_inner in get_critic_extra after dense_layer_parallel_2: {action_inner.shape}")

#     # Combine action and output_after_dense
#     # comb = tf.concat([action, output_after_dense], axis=0)
#     # comb = layers.Concatenate(axis=0)([action_inner, output_after_dense])
#     comb = layers.Concatenate(axis=1)([action_inner, output_after_dense])

#     print("shape of comb:", comb.shape)
#     print(f"\nshape of comb in get_critic_extra after action_inner, output_after_dense concat: {comb.shape}")

#     # comb = tf.reshape(comb, (tf.shape(comb)[0], -1))
#     comb = layers.Reshape((-1,))(comb)
#     print("shape of comb 2:", comb.shape)
#     print(f"\nshape of comb in get_critic_extra after comb reshape: {comb.shape}")

#     # Apply the combined dense layers
#     comb = dense_layer_comb_1(comb)
#     print(f"\nshape of comb in get_critic_extra after dense_layer_comb_1: {comb.shape}")

#     comb = dense_layer_comb_2(comb)
#     print("shape of comb 3:", comb.shape)
#     print(f"\nshape of comb in get_critic_extra after dense_layer_comb_2: {comb.shape}")

#     # Final output
#     final_output = dense_layer_final(comb)
#     print("shape of final_output:", final_output.shape)
#     print(f"\nshape of final_output in get_critic_extra for final output: {final_output.shape}")

#     # # Doing numpy argmax now:
#     # final_output = int(np.argmax(final_output))

#     model = keras.Model([right_eye, left_eye, action], final_output)

#     print("model summary of critic model:", model.summary())

#     return model

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x

        # return torch.tensor(x)
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

num_states = 2
num_actions = 6

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(6), std_deviation=float(std_dev) * np.ones(6))

ST_GCN_Light___graph_args_inner = {"layout": 'mediapipe_face_9'}
head_mov_args_dict = {"in_channels": 4, "out_channels": 32, 
                      "graph_args": ST_GCN_Light___graph_args_inner, "edge_importance_weighting": True}

# actor_model = get_actor_extra()
# actor_model = get_actor_extra_ST_GCN()
# actor_model = get_actor_extra_ST_GCN_torch()
actor_model = actor_model_RU(head_mov_args=head_mov_args_dict, 
                             eye_proc_args=None, 
                             eye_proc_hidden_dim=None, 
                             common_hidden_dim=16, 
                             prev_iter_music_shape=7, 
                             block_size=block_size, 
                             eye_mov_mode=INPUT_MODE)

# critic_model = get_critic_extra()
# critic_model = get_critic_extra_ST_GCN()
critic_model = get_critic_extra_ST_GCN_torch()

# target_actor = get_actor_extra()
# target_actor = get_actor_extra_ST_GCN()
# target_actor = get_actor_extra_ST_GCN_torch()
target_actor = actor_model_RU(head_mov_args=head_mov_args_dict, 
                              eye_proc_args=None, 
                              eye_proc_hidden_dim=None, 
                              common_hidden_dim=16, 
                              prev_iter_music_shape=7, 
                              block_size=block_size, 
                              eye_mov_mode=INPUT_MODE)

# target_critic = get_critic_extra()
# target_critic = get_critic_extra_ST_GCN()
target_critic = get_critic_extra_ST_GCN_torch()

actor_model_GRU = get_actor_extra_RNN()

# critic_optimizer = optim.Adam(critic_model.parameters(), lr=0.001)
# actor_optimizer = optim.Adam(actor_model.parameters(), lr=0.001)

critic_lr = 0.002
actor_lr = 0.001

# critic_optimizer = keras.optimizers.Adam(critic_lr)
# actor_optimizer = keras.optimizers.Adam(actor_lr)

critic_optimizer = optim.Adam(critic_model.parameters(), critic_lr)
actor_optimizer = optim.Adam(actor_model.parameters(), actor_lr)

# critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=critic_lr)
# actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=actor_lr)

# class Buffer:
#     def __init__(self, buffer_capacity=100000, batch_size=64):
#         # Number of "experiences" to store at max
#         self.buffer_capacity = buffer_capacity
#         # Num of tuples to train on.
#         self.batch_size = batch_size

#         # Its tells us num of times record() was called.
#         self.buffer_counter = 0

#         # Instead of list of tuples as the exp.replay concept go
#         # We use different np.arrays for each tuple element

#         # self.right_eye_buffer = np.zeros((self.buffer_capacity, 1024, 7, 7))
#         # self.left_eye_buffer = np.zeros((self.buffer_capacity, 1024, 7, 7))
#         # self.next_right_eye_buffer = np.zeros((self.buffer_capacity, 1024, 7, 7))
#         # self.next_left_eye_buffer = np.zeros((self.buffer_capacity, 1024, 7, 7))

#         self.head_movement_buffer = np.zeros((self.buffer_capacity, 4, num_steps_in_batch, 9, 1))
#         self.next_head_movement_buffer = np.zeros((self.buffer_capacity, 4, num_steps_in_batch, 9, 1))

#         # action in this situation would be the output music probs
#         self.action_buffer = np.zeros((self.buffer_capacity, num_steps_in_batch // block_size, 
#                                        num_music_options))
#         self.next_action_buffer = np.zeros((self.buffer_capacity, num_steps_in_batch // block_size, 
#                                        num_music_options))

#         self.reward_buffer = np.zeros((self.buffer_capacity, 1))

#     # Takes (s,a,r,s') observation tuple as input
#     def record(self, obs_tuple):
#         # Set index to zero if buffer_capacity is exceeded,
#         # replacing old records
#         index = self.buffer_counter % self.buffer_capacity
#         print("index in Buffer:", index)

#         # print("input item in record:", obs_tuple[4])

#         # print("obs_tuple[0]:", obs_tuple[0].shape)
#         # self.right_eye_buffer[index] = obs_tuple[0]
#         # print("self.right_eye_buffer[index] in Buffer done")

#         # print("obs_tuple[1]:", obs_tuple[1].shape)
#         # self.left_eye_buffer[index] = obs_tuple[1]
#         # print("self.left_eye_buffer[index] in Buffer done")

#         # print("obs_tuple[4]:", obs_tuple[4].shape)
#         # self.next_right_eye_buffer[index] = obs_tuple[4]
#         # print("self.next_right_eye_buffer[index] in Buffer done:")

#         # print("obs_tuple[5]:", obs_tuple[5].shape)
#         # self.next_left_eye_buffer[index] = obs_tuple[5]
#         # print("self.next_left_eye_buffer[index] in Buffer done:")

#         # print("obs_tuple[0]:", obs_tuple[0].shape)
#         self.head_movement_buffer[index] = obs_tuple[0]
#         # print("self.head_movement_buffer[index] in Buffer done:")

#         # print("obs_tuple[1]:", obs_tuple[1].shape)
#         self.next_head_movement_buffer[index] = obs_tuple[1]
#         # print("self.next_head_movement_buffer[index] in Buffer done:")

#         # print("obs_tuple[2]:", obs_tuple[2].shape)
#         # print("obs_tuple[2]:", obs_tuple[2].shape)
#         self.action_buffer[index] = obs_tuple[2][0]
#         # print("self.action_buffer[index] in Buffer done:")

#         # print("obs_tuple[3]:", obs_tuple[3].shape)
#         self.next_action_buffer[index] = obs_tuple[3][0]
#         # print("self.next_action_buffer[index] in Buffer done:")

#         # print("obs_tuple[4]:", obs_tuple[4].shape)
#         # print("obs_tuple[4]:", obs_tuple[4])
#         self.reward_buffer[index] = obs_tuple[4]
#         # print("self.reward_buffer[index] in Buffer done:")

#         self.buffer_counter += 1
#         print("buffer_counter:", self.buffer_counter)

#     # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
#     # TensorFlow to build a static graph out of the logic and computations in our function.
#     # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
#     @tf.function
#     # def update(
#     #     self,
#     #     right_eye_batch,
#     #     left_eye_batch,
#     #     action_batch,
#     #     reward_batch,
#     #     next_right_eye_batch,
#     #     next_left_eye_batch,
#     # ):
#     def update(
#         self,
#         head_movement_batch,
#         next_head_movement_batch,
#         action_batch,
#         next_action_batch,
#         reward_batch,
#     ):
#         # # Training and updating Actor & Critic networks.
#         # # See Pseudo Code.

#         # print("shape of next_state_batch in Buffer update:", next_right_eye_batch.shape, next_left_eye_batch.shape)
#         print("shape of next_head_movement_batch in Buffer update:", next_head_movement_batch.shape)
#         print("shape of next_action_batch in Buffer update:", next_action_batch.shape)

#         with tf.GradientTape() as tape:
#             print("before target_actor in Buffer update")

#             # target_actions = target_actor([next_right_eye_batch, next_left_eye_batch], training=True)
#             # target_actions = target_actor(next_head_movement_batch, training=True)
#             target_actions, __ = target_actor(head_mov_x=next_head_movement_batch,
#                                               music_prev_x=next_action_batch)

#             print("target_actions shape:", reward_batch.shape, gamma, target_actions.shape)
#             # print("target_actions shape 2:", next_right_eye_batch.shape, next_left_eye_batch.shape, target_actions.shape)
#             print("target_actions shape 2:", next_head_movement_batch.shape, target_actions.shape)

#             # target_critic_output = target_critic(
#             #     [next_right_eye_batch, next_left_eye_batch, target_actions], training=True
#             # )
#             target_critic_output = target_critic(
#                 [next_head_movement_batch, target_actions], training=True
#             )

#             print(f"\nvalue of final_output form target_critic: {target_critic_output}, {type(target_critic_output)}")
#             print("shape of target_critic_output:", target_critic_output.shape, reward_batch.shape)
#             # print("shapes of objects before critic_model in Buffer update:", right_eye_batch.shape, left_eye_batch.shape, action_batch.shape)
#             print("shapes of objects before critic_model in Buffer update:", head_movement_batch.shape, action_batch.shape)
            
#             y = reward_batch + gamma * target_critic_output

#             print("shape of y:", y.shape)
#             print("first item of target_actions:", target_actions[1])
#             print("first item of target_actions:", action_batch[1])

#             # print("shape of tensors just before the critic model in update function:", right_eye_batch.shape, left_eye_batch.shape, action_batch.shape)
#             print("shape of tensors just before the critic model in update function:", head_movement_batch.shape, action_batch.shape)

#             # critic_value = critic_model([right_eye_batch, left_eye_batch, action_batch], training=True)
#             critic_value = critic_model([head_movement_batch, action_batch], training=True)

#             critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

#         critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
#         critic_optimizer.apply_gradients(
#             zip(critic_grad, critic_model.trainable_variables)
#         )

#         # with tf.GradientTape() as tape:
#         #     actions = actor_model([right_eye_batch, left_eye_batch], training=True)
#         #     critic_value = critic_model([right_eye_batch, left_eye_batch, actions], training=True)
#         #     # Used `-value` as we want to maximize the value given
#         #     # by the critic for our actions
#         #     actor_loss = -keras.ops.mean(critic_value)

#         with tf.GradientTape() as tape:
#             actions = actor_model(head_movement_batch, training=True)
#             critic_value = critic_model([head_movement_batch, actions], training=True)
#             # Used `-value` as we want to maximize the value given
#             # by the critic for our actions
#             actor_loss = -keras.ops.mean(critic_value)

#         actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
#         actor_optimizer.apply_gradients(
#             zip(actor_grad, actor_model.trainable_variables)
#         )

#         # # Training and updating Actor & Critic networks.
#         # # See Pseudo Code.
#         #
#         # print("shape of next_state_batch in Buffer update:", next_right_eye_batch.shape, next_left_eye_batch.shape)
#         #
#         # critic_optimizer.zero_grad()
#         # print("pointer actor model 4")
#         # with torch.no_grad():
#         #     target_actions = target_actor(next_right_eye_batch, next_left_eye_batch)
#         #     print("pointer critic model 4")
#         #     y = reward_batch + gamma * target_critic(next_right_eye_batch, next_left_eye_batch, target_actions)
#         #
#         # print("pointer critic model 1")
#         # critic_value = critic_model(right_eye_batch, left_eye_batch, action_batch)
#         # critic_loss = F.mse_loss(critic_value, y)
#         # critic_loss.backward()
#         # critic_optimizer.step()
#         #
#         # actor_optimizer.zero_grad()
#         # print("pointer actor model 1")
#         # actions = actor_model(right_eye_batch, left_eye_batch)
#         #
#         # print("pointer critic model 2")
#         # critic_value = critic_model(right_eye_batch, left_eye_batch, actions)
#         # actor_loss = -critic_value.mean()
#         # actor_loss.backward()
#         # actor_optimizer.step()

#     # We compute the loss and update parameters
#     def learn(self):
#         # Get sampling range
#         record_range = min(self.buffer_counter, self.buffer_capacity)
#         # Randomly sample indices
#         batch_indices = np.random.choice(record_range, self.batch_size)

#         print("batch indices:", batch_indices)

#         # Convert to tensors
#         # right_eye_batch = torch.from_numpy(self.right_eye_buffer[batch_indices])
#         # left_eye_batch = torch.from_numpy(self.left_eye_buffer[batch_indices])
#         # action_batch = torch.from_numpy(self.action_buffer[batch_indices])
#         # reward_batch = torch.from_numpy(self.reward_buffer[batch_indices])

#         # right_eye_batch = keras.ops.convert_to_tensor(self.right_eye_buffer[batch_indices])
#         # left_eye_batch = keras.ops.convert_to_tensor(self.left_eye_buffer[batch_indices])
#         head_movement_batch = keras.ops.convert_to_tensor(self.head_movement_buffer[batch_indices])

#         action_batch = keras.ops.convert_to_tensor(self.action_buffer[batch_indices])
#         reward_batch = keras.ops.convert_to_tensor(self.reward_buffer[batch_indices])

#         reward_batch = keras.ops.cast(reward_batch, dtype="float32")
#         # reward_batch = reward_batch.float()

#         print("action batch in learn function:", action_batch.shape)

#         # next_right_eye_batch = torch.from_numpy(self.next_right_eye_buffer[batch_indices])
#         # next_left_eye_batch = torch.from_numpy(self.next_left_eye_buffer[batch_indices])

#         # next_right_eye_batch = keras.ops.convert_to_tensor(self.next_right_eye_buffer[batch_indices])
#         # next_left_eye_batch = keras.ops.convert_to_tensor(self.next_left_eye_buffer[batch_indices])
#         next_head_movement_batch = keras.ops.convert_to_tensor(self.next_head_movement_buffer[batch_indices])
#         next_action_batch = keras.ops.convert_to_tensor(self.next_action_buffer[batch_indices])

#         # print("shapes in Buffer learn:", right_eye_batch.shape, left_eye_batch.shape,
#         #       action_batch.shape, reward_batch.shape,
#         #       next_right_eye_batch.shape, next_left_eye_batch.shape)
#         print("shapes in Buffer learn:", head_movement_batch.shape, next_head_movement_batch.shape,
#               action_batch.shape, next_action_batch.shape, 
#               reward_batch.shape)

#         self.update(head_movement_batch, next_head_movement_batch,
#                     action_batch, next_action_batch, 
#                     reward_batch)


# Same Buffer class in PyTorch
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.head_movement_buffer = np.zeros((self.buffer_capacity, 4, num_steps_in_batch, 9, 1))
        self.next_head_movement_buffer = np.zeros((self.buffer_capacity, 4, num_steps_in_batch, 9, 1))

        # action in this situation would be the output music probs
        self.action_buffer = np.zeros((self.buffer_capacity, num_steps_in_batch // block_size, 
                                       num_music_options))
        self.next_action_buffer = np.zeros((self.buffer_capacity, num_steps_in_batch // block_size, 
                                       num_music_options))

        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
    
    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        print("index in Buffer:", index)

        self.head_movement_buffer[index] = obs_tuple[0]
        self.next_head_movement_buffer[index] = obs_tuple[1]

        self.action_buffer[index] = obs_tuple[2][0]
        self.next_action_buffer[index] = obs_tuple[3][0]
        # self.action_buffer[index] = obs_tuple[2]
        # self.next_action_buffer[index] = obs_tuple[3]

        self.reward_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1
        print("buffer_counter:", self.buffer_counter)
    
    # @tf.function
    def update(
        self,
        head_movement_batch,
        next_head_movement_batch,
        action_batch,
        next_action_batch,
        reward_batch,
    ):
        # # Training and updating Actor & Critic networks.
        # # See Pseudo Code.

        head_movement_batch = head_movement_batch.to(torch.float32)
        next_head_movement_batch = next_head_movement_batch.to(torch.float32)
        action_batch = action_batch.to(torch.float32)
        next_action_batch = next_action_batch.to(torch.float32)
        reward_batch = reward_batch.to(torch.float32)

        # print("shape of next_state_batch in Buffer update:", next_right_eye_batch.shape, next_left_eye_batch.shape)
        print("shape of next_head_movement_batch in Buffer update:", next_head_movement_batch.shape, next_head_movement_batch.dtype)
        print("shape of next_action_batch in Buffer update:", next_action_batch.shape, next_action_batch.dtype)
        print("other dtypes:", head_movement_batch.dtype, action_batch.dtype, reward_batch.dtype)

        head_movement_batch.requires_grad_(True)
        action_batch.requires_grad_(True)

        critic_optimizer.zero_grad()

        print("before target_actor in Buffer update")
        target_actions, _ = target_actor(head_mov_x=next_head_movement_batch, 
                                        music_prev_x=next_action_batch)

        print("target_actions shape:", reward_batch.shape, gamma, target_actions.shape)
        print("target_actions shape 2:", next_head_movement_batch.shape, target_actions.shape)

        target_critic_output = target_critic([next_head_movement_batch, target_actions])

        print(f"\nvalue of final_output form target_critic: {target_critic_output}, {type(target_critic_output)}")
        print("shape of target_critic_output:", target_critic_output.shape, reward_batch.shape)

        print("shapes of objects before critic_model in Buffer update:", head_movement_batch.shape, action_batch.shape)

        y = reward_batch + gamma * target_critic_output

        print("shape of y:", y.shape)
        print("first item of target_actions:", target_actions[1])
        print("first item of action_batch:", action_batch[1])

        print("shape of tensors just before the critic model in update function:", head_movement_batch.shape, action_batch.shape)

        # critic_value = critic_model([right_eye_batch, left_eye_batch, action_batch])
        critic_value = critic_model([head_movement_batch, action_batch])
        print("done critic_model in update function 1:", y - critic_value)

        critic_loss = torch.mean((y - critic_value) ** 2)

        critic_loss.backward()
        critic_optimizer.step()

        # with tf.GradientTape() as tape:
        #     actions = actor_model([right_eye_batch, left_eye_batch], training=True)
        #     critic_value = critic_model([right_eye_batch, left_eye_batch, actions], training=True)
        #     # Used `-value` as we want to maximize the value given
        #     # by the critic for our actions
        #     actor_loss = -keras.ops.mean(critic_value)

        head_movement_batch.requires_grad_(True)
        actor_optimizer.zero_grad()

        print("done critic_model in update function 2 before:", head_movement_batch.shape, next_action_batch.shape)
        actions, _ = actor_model(head_movement_batch, next_action_batch)
        print("done actor_model in update function 2:", actions.shape)
        critic_value = critic_model([head_movement_batch, actions])
        print("done critic_model in update function 3:", critic_value)

        actor_loss = -torch.mean(critic_value)
        
        actor_loss.backward()
        actor_optimizer.step()


    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        print("batch indices:", batch_indices)
        
        head_movement_batch = self.head_movement_buffer[batch_indices] if isinstance(self.head_movement_buffer[batch_indices], torch.Tensor) else torch.tensor(self.head_movement_buffer[batch_indices])
        next_head_movement_batch = self.next_head_movement_buffer[batch_indices] if isinstance(self.next_head_movement_buffer[batch_indices], torch.Tensor) else torch.tensor(self.next_head_movement_buffer[batch_indices])

        action_batch = self.action_buffer[batch_indices] if isinstance(self.action_buffer[batch_indices], torch.Tensor) else torch.tensor(self.action_buffer[batch_indices])
        next_action_batch = self.next_action_buffer[batch_indices] if isinstance(self.next_action_buffer[batch_indices], torch.Tensor) else torch.tensor(self.next_action_buffer[batch_indices])

        reward_batch = self.reward_buffer[batch_indices] if isinstance(self.reward_buffer[batch_indices], torch.Tensor) else torch.tensor(self.reward_buffer[batch_indices])
        reward_batch = reward_batch.to(dtype=torch.float32)

        # reward_batch = reward_batch.float()

        print("action batch in learn function:", action_batch.shape)

        print("shapes in Buffer learn:", head_movement_batch.shape, next_head_movement_batch.shape,
              action_batch.shape, next_action_batch.shape, 
              reward_batch.shape)

        self.update(head_movement_batch, next_head_movement_batch,
                    action_batch, next_action_batch, 
                    reward_batch)



# This update target parameters slowly
# Based on rate `tau`, which is much less than one.

# def update_target(target, original, tau):
#     target_weights = target.get_weights()
#     original_weights = original.get_weights()

#     for i in range(len(target_weights)):
#         target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

#     target.set_weights(target_weights)

# Same update_target function using pytorch
def update_target(target, original, tau):
    for target_param, original_param in zip(target.parameters(), original.parameters()):
        target_param.data.copy_(tau * original_param.data + (1.0 - tau) * target_param.data)


# Making the weights equal initially
# target_actor.set_weights(actor_model.get_weights())
# target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

# total_episodes = 100
total_episodes = 10

# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# def policy(state, noise_object):
#     # state is the pre-processed video of the right and left eyes

#     # TODO: Uncomment to process on state
#     # ## Concatenation of inputs starts
#     # last_state_concat = np.concatenate((state[0], state[1]), axis=0)
#     # print("shape of last_state_concat:", last_state_concat.shape)
#     #
#     # concat_layer_flatten = last_state_concat.reshape(last_state_concat.shape[0], -1)
#     #
#     # sampled_actions = torch.squeeze(actor_model(concat_layer_flatten))

#     print("pointer actor model 2:", type(state))
#     print("pointer actor model 2.2:", len(state))
#     # print("shape of state received:", state.shape)

#     # state = keras.ops.expand_dims(
#     #     keras.ops.convert_to_tensor(state), 0
#     # )

#     # sampled_actions = torch.squeeze(actor_model(state[0], state[1])[0])
#     actor_model_output = actor_model(state)
#     print("actor_model_output:", actor_model_output.shape, actor_model_output)

#     sampled_actions = keras.ops.squeeze(actor_model_output)
#     # sampled_actions = keras.ops.squeeze(actor_model(state[0], state[1]))
#     print("sampled_actions from policy before noise:", sampled_actions)

#     noise = noise_object()
#     print("noise in policy:", noise)

#     # Adding noise to action
#     sampled_actions = sampled_actions.numpy() + noise
#     # sampled_actions = sampled_actions.numpy() + noise
#     print("sampled_actions from policy after noise:", sampled_actions)

#     # We make sure action is within bounds
#     legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
#     print("legal_action in policy:", legal_action)

#     return [np.squeeze(legal_action)]
#     # return legal_action

# # Takes about 4 min to train
# for ep in range(total_episodes):
#     prev_state, _ = env.reset()
#     episodic_reward = 0
#
#     while True:
#         tf_prev_state = keras.ops.expand_dims(
#             torch.from_numpy(prev_state), 0
#         )
#
#         action = policy(tf_prev_state, ou_noise)
#         # Receive state and reward from environment.
#         state, reward, done, truncated, _ = env.step(action)
#
#         buffer.record((prev_state, action, reward, state))
#         episodic_reward += reward
#
#         buffer.learn()
#
#         update_target(target_actor, actor_model, tau)
#         update_target(target_critic, critic_model, tau)
#
#         # End this episode when `done` or `truncated` is True
#         if done or truncated:
#             break
#
#         prev_state = state
#
#     ep_reward_list.append(episodic_reward)
#
#     # # Mean of last 40 episodes
#     # avg_reward = np.mean(ep_reward_list[-40:])
#     # print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
#     # avg_reward_list.append(avg_reward)

'''Actual code to be uncommented for running DDPG'''
# for ep in range(total_episodes):
#     # right_eye, left_eye, reward = EyeCropping.camera_input()
#
#     print("reached in the main loop")
#
#     right_eye, left_eye, reward = EyeCropping.camera_input_2()
#
#     # # _, softmaxed_layer = EyeVidPre_and_ViViT.PreProcessandFreezedOutput(
#     # #     right_eye, left_eye)
#     # # # print("last_layer_from_upper_branch:", _)
#     # # print("softmaxed_layer:", softmaxed_layer.shape)
#     #
#     # _, softmaxed_layer = EyeVidPre_and_ViViT.PreProcessandFreezedOutput_MobileNetV1(
#     #     right_eye, left_eye)
#     # # print("last_layer_from_upper_branch:", _)
#     # print("softmaxed_layer:", softmaxed_layer.shape)
#     #
#     # prev_state = softmaxed_layer
#     #
#     # episodic_reward = 0
#     #
#     # # while True:
#     # if True:
#     #     # tf_prev_state = keras.ops.expand_dims(
#     #     #     torch.from_numpy(prev_state), 0
#     #     # )
#     #     # action = policy(tf_prev_state, ou_noise)
#     #     action = policy(prev_state, ou_noise)
#     #
#     #     state = None
#     #
#     #     buffer.record((prev_state, action, reward, state))
#     #     episodic_reward += reward
#     #
#     #     buffer.learn()
#     #
#     #     update_target(target_actor, actor_model, tau)
#     #     update_target(target_critic, critic_model, tau)
#     #
#     #     # prev_state = state
#     #
#     # ep_reward_list.append(episodic_reward)

# Plotting graph
# Episodes versus Avg. Rewards

''' Test code for running video input and model inference in threading'''

# class NumberedThread(threading.Thread):
#     def __init__(self, frame_queue, thread_number):
#         super().__init__()
#         self.frame_queue = frame_queue
#         self.thread_number = thread_number
#         self.running = True
#
#     def run(self):
#         while self.running:
#             try:
#                 # Try to get a frame from the queue with a timeout to avoid blocking indefinitely
#                 frames_array_item = self.frame_queue.get(timeout=1)
#                 # Print the thread number instead of processing the frame
#                 print(f"Thread {self.thread_number} is processing a frame")
#                 print(f"data from thread {self.thread_number}:", frames_array_item.shape)
#
#                 # Simulate some processing time
#                 time.sleep(0.1)
#
#             except queue.Empty:
#                 continue
#
#     def stop(self):
#         self.running = False

def calculate_temporal_smoothness_reward(stability_scores):
    stability_scores = np.array(stability_scores)
    stability_diff = np.diff(stability_scores)
    temporal_smoothness_penalty = np.sum(np.square(stability_diff))
    temporal_smoothness_reward = -temporal_smoothness_penalty

    return temporal_smoothness_reward

# def play_music():
#     current_music = None

#     while True:
#         try:
#             # Check if there is a new music file in the queue
#             new_music = music_queue.get_nowait()  # Non-blocking check
#             if new_music is not None:
#                 print("new_music just after getting from queue:", new_music)

#                 if new_music != current_music:
#                     pygame.mixer.music.stop()  # Stop current music
#                     pygame.mixer.music.load(new_music)
#                     pygame.mixer.music.play(-1)  # Play indefinitely
#                     print(f"Playing new music: {new_music}")
#                     current_music = new_music
#             else:
#                 pygame.mixer.music.stop()
#         except queue.Empty as e:
#             print("mp 1, error in music queue handling in play_music function:", e)
#             pass  # No new music in the queue

#         # Continue the loop and let the music play
#         pygame.time.Clock().tick(10)  # Small delay to prevent CPU overload

#     # pygame.mixer.init()
#     # pygame.mixer.music.load(file_path)
#     # pygame.mixer.music.play(-1)  # -1 means play indefinitely
#     # while pygame.mixer.music.get_busy():
#     #     time.sleep(1)  # Check periodically if the music is still playing

import pygame
import queue
import time

# Initialize pygame mixer safely
pygame.mixer.init()

# Music queue
music_queue = queue.Queue()

def play_music():
    current_music = None

    while True:
        try:
            # Try to get a new music file from the queue (non-blocking)
            new_music = music_queue.get_nowait()

            if new_music and new_music != current_music:
                print(f"Trying to play: {new_music}")

                # Stop current playback
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    time.sleep(0.1)  # Allow hardware to settle

                try:
                    pygame.mixer.music.load(new_music)
                    pygame.mixer.music.play(-1)  # Loop indefinitely
                    print(f"Now playing: {new_music}")
                    current_music = new_music
                except pygame.error as e:
                    print(f"Error loading/playing {new_music}: {e}")
        except queue.Empty:
            # No new music; continue playing current
            pass

        time.sleep(0.1)  # Small delay to prevent CPU hogging

chosen_music_array = []
music_index_iter_array = []

def video_capture_and_stuff(path_to_json_root = "/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DDPG with new changes"):
    # app = QApplication(sys.argv)

    ht__hidden_dimension = 16
    num_music_options = 7
    num_RU_blocks = 6

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

    prev_head_movement = None
    curr_head_movement = None

    def calculate_angle(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        return np.degrees(angle)

    BRIGHTNESS_THRESHOLD = 100  # You can adjust this value based on testing

    def adjust_brightness(frame):
        # Convert the frame to HSV color space
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Calculate the average brightness (V channel)
        brightness = np.mean(hsv[:, :, 2])

        # Check if the brightness is below the threshold
        if brightness < BRIGHTNESS_THRESHOLD:
            # Increase the brightness of the V channel
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + (BRIGHTNESS_THRESHOLD - brightness), 0, 255)
            # Convert back to BGR format
            frame = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        return frame

    # Task1:

    # drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    # resized_re_t_before = None
    # resized_le_t_before = None

    # resized_re_OTSU_before = None
    # resized_le_OTSU_before = None

    # resized_re_KMeans_before = None
    # resized_le_KMeans_before = None

    # eye_right_cropped_model_input = []
    # eye_left_cropped_model_input = []

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
    cap = cv.VideoCapture(0)

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
        
        # if RUN_MODE == 0:
        #     with mp_holistic.Holistic(
        #                         static_image_mode=False,
        #                         model_complexity=1,
        #                         smooth_landmarks=True,
        #                         min_detection_confidence=0.5,
        #                         min_tracking_confidence=0.5) as holistic:

        #         for ep in range(num_frames*total_episodes):
        #             if block_time_started is False:
        #                 block_start_time = time.time()
        #                 block_time_started = True

        #             face_coords = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}

        #             success, img2 = cap.read()

        #             if (success) and (img2 is not None):
        #                 img = adjust_brightness(img2)
                        
        #                 # if flag_red_sig is True:
        #                 if flag_user_rep_done is False:
        #                     # if (time.time() - green_signal) < 5:
        #                     if steps_batch_counter < num_steps_in_batch:
        #                             if success:
        #                                 cv.putText(img2, f"Frames left {num_steps_in_batch - steps_batch_counter}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        #                                 steps_batch_counter += 1
        #                                 # cv.putText(img2, str(5 - (int(time.time() - green_signal))), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        #                                 # print("flag_green_sig is still False: ", 5 - (int(time.time() - green_signal)), time.time(), green_signal)
                                    
        #                             else:
        #                                 raise Exception("Error in initial video feed")
                                
        #                     else:
        #                         flag_user_rep_done = True
        #                         print("flag_user_rep_done is now True")

        #                 right_eye_main = []
        #                 left_eye_main = []

        #                 try:
        #                     if (ep % block_size) == 0:
        #                         decided_music_index = music_one_hot_vectors[0][(ep // block_size) % len(music_files)]
        #                         music_file_temp = music_files[decided_music_index]
        #                         print("music_file_temp in music queue:", music_file_temp)
                                
        #                         # music_queue.put(music_file_temp)

        #                         block_end_time = time.time()
        #                         music_times__per_block.append(block_end_time - block_start_time)
        #                         block_start_time = time.time()

        #                         block_time_started = False
        #                 except Exception as music_playing_error:
        #                     print("error in sending music through queue:", music_playing_error)

        #                 # with mp_face_mesh.FaceMesh(
        #                 #         static_image_mode=True,
        #                 #         min_detection_confidence=0.5) as face_mesh:
        #                     # mp_pose.Pose(
        #                     #     static_image_mode=False, 
        #                     #     model_complexity=1, 
        #                     #     smooth_landmarks=True, 
        #                     #     enable_segmentation=False, 
        #                     #     min_detection_confidence=0.5, 
        #                     #     min_tracking_confidence=0.5) as pose:
                        
        #                 # with mp_holistic.Holistic(
        #                 #         static_image_mode=False,
        #                 #         model_complexity=1,
        #                 #         smooth_landmarks=True,
        #                 #         min_detection_confidence=0.5,
        #                 #         min_tracking_confidence=0.5) as holistic:
                        

        #                 # Convert the BGR image to RGB before processing.
        #                 img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        #                 results = holistic.process(img_rgb)

        #                 # total result
        #                 # if INPUT_MODE == 0:
        #                 # if True:
        #                 #     with mp_holistic.Holistic(
        #                 #                 static_image_mode=False,
        #                 #                 model_complexity=1,
        #                 #                 smooth_landmarks=True,
        #                 #                 min_detection_confidence=0.5,
        #                 #                 min_tracking_confidence=0.5) as holistic:
                                
        #                 #         results = holistic.process(img_rgb)
        #                 # else:
        #                 #     with mp_pose.Pose(
        #                 #             static_image_mode=False, 
        #                 #             model_complexity=1, 
        #                 #             smooth_landmarks=True, 
        #                 #             enable_segmentation=False, 
        #                 #             min_detection_confidence=0.5, 
        #                 #             min_tracking_confidence=0.5) as pose:
                                
        #                 #         results = pose.process(img_rgb)
                        
        #                 if INPUT_MODE != 0:
        #                 # if True:
        #                     # Draw face landmarks (without connections)
        #                     if results.face_landmarks:
        #                         mp_drawing.draw_landmarks(img2, results.face_landmarks)

        #                     # Draw pose landmarks
        #                     if results.pose_landmarks:
        #                         mp_drawing.draw_landmarks(img2, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        #                 else:
        #                     pose_landmarks = results.pose_landmarks

        #                     if pose_landmarks:
        #                         # mp_drawing.draw_landmarks(img2, pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        #                         for key in face_coords:
        #                             t = pose_landmarks.landmark[key]
        #                             face_coords[key] = [t.x, t.y, t.z, t.visibility]
                                
        #                         N_M = []

        #                         for key in face_coords:
        #                             x, y, z, vis = face_coords[key]
        #                             N_M.append([x, y, z, vis])

        #                             x *= img2.shape[1]
        #                             x = int(x)
                                    
        #                             y *= img2.shape[0]
        #                             y = int(y)

        #                             cv.circle(img2, (x, y), 3, (255, 0, 0), 3)
                                
        #                         # print("N_M:", N_M)
                                
        #                         T_N_M.append(N_M)

        #                 # # results_2 = face_mesh.process(img_rgb)
        #                 # results_2 = results.face_landmarks
        #                 # # pose_result = pose.process(img_rgb)

        #                 # face_landmarks = results_2.multi_face_landmarks[0]
        #                 # landmarks = {i: (lm.x, lm.y) for i, lm in enumerate(face_landmarks.landmark)}
                    

        #                 # # Generate the sequence using this starting pose:
        #                 # # if len(VRNN_pose_input) != 0:

        #                 # if (flag_green_sig is True) and (flag_red_sig is True) and (flag_generated_motion_once is False):
        #                 #     if prev_state is not None:
        #                 #         print("reached the condition where not the first iteration first:", type(pose_seq_user_rep_5s), len(pose_seq_user_rep_5s))
        #                 #         print("reached the condition where not the first iteration second:", pose_seq_user_rep_5s)

        #                 #     if flag_sig_movement_demo is False:
        #                 #         sig_movement_demo = time.time()
        #                 #         flag_sig_movement_demo = True
                            
        #                 #     print("pointer before getting user input for demo reaction...")

        #                 #     if ((time.time() - sig_movement_demo) < 5) and (flag_do_demo_warn is False):
        #                 #         print("inside the demo condition:", time.time() - sig_movement_demo)
        #                 #         cv.putText(img2, "Get ready to copy the movement, you will get a small amount of time to replicate the shown dance motion", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        #                 #         cv.putText(img2, "Message will disappear in " + str(5 - (int(time.time() - sig_movement_demo))), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        #                 #     else:
        #                 #         print("inside the demo condition else timings:", time.time(), sig_movement_demo)
        #                 #         flag_do_demo_warn = True
                            
        #                 #     if flag_do_demo_warn is True:
        #                 #         remaining_frames = len(VRNN_seq_gen) - len(pose_seq_user_rep)

        #                 #         if flag_user_rep_done is False:
        #                 #             print("inside the user input for demo repetition condition", len(pose_seq_user_rep), len(VRNN_seq_gen), "ep:", ep)
        #                 #             cv.putText(img2, "Number of frames left to capture...", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        #                 #             cv.putText(img2, str(remaining_frames), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)

        #                 #             print("remaining_frames is:", remaining_frames)

        #                 #             if remaining_frames == 0:
        #                 #                 print("remaining_frames is 0")
        #                 #             elif len(VRNN_seq_gen) - len(pose_seq_user_rep) == 1:
        #                 #                 print("remaining_frames is 1")
                                
        #                 #         # if (flag_user_rep_done is True) or (remaining_frames == 0):
        #                 #         #     print("inside the condition to compare the dance motions")
        #                 #         #     # The code which takes care of the comparison
        #                 #         #     pose_estimator = dance_comp_Dyuthi.PoseEstimator()

        #                 #         #     pose_seq_similarity_score = pose_estimator.compare_kps(pose_seq_user_rep, VRNN_seq_gen)['average']
        #                 #         #     print("pose_seq_similarity_score:", pose_seq_similarity_score)

        #                 # # MARKER:  at eye processing for visual attention... - do later...
        #                 # if flag_do_demo_warn is True:
        #                 #     # # cv.imwrite(f"/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DDPG with new changes/{episode_counter}_{(ep + 1) if episode_counter == 0 else (int(ep / episode_counter) - 59)}.jpg", img)
        #                 #     # # cv.imwrite(f"/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DDPG with new changes/{episode_counter}_{len(pose_seq_user_rep)}.jpg", img)
        #                 #     # cv.imwrite(f"/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DDPG with new changes/{len(pose_seq_user_rep)}.jpg", img)

        #                 #     # with open(os.path.join(path_to_json_root, f"keypoints_data_{len(pose_seq_user_rep)}.json"), "w") as json_file:
        #                 #     #     dictionary_temp = {"ep": episode_counter,
        #                 #     #                     #    "frame_num": (ep + 1) if episode_counter == 0 else (int(ep / episode_counter) - 59),
        #                 #     #                        "frame_num": len(pose_seq_user_rep),
        #                 #     #                        "right_eye_main_dict": right_eye_main_dict,
        #                 #     #                        "left_eye_main_dict": left_eye_main_dict,
        #                 #     #                        "regions_re": regions_re,
        #                 #     #                        "regions_le": regions_le,
        #                 #     #                        "right_upper_lobe": right_upper_lobe,
        #                 #     #                        "right_lower_lobe": right_lower_lobe,
        #                 #     #                        "left_upper_lobe": left_upper_lobe,
        #                 #     #                        "left_lower_lobe": left_lower_lobe,
        #                 #     #                        "image_file_path": f"/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DDPG with new changes/{len(pose_seq_user_rep)}.jpg"}
        #                 #     #                     #    "image_file_path": f"/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DDPG with new changes/{episode_counter}_{len(pose_seq_user_rep)}.jpg"}
        #                 #     #                     #    "image_file_path": f"/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DDPG with new changes/{episode_counter}_{(ep + 1) if episode_counter == 0 else (int(ep / episode_counter) - 59)}.jpg"}
                                
        #                 #     #     json.dump(dictionary_temp, json_file)

        #                 #     # MARKER:  store the 3/5 keypoints for ST_GCN
        #                 #     xs = []
        #                 #     ys = []
        #                 #     zs = []
        #                 #     viss = []

        #                 #     # for face_key in face_coords:
        #                 #     #     print("face_key data point:", face_coords[face_key])
        #                 #     #     xs.append([face_coords[face_key][0]])
        #                 #     #     ys.append([face_coords[face_key][1]])
        #                 #     #     zs.append([face_coords[face_key][2]])
        #                 #     #     viss.append([face_coords[face_key][3]])

        #                 #     # TODO: Uncomment this for inference
        #                 #     if num_frames != 0:
        #                 #         # # MARKER:  Store the frames and other per-frame data in the JSON file

        #                 #         num_frames -= 1

        #                 #     else:
        #                 #         print("plain_raw_video size:", plain_raw_video.size())

        #                 #         frame_arr = []
        #                 #         plain_raw_video = [[], [], [], []]

        #                 #         print("video window num:", window_num)
        #                 #         window_num += 1
        #                 #         # window_num = 1

        #                 #         print("---------- flag values and time values before reset ----------")
        #                 #         print("flag_red_sig:", flag_red_sig)
        #                 #         print("flag_green_sig:", flag_green_sig)
        #                 #         print("green_sig_init_once:", green_sig_init_once)
        #                 #         print("flag_generated_motion_once:", flag_generated_motion_once)
        #                 #         print("flag_movement_demo:", flag_movement_demo)
        #                 #         print("flag_sig_movement_demo:", flag_sig_movement_demo)
        #                 #         print("sig_movement_demo:", sig_movement_demo)
        #                 #         print("flag_do_demo_warn:", flag_do_demo_warn)
        #                 #         print("flag_take_usr_input_rep_demo:", flag_take_usr_input_rep_demo)
        #                 #         print("flag_user_rep_done:", flag_user_rep_done)
        #                 #         print("pose_seq_similarity_score:", pose_seq_similarity_score)
        #                 #         print("red_signal:", red_signal)
        #                 #         print("green_signal:", green_signal)

        #                 #         # Re-init of the marker variables
        #                 #         flag_red_sig = False
        #                 #         flag_green_sig = False

        #                 #         green_sig_init_once = False
        #                 #         flag_generated_motion_once = False
        #                 #         flag_movement_demo = False
        #                 #         flag_sig_movement_demo = False
        #                 #         sig_movement_demo = None
        #                 #         flag_do_demo_warn = False
        #                 #         flag_take_usr_input_rep_demo = False
        #                 #         flag_user_rep_done = False
        #                 #         flag_shown_GIF_loop_once = False

        #                 #         pose_seq_similarity_score = None

        #                 #         VRNN_pose_input = []

        #                 #         pose_seq_user_rep_5s = pose_seq_user_rep[len(pose_seq_user_rep) - 5:]
        #                 #         pose_seq_user_rep = []

        #                 #         red_signal = time.time()
        #                 #         green_signal = time.time()

        #                         # print("---------- flag values and time values after reset ----------")
        #                         # print("flag_red_sig:", flag_red_sig)
        #                         # print("flag_green_sig:", flag_green_sig)
        #                         # print("green_sig_init_once:", green_sig_init_once)
        #                         # print("flag_generated_motion_once:", flag_generated_motion_once)
        #                         # print("flag_movement_demo:", flag_movement_demo)
        #                         # print("flag_sig_movement_demo:", flag_sig_movement_demo)
        #                         # print("sig_movement_demo:", sig_movement_demo)
        #                         # print("flag_do_demo_warn:", flag_do_demo_warn)
        #                         # print("flag_take_usr_input_rep_demo:", flag_take_usr_input_rep_demo)
        #                         # print("flag_user_rep_done:", flag_user_rep_done)
        #                         # print("pose_seq_similarity_score:", pose_seq_similarity_score)
        #                         # print("red_signal:", red_signal)
        #                         # print("green_signal:", green_signal)

        #                 #         episode_counter += 1

        #                 # print("---------- flag values and time values before reset ----------")
        #                 # print("flag_user_rep_done:", flag_user_rep_done)
        #                 # # print("flag_green_sig:", flag_green_sig)
        #                 # # print("flag_red_sig:", flag_red_sig)

        #                 if flag_user_rep_done is True:
        #                     # music_queue.put(None)

        #                     if True:
        #                         flag_user_rep_done = False
        #                         steps_batch_counter = 0
        #                         # flag_red_sig = False
        #                         # flag_green_sig = False

        #                         T_N_M = T_N_M[:-1]

        #                         print("T_N_M:", torch.tensor(T_N_M).shape)

        #                         face_9_keypoints.append(T_N_M)
        #                         # print("face_9_keypoints:", face_9_keypoints)

        #                         T_N_M = []

        #                         face_9_keypoints = torch.tensor(face_9_keypoints)
        #                         print("face_9_keypoints shape 1st:", face_9_keypoints.shape)

        #                         face_9_keypoints = torch.unsqueeze(face_9_keypoints, 0)
        #                         print("face_9_keypoints shape 2nd:", face_9_keypoints.shape)

        #                         B, P, T, N, M = face_9_keypoints.shape
        #                         print("dims:", B, P, T, N, M)

        #                         face_9_keypoints = face_9_keypoints.permute(0, 4, 2, 3, 1)
        #                         print("face_9_keypoints shape 3rd:", face_9_keypoints.shape)
        #                         print("----------------------------------------------------")
                            
        #                     if prev_head_movement is not None:
        #                         curr_head_movement = face_9_keypoints
        #                         print("music_probs__t_0:", music_probs__t_0)

        #                         # Using the model to get the output::
        #                         music_rec_probs_temp, ht_0 = actor_model(head_mov_x=curr_head_movement, 
        #                                                                     music_prev_x=music_probs__t_0,
        #                                                                     ht__0=ht_0)
        #                         print("music_rec_probs_temp:", music_rec_probs_temp.shape)
        #                         print("ht_0:", ht_0.shape)

        #                         try:
        #                             # music_probs__t_0 = torch.tensor(music_rec_probs_temp)

        #                             music_probs__t_0_temp = torch.stack(music_rec_probs_temp)
        #                             music_probs__t_0_temp = torch.unsqueeze(music_probs__t_0_temp, dim=0)

        #                             # music_probs__t_0 = music_rec_probs_temp
        #                         except Exception as e_RU_recursion:
        #                             print("error in RU recursion:", e_RU_recursion)
        #                             music_probs__t_0_temp = music_rec_probs_temp
                                
        #                         try:
        #                             # buffer.record((np.array(prev_head_movement), np.array(curr_head_movement),
        #                             #             music_probs__t_0.clone(), music_probs__t_0_temp.clone(),
        #                             #             reward))

        #                             # print(type(music_probs__t_0))
        #                             # print(type(music_probs__t_0_temp))

        #                             buffer.record((prev_head_movement.detach().numpy(), curr_head_movement.detach().numpy(),
        #                                         music_probs__t_0.clone().detach().numpy(), music_probs__t_0_temp.clone().detach().numpy(),
        #                                         reward))
        #                             print("recorded in buffer")
        #                         except Exception as e__buffer_record_exception:
        #                             print("problem in buffer record function:", e__buffer_record_exception)
                                
        #                         try:
        #                             buffer.learn()
        #                         except Exception as e_buffer_learn:
        #                             print("Error in buffer learning:", e_buffer_learn)
                                
        #                         try:
        #                             music_probs__t_0 = music_probs__t_0_temp.clone()
        #                         except Exception as e_cloning_error:
        #                             print("Error in pytorch tensor cloning:", e_cloning_error)
                                
        #                         try:
        #                             # print("pointer actor model 3")
        #                             update_target(target_actor, actor_model, tau)
        #                             # print("pointer 32 after reward_avg")
        #                         except Exception as e__target_actor__update:
        #                             print("Error in updating the target_actor model:", e__target_actor__update)
                                
        #                         try:
        #                             # print("pointer critic model 3")
        #                             update_target(target_critic, critic_model, tau)
        #                             # print("pointer 33 after reward_avg")
        #                         except Exception as e__target_critic__update:
        #                             print("Error in updating the target_critic model:", e__target_critic__update)
                                
        #                         prev_head_movement = curr_head_movement
        #                     else:
        #                         prev_head_movement = face_9_keypoints

        #                     green_signal = time.time()

        #                     face_9_keypoints = []
        #                     reward = 0

        #                     # max_indices = music_probs__t_0.argmax(dim=-1)  # Shape: [1, 6]
        #                     # one_hot = F.one_hot(max_indices, num_classes=7)  # Shape: [1, 6, 7]
        #                     # one_hot = one_hot.float()
        #                     # music_one_hot_vectors = one_hot.argmax(dim=-1)

        #                     # print("music_times__per_block:", music_times__per_block)
        #                     # print("music_one_hot_vectors:", music_one_hot_vectors)
        #                     # print("music_probs__t_0:", music_probs__t_0)

        #                     # spectro_combined.combine_spectros(music_one_hot_vectors, ep)
        #                     # combined_music_pydub.combine_musics(music_one_hot_vectors, ep, block_duration=block_duration)
        #                     music_index_iter = combined_music_pydub.vanilla_music_selector(music_probs__t_0) + 1
        #                     print("music_index_iter:", music_index_iter)

        #                     music_times__per_block = []
                        
        #                 # print("---------- flag values and time values after reset ----------")
        #                 # print("flag_user_rep_done:", flag_user_rep_done)
        #                 # # print("flag_green_sig:", flag_green_sig)
        #                 # # print("flag_red_sig:", flag_red_sig)

        #             # cv.imshow("img", img)
        #             cv.imshow("img2", img2)
        #             # print("shape of image:", img.shape)

        #             # if (frame_queue.full()) or (cv.waitKey(1) & 0xFF == ord('q')):
        #             if False or (cv.waitKey(1) & 0xFF == ord('q')):
        #                 break

        #             end_time_total = time.time()

        #             # print("total time taken:", end_time_total - start_time_total)

        #             # for ep in range(total_episodes):
        #             #     _, __, ___ = EyeCropping.camera_input_2()
        #             #     print("window num:", window_num)
        #             #     window_num += 1

        #                 # ret, frame = cap.read()
        #                 # if not ret:
        #                 #     break
        #                 #
        #                 # if num_frames != 0:
        #                 #     frame_arr.append(frame)
        #                 #     num_frames -= 1
        #                 # else:
        #                 #     # Add the frame to the queue for processing, blocking if queue is full
        #                 #     frame_queue.put(frame_arr, block=True)
        #                 #     num_frames = 32
        #                 #     frame_arr = []
        #                 #
        #                 #     # print("window number:", window_num)
        #                 #     #
        #                 #     # window_num += 1
        #                 #
        #                 # # Display the original frame
        #                 # cv.imshow('Original Frame', frame)
        #                 #
        #                 # # Give some time for the threads to process
        #                 # # time.sleep(0.05)  # Adjust this to balance between frame capture and processing
        #                 #
        #                 # if (frame_queue.full()) or (cv.waitKey(1) & 0xFF == ord('q')):
        #                 #     break

        #             # vid_end_time = time.time()
        #             # print("total time:", vid_end_time - vid_start_time)
        # elif RUN_MODE == 1:
        if RUN_MODE == 1:
            success, img2 = cap.read()

            nose_point_start_time = None
            nose_point_start_flag = False

            face_coords = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}
            rand_kps_1 = random_selection.tester_call_function(music_index_iter)

            for k in rand_kps_1.keys():
                face_coords[int(k)] = rand_kps_1[k]

            if success:
                for ep in range(num_frames*total_episodes):
                    if block_time_started is False:
                        block_start_time = time.time()
                        block_time_started = True
                    
                    if nose_point_start_flag is False:
                        nose_point_start_flag = True
                        nose_point_start_time = time.time()
                    
                    if time.time() - nose_point_start_time >= head_mov_delay_time:
                        face_coords = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}
                        rand_kps_1 = random_selection.tester_call_function(music_index_iter)

                        for k in rand_kps_1.keys():
                            face_coords[int(k)] = rand_kps_1[k]
                        
                        nose_point_start_flag = False
                        nose_point_start_time = time.time()

                    success, img2 = cap.read()

                    if (success) and (img2 is not None):
                        img = adjust_brightness(img2)
                        
                        # if flag_red_sig is True:
                        if flag_user_rep_done is False:
                            # if (time.time() - green_signal) < 5:
                            if steps_batch_counter < num_steps_in_batch:
                                    if success:
                                        cv.putText(img2, f"Frames left {num_steps_in_batch - steps_batch_counter}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
                                        steps_batch_counter += 1
                                        # cv.putText(img2, str(5 - (int(time.time() - green_signal))), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
                                        # print("flag_green_sig is still False: ", 5 - (int(time.time() - green_signal)), time.time(), green_signal)
                                    
                                    else:
                                        raise Exception("Error in initial video feed")
                                
                            else:
                                flag_user_rep_done = True
                                print("flag_user_rep_done is now True")

                        right_eye_main = []
                        left_eye_main = []

                        try:
                            if (ep % block_size) == 0:
                                decided_music_index = music_one_hot_vectors[0][(ep // block_size) % len(music_files)]
                                music_file_temp = music_files[decided_music_index]
                                # print("music_file_temp in music queue:", music_file_temp)
                                
                                # music_queue.put(music_file_temp)

                                block_end_time = time.time()
                                music_times__per_block.append(block_end_time - block_start_time)
                                block_start_time = time.time()

                                block_time_started = False
                        except Exception as music_playing_error:
                            print("error in sending music through queue:", music_playing_error)

                        # Convert the BGR image to RGB before processing.
                        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                        # results = holistic.process(img_rgb)
                        
                        # if INPUT_MODE != 0:
                        # # if True:
                        #     # Draw face landmarks (without connections)
                        #     if results.face_landmarks:
                        #         mp_drawing.draw_landmarks(img2, results.face_landmarks)

                        #     # Draw pose landmarks
                        #     if results.pose_landmarks:
                        #         mp_drawing.draw_landmarks(img2, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                        # else:
                        if True:
                            # pose_landmarks = results.pose_landmarks

                            # if pose_landmarks:
                            if True:
                                # # mp_drawing.draw_landmarks(img2, pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                                # for key in face_coords:
                                #     t = pose_landmarks.landmark[key]
                                #     face_coords[key] = [t.x, t.y, t.z, t.visibility]
                                
                                N_M = []

                                for key in face_coords:
                                    x, y, z, vis = face_coords[key]
                                    N_M.append([x, y, z, vis])

                                    x *= img2.shape[1]
                                    x = int(x)
                                    
                                    y *= img2.shape[0]
                                    y = int(y)

                                    cv.circle(img2, (x, y), 3, (255, 0, 0), 3)
                                
                                # print("N_M:", N_M)
                                
                                T_N_M.append(N_M)

                        if flag_user_rep_done is True:
                            # music_queue.put(None)

                            if True:
                                flag_user_rep_done = False
                                steps_batch_counter = 0
                                # flag_red_sig = False
                                # flag_green_sig = False

                                T_N_M = T_N_M[:-1]

                                print("T_N_M:", torch.tensor(T_N_M).shape)

                                face_9_keypoints.append(T_N_M)
                                # print("face_9_keypoints:", face_9_keypoints)

                                T_N_M = []

                                face_9_keypoints = torch.tensor(face_9_keypoints)
                                print("face_9_keypoints shape 1st:", face_9_keypoints.shape)

                                face_9_keypoints = torch.unsqueeze(face_9_keypoints, 0)
                                print("face_9_keypoints shape 2nd:", face_9_keypoints.shape)

                                B, P, T, N, M = face_9_keypoints.shape
                                print("dims:", B, P, T, N, M)

                                face_9_keypoints = face_9_keypoints.permute(0, 4, 2, 3, 1)
                                print("face_9_keypoints shape 3rd:", face_9_keypoints.shape)
                                print("----------------------------------------------------")
                            
                            if prev_head_movement is not None:
                                curr_head_movement = face_9_keypoints
                                print("music_probs__t_0:", music_probs__t_0)

                                # Using the model to get the output::
                                music_rec_probs_temp, ht_0 = actor_model(head_mov_x=curr_head_movement, 
                                                                            music_prev_x=music_probs__t_0,
                                                                            ht__0=ht_0)
                                print("music_rec_probs_temp:", music_rec_probs_temp.shape)
                                print("ht_0:", ht_0.shape)

                                try:
                                    # music_probs__t_0 = torch.tensor(music_rec_probs_temp)

                                    music_probs__t_0_temp = torch.stack(music_rec_probs_temp)
                                    music_probs__t_0_temp = torch.unsqueeze(music_probs__t_0_temp, dim=0)

                                    # music_probs__t_0 = music_rec_probs_temp
                                except Exception as e_RU_recursion:
                                    print("error in RU recursion:", e_RU_recursion)
                                    music_probs__t_0_temp = music_rec_probs_temp
                                
                                try:
                                    # buffer.record((np.array(prev_head_movement), np.array(curr_head_movement),
                                    #             music_probs__t_0.clone(), music_probs__t_0_temp.clone(),
                                    #             reward))

                                    # print(type(music_probs__t_0))
                                    # print(type(music_probs__t_0_temp))

                                    chosen_music, reward = random_selection.tester_call_function__reward(music_index_iter)

                                    buffer.record((prev_head_movement.detach().numpy(), curr_head_movement.detach().numpy(),
                                                music_probs__t_0.clone().detach().numpy(), music_probs__t_0_temp.clone().detach().numpy(),
                                                reward))
                                    print("recorded in buffer")
                                except Exception as e__buffer_record_exception:
                                    print("problem in buffer record function:", e__buffer_record_exception)
                                
                                try:
                                    buffer.learn()
                                except Exception as e_buffer_learn:
                                    print("Error in buffer learning:", e_buffer_learn)
                                
                                try:
                                    music_probs__t_0 = music_probs__t_0_temp.clone()
                                except Exception as e_cloning_error:
                                    print("Error in pytorch tensor cloning:", e_cloning_error)
                                
                                try:
                                    # print("pointer actor model 3")
                                    update_target(target_actor, actor_model, tau)
                                    # print("pointer 32 after reward_avg")
                                except Exception as e__target_actor__update:
                                    print("Error in updating the target_actor model:", e__target_actor__update)
                                
                                try:
                                    # print("pointer critic model 3")
                                    update_target(target_critic, critic_model, tau)
                                    # print("pointer 33 after reward_avg")
                                except Exception as e__target_critic__update:
                                    print("Error in updating the target_critic model:", e__target_critic__update)
                                
                                prev_head_movement = curr_head_movement
                            else:
                                prev_head_movement = face_9_keypoints

                            green_signal = time.time()

                            face_9_keypoints = []
                            reward = 0

                            print("chosen_music,music_index_iter:", chosen_music, music_index_iter)
                            chosen_music_array.append(chosen_music)
                            music_index_iter_array.append(music_index_iter)
                            music_index_iter = combined_music_pydub.vanilla_music_selector(music_probs__t_0)
                            # print("music_index_iter:", music_index_iter)

                            music_times__per_block = []

                    # cv.imshow("img", img)
                    cv.imshow("img2", img2)
                    # print("shape of image:", img.shape)

                    # if (frame_queue.full()) or (cv.waitKey(1) & 0xFF == ord('q')):
                    if False or (cv.waitKey(1) & 0xFF == ord('q')):
                        break

                    end_time_total = time.time()

    except Exception as e_video_input_threading:
        print("Error in video input threading...:", e_video_input_threading)
        pass

    # finally:
    #     # Clean up
    #     cap.release()
    #     cv.destroyAllWindows()

        # pass

        # for thread in threads:
        #     thread.stop()
        #     thread.join()

# plt.plot(avg_reward_list)
# plt.xlabel("Episode")
# plt.ylabel("Avg. Episodic Reward")
# plt.show()

# # Save the weights
# actor_model.save_weights("pendulum_actor.weights.h5")
# critic_model.save_weights("pendulum_critic.weights.h5")
#
# target_actor.save_weights("pendulum_target_actor.weights.h5")
# target_critic.save_weights("pendulum_target_critic.weights.h5")

def main():
    # music_thread = threading.Thread(target=play_music, args=("Musics/music1.mp3",))
    # music_thread.start()
    #
    # # Start video capture
    # music_thread = threading.Thread(target=play_music)
    # music_thread.daemon = True  # Ensure the thread exits when the main program exits
    # music_thread.start()

    video_capture_and_stuff()

    # # Stop music after video ends
    # pygame.mixer.music.stop()

    x = [_ for _ in range(len(chosen_music_array))]

    plt.plot(x, chosen_music_array, label='Tester Chosen Music Index', color='blue')
    plt.plot(x, music_index_iter_array, label='Selected Music Index', color='red')

    plt.title('Comparison of music index preferred by tester and chosen by model')
    plt.xlabel('Episode')
    plt.ylabel('Music Index')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
