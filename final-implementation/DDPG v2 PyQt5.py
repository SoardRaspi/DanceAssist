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

import tkinter as tk
from PIL import Image, ImageTk, ImageSequence
from itertools import count

from transformers import VivitImageProcessor, VivitForVideoClassification, \
    AutoImageProcessor, MobileNetV1Model
from DanceGenerator import dance_comp_Dyuthi
from DanceGenerator.DyuthiVRNN import modelv1

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QTimer

import pygame

import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

try:
    dance_gen_model_path = "/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DanceGenerator/models/vrnn_trial_2__only_top__epoch14.pth"
    dance_gen_model_path = "/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/DanceGenerator/models/vrnn_trial_2__only_top__epoch19 right dataset.pth"

    model_seq_gen = modelv1.get_model(dance_gen_model_path)
    print("dance generator model:", type(model_seq_gen))
except:
    model_seq_gen = None
    raise Exception("Error initializing the Dance Generator model")

pygame.mixer.init()

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

image_processor_1 = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
model_1 = MobileNetV1Model.from_pretrained("google/mobilenet_v1_1.0_224")

# Load GIF
root = tk.Tk()
root.title("GIF Player")

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
    num_frames = 32
    # shape_single_frame = raw_video_input[0][0].shape

    # print("num_frames, shape_single_frame:", num_frames, shape_single_frame)

    print("\nStart of shapes in get_actor_extra_RNN---", file=open(log_file, 'a'))

    right_eye_seq = layers.Input(shape=(num_frames, 32, 
                                        128, 3))
    left_eye_seq = layers.Input(shape=(num_frames, 32, 
                                       128, 3))
    
    print(f"\n shape of right_eye_seq: {right_eye_seq.shape}", file=open(log_file, 'a'))
    
    # TimeDistributed CNN for spatial feature extraction
    x = layers.TimeDistributed(
        tf.keras.Sequential([
            layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.Flatten()
        ]),
        name="TimeDistributed_CNN"
    )(right_eye_seq)  # Shape: (batch_size, sequence_length, features)
    print(f"\n shape of x 1: {x.shape}", file=open(log_file, 'a'))

    # 3-layer GRU for temporal sequence modeling
    x = layers.GRU(64, return_sequences=True, activation="tanh", name="GRU_Layer_1")(x)
    print(f"\n shape of x 2: {x.shape}", file=open(log_file, 'a'))

    x = layers.GRU(64, return_sequences=True, activation="tanh", name="GRU_Layer_2")(x)
    print(f"\n shape of x 3: {x.shape}", file=open(log_file, 'a'))

    x_right_eye = layers.GRU(64, return_sequences=False, activation="tanh", name="GRU_Layer_3")(x)
    print(f"\n shape of x_right_eye: {x_right_eye.shape}", file=open(log_file, 'a'))

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
    print(f"\n shape of dense_right_eye: {dense_right_eye.shape}", file=open(log_file, 'a'))

    model = keras.Model(inputs=right_eye_seq, outputs=dense_right_eye)
    model.summary()

    return model

def get_actor_extra():
    # model = nn.Sequential(
    #     nn.Linear(3137 * 768, 128),
    #     nn.ReLU(),
    #     nn.Flatten(),
    #     nn.Linear(256, 128),
    #     nn.Softmax()
    # )

    '''model = EyeVidPre_and_ViViT.Actor_PreProcessandFreezedOutput_MobileNetV1_Model_Tf()'''

    # model = models.Sequential([
    #     layers.Dense(128, input_shape=(3137 * 768,)),
    #     layers.Activation(relu),
    #     layers.Flatten(),
    #     layers.Dense(128),
    #     layers.Activation(softmax)
    # ])

    print("\nStart of shapes in get_actor_extra---", file=open(log_file, 'a'))

    right_eye = layers.Input(shape=(1024, 7, 7))
    left_eye = layers.Input(shape=(1024, 7, 7))

    dense_layer_1 = layers.Dense(128, input_shape=(2*50176,))
    dense_layer_1_conc = layers.Dense(128)
    dense_layer_2 = layers.Dense(64)
    dense_layer_3 = layers.Dense(6)
    softmax = layers.Softmax()
    sigmoid = layers.Activation('sigmoid')

    last_state_concat = layers.Concatenate(axis=-1)([right_eye, left_eye])
    print("shape of last_state_concat in get_actor_extra:", last_state_concat.shape)
    print(f"\nshape of last_state_concat in get_actor_extra: {last_state_concat.shape}", file=open(log_file, 'a'))

    # concat_layer_flatten = tf.reshape(last_state_concat, (tf.shape(last_state_concat)[0], -1))
    concat_layer_flatten = layers.Reshape((-1,))(last_state_concat)
    print("shape of concat_layer_flatten in get_actor_extra:", concat_layer_flatten.shape)
    print(f"\nshape of concat_layer_flatten in get_actor_extra: {concat_layer_flatten.shape}", file=open(log_file, 'a'))

    output_after_dense = dense_layer_1(concat_layer_flatten)
    print(f"\nshape of output_after_dense in get_actor_extra after dense_layer_1: {output_after_dense.shape}", file=open(log_file, 'a'))

    # output_after_dense = tf.reshape(output_after_dense, (-1,))
    output_after_dense = layers.Reshape((-1,))(output_after_dense)
    print(f"\nshape of output_after_dense in get_actor_extra after reshaping 1: {output_after_dense.shape}", file=open(log_file, 'a'))

    # output_after_dense = tf.expand_dims(output_after_dense, axis=0)
    output_after_dense = layers.Reshape((1, -1))(output_after_dense)
    print(f"\nshape of output_after_dense in get_actor_extra after reshaping 2: {output_after_dense.shape}", file=open(log_file, 'a'))

    output_after_dense = dense_layer_1_conc(output_after_dense)
    print(f"\nshape of output_after_dense in get_actor_extra after dense_layer_1_conc: {output_after_dense.shape}", file=open(log_file, 'a'))

    output_after_dense = dense_layer_2(output_after_dense)
    print(f"\nshape of output_after_dense in get_actor_extra after dense_layer_2: {output_after_dense.shape}", file=open(log_file, 'a'))

    output_after_dense = dense_layer_3(output_after_dense)
    print(f"\nshape of output_after_dense in get_actor_extra after dense_layer_3: {output_after_dense.shape}", file=open(log_file, 'a'))

    final_output = softmax(output_after_dense)
    print(f"\nshape of final_output in get_actor_extra: {final_output.shape}", file=open(log_file, 'a'))
    print("\nEnd of shapes in get_actor_extra---", file=open(log_file, 'a'))

    model = keras.Model([right_eye, left_eye], final_output)
    return model

def get_critic_extra():
    # class CustomModel(nn.Module):
    #     def __init__(self):
    #         super(CustomModel, self).__init__()
    #         self.layer1 = nn.Sequential(
    #             nn.Linear(3137 * 768, 128),
    #             nn.ReLU(),
    #             nn.Flatten(),
    #             nn.Linear(256, 128)
    #         )
    #         self.layer2 = nn.Sequential(
    #             nn.Linear(256, 256),
    #             nn.ReLU(),
    #             nn.Linear(256, 256),
    #             nn.ReLU(),
    #             nn.Linear(256, 1),
    #             nn.Tanh()
    #         )
    #
    #     def forward(self, x1, x2):
    #         x1 = self.layer1(x1)
    #         x = torch.cat((x1, x2), dim=1)
    #         x = self.layer2(x)
    #         return x
    #
    # model = CustomModel()
    # return model

    # Input layers

    # input1 = Input(shape=(3137 * 768,))
    # input2 = Input(shape=(256,))
    #
    # # Layer 1
    # x1 = layers.Dense(128)(input1)
    # x1 = layers.Activation(relu)(x1)
    # x1 = layers.Flatten()(x1)
    # x1 = layers.Dense(128)(x1)
    #
    # # Concatenation
    # x = layers.Concatenate()([x1, input2])
    #
    # # Layer 2
    # x = layers.Dense(256)(x)
    # x = layers.Activation(relu)(x)
    # x = layers.Dense(256)(x)
    # x = layers.Activation(relu)(x)
    # x = layers.Dense(1)(x)
    # x = layers.Activation(tanh)(x)
    #
    # # Model
    # model = models.Model(inputs=[input1, input2], outputs=x)

    '''model = EyeVidPre_and_ViViT.Critic_PreProcessandFreezedOutput_MobileNetV1_Model_Tf()'''

    print("Inside critic model definition")

    right_eye = layers.Input(shape=(1024, 7, 7))  # 1024 feature map images of size 7x7
    left_eye = layers.Input(shape=(1024, 7, 7))  # 1024 feature map images of size 7x7
    action = layers.Input(shape=(1, 6))

    print("critic model, init done...")

    # dense_layer_1 = keras.layers.Dense(128, input_shape=(50176,))
    dense_layer_1 = keras.layers.Dense(128, input_shape=(100352,))
    # dense_layer_1 = keras.layers.Dense(64, input_shape=(50176,))

    dense_layer_1_conc = keras.layers.Dense(128)
    dense_layer_2 = keras.layers.Dense(64)
    dense_layer_3 = keras.layers.Dense(32)
    dense_layer_parallel_1 = keras.layers.Dense(16)
    dense_layer_parallel_2 = keras.layers.Dense(32)
    dense_layer_comb_1 = keras.layers.Dense(32)
    dense_layer_comb_2 = keras.layers.Dense(8)
    dense_layer_final = keras.layers.Dense(1)

    print("\nStart of shapes in get_critic_extra---", file=open(log_file, 'a'))

    print("critic model, layer definitions done...")

    # Concatenation of the last hidden states
    # last_state_concat = layers.Concatenate(axis=-3)([right_eye, left_eye])
    last_state_concat = layers.Concatenate(axis=-1)([right_eye, left_eye])

    print("shape of last_state_concat:", last_state_concat.shape)
    print(f"\nshape of last_state_concat in get_critic_extra: {last_state_concat.shape}", file=open(log_file, 'a'))

    # Flatten the concatenated states
    # concat_layer_flatten = tf.reshape(last_state_concat, (tf.shape(last_state_concat)[0], -1))
    concat_layer_flatten = layers.Reshape((-1,))(last_state_concat)
    print("shape of concat_layer_flatten:", concat_layer_flatten.shape)  
    print(f"\nshape of concat_layer_flatten in get_critic_extra: {concat_layer_flatten.shape}", file=open(log_file, 'a'))

    # Pass through the first dense layer
    output_after_dense = dense_layer_1(concat_layer_flatten)
    print(f"\nshape of output_after_dense in get_critic_extra after dense_layer_1: {output_after_dense.shape}", file=open(log_file, 'a'))
    
    # output_after_dense = tf.reshape(output_after_dense, (-1,))
    output_after_dense = layers.Reshape((-1,))(output_after_dense)
    print("shape of output_after_dense:", output_after_dense.shape)
    print(f"\nshape of output_after_dense in get_critic_extra after dense_layer_1 and reshape: {output_after_dense.shape}", file=open(log_file, 'a'))

    output_after_dense = dense_layer_1_conc(output_after_dense)
    print(f"\nshape of output_after_dense in get_critic_extra after dense_layer_1_conc: {output_after_dense.shape}", file=open(log_file, 'a'))

    # Pass through the additional dense layers
    output_after_dense = dense_layer_2(output_after_dense)
    print(f"\nshape of output_after_dense in get_critic_extra after dense_layer_2: {output_after_dense.shape}", file=open(log_file, 'a'))

    output_after_dense = dense_layer_3(output_after_dense)
    print(f"\nshape of output_after_dense in get_critic_extra after dense_layer_3: {output_after_dense.shape}", file=open(log_file, 'a'))

    # output_after_dense = tf.reshape(output_after_dense, (1, -1))
    output_after_dense = layers.Reshape((1, -1))(output_after_dense)
    print("shape of output_after_dense 2:", output_after_dense.shape)
    print(f"\nshape of output_after_dense in get_critic_extra after dense_layer_3 and reshape: {output_after_dense.shape}", file=open(log_file, 'a'))

    # Parallel action processing layer in the model
    print(f"\nshape of action in get_critic_extra for parallel: {action.shape}", file=open(log_file, 'a'))

    # Process action input through parallel dense layers
    action_inner = dense_layer_parallel_1(action)
    print(f"\nshape of action_inner in get_critic_extra after dense_layer_parallel_1: {action_inner.shape}", file=open(log_file, 'a'))

    action_inner = dense_layer_parallel_2(action_inner)
    print("shape of action_inner:", action_inner.shape)
    print(f"\nshape of action_inner in get_critic_extra after dense_layer_parallel_2: {action_inner.shape}", file=open(log_file, 'a'))

    # Combine action and output_after_dense
    # comb = tf.concat([action, output_after_dense], axis=0)
    # comb = layers.Concatenate(axis=0)([action_inner, output_after_dense])
    comb = layers.Concatenate(axis=1)([action_inner, output_after_dense])

    print("shape of comb:", comb.shape)
    print(f"\nshape of comb in get_critic_extra after action_inner, output_after_dense concat: {comb.shape}", file=open(log_file, 'a'))

    # comb = tf.reshape(comb, (tf.shape(comb)[0], -1))
    comb = layers.Reshape((-1,))(comb)
    print("shape of comb 2:", comb.shape)
    print(f"\nshape of comb in get_critic_extra after comb reshape: {comb.shape}", file=open(log_file, 'a'))

    # Apply the combined dense layers
    comb = dense_layer_comb_1(comb)
    print(f"\nshape of comb in get_critic_extra after dense_layer_comb_1: {comb.shape}", file=open(log_file, 'a'))

    comb = dense_layer_comb_2(comb)
    print("shape of comb 3:", comb.shape)
    print(f"\nshape of comb in get_critic_extra after dense_layer_comb_2: {comb.shape}", file=open(log_file, 'a'))

    # Final output
    final_output = dense_layer_final(comb)
    print("shape of final_output:", final_output.shape)
    print(f"\nshape of final_output in get_critic_extra for final output: {final_output.shape}", file=open(log_file, 'a'))

    # # Doing numpy argmax now:
    # final_output = int(np.argmax(final_output))

    model = keras.Model([right_eye, left_eye, action], final_output)

    print("model summary of critic model:", model.summary())

    return model
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

actor_model = get_actor_extra()
critic_model = get_critic_extra()

target_actor = get_actor_extra()
target_critic = get_critic_extra()

actor_model_GRU = get_actor_extra_RNN()

# critic_optimizer = optim.Adam(critic_model.parameters(), lr=0.001)
# actor_optimizer = optim.Adam(actor_model.parameters(), lr=0.001)

critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = keras.optimizers.Adam(critic_lr)
actor_optimizer = keras.optimizers.Adam(actor_lr)

# critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=critic_lr)
# actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=actor_lr)

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.right_eye_buffer = np.zeros((self.buffer_capacity, 1024, 7, 7))
        self.left_eye_buffer = np.zeros((self.buffer_capacity, 1024, 7, 7))
        self.action_buffer = np.zeros((self.buffer_capacity, 1, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_right_eye_buffer = np.zeros((self.buffer_capacity, 1024, 7, 7))
        self.next_left_eye_buffer = np.zeros((self.buffer_capacity, 1024, 7, 7))

    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        print("index in Buffer:", index)

        print("obs_tuple[0]:", obs_tuple[0].shape)
        self.right_eye_buffer[index] = obs_tuple[0]
        print("self.right_eye_buffer[index] in Buffer done")

        print("obs_tuple[1]:", obs_tuple[1].shape)
        self.left_eye_buffer[index] = obs_tuple[1]
        print("self.left_eye_buffer[index] in Buffer done")

        # print("obs_tuple[2]:", obs_tuple[2].shape)
        print("obs_tuple[2]:", obs_tuple[2])
        self.action_buffer[index] = obs_tuple[2]
        print("self.action_buffer[index] in Buffer done:")

        # print("obs_tuple[3]:", obs_tuple[3].shape)
        print("obs_tuple[3]:", obs_tuple[3])
        self.reward_buffer[index] = obs_tuple[3]
        print("self.reward_buffer[index] in Buffer done:")

        print("obs_tuple[4]:", obs_tuple[4].shape)
        self.next_right_eye_buffer[index] = obs_tuple[4]
        print("self.next_right_eye_buffer[index] in Buffer done:")

        print("obs_tuple[5]:", obs_tuple[5].shape)
        self.next_left_eye_buffer[index] = obs_tuple[5]
        print("self.next_left_eye_buffer[index] in Buffer done:")

        self.buffer_counter += 1
        print("buffer_counter:", self.buffer_counter)

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self,
        right_eye_batch,
        left_eye_batch,
        action_batch,
        reward_batch,
        next_right_eye_batch,
        next_left_eye_batch,
    ):
        # # Training and updating Actor & Critic networks.
        # # See Pseudo Code.

        print("shape of next_state_batch in Buffer update:", next_right_eye_batch.shape, next_left_eye_batch.shape)

        with tf.GradientTape() as tape:
            print("before target_actor in Buffer update")
            target_actions = target_actor([next_right_eye_batch, next_left_eye_batch], training=True)
            print("target_actions shape:", reward_batch.shape, gamma, target_actions.shape)
            print("target_actions shape 2:", next_right_eye_batch.shape,
                  next_left_eye_batch.shape, target_actions.shape)
            target_critic_output = target_critic(
                [next_right_eye_batch, next_left_eye_batch, target_actions], training=True
            )
            print(f"\nvalue of final_output form target_critic: {target_critic_output}, {type(target_critic_output)}", file=open(log_file, 'a'))
            print("shape of target_critic_output:", target_critic_output.shape, reward_batch.shape)
            print("shapes of objects before critic_model in Buffer update:", right_eye_batch.shape,
                  left_eye_batch.shape, action_batch.shape)
            
            y = reward_batch + gamma * target_critic_output

            print("shape of y:", y.shape)
            print("first item of target_actions:", target_actions[1])
            print("first item of target_actions:", action_batch[1])

            print("shape of tensors just before the critic model in update function:", right_eye_batch.shape, left_eye_batch.shape, action_batch.shape)
            critic_value = critic_model([right_eye_batch, left_eye_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model([right_eye_batch, left_eye_batch], training=True)
            critic_value = critic_model([right_eye_batch, left_eye_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

        # # Training and updating Actor & Critic networks.
        # # See Pseudo Code.
        #
        # print("shape of next_state_batch in Buffer update:", next_right_eye_batch.shape, next_left_eye_batch.shape)
        #
        # critic_optimizer.zero_grad()
        # print("pointer actor model 4")
        # with torch.no_grad():
        #     target_actions = target_actor(next_right_eye_batch, next_left_eye_batch)
        #     print("pointer critic model 4")
        #     y = reward_batch + gamma * target_critic(next_right_eye_batch, next_left_eye_batch, target_actions)
        #
        # print("pointer critic model 1")
        # critic_value = critic_model(right_eye_batch, left_eye_batch, action_batch)
        # critic_loss = F.mse_loss(critic_value, y)
        # critic_loss.backward()
        # critic_optimizer.step()
        #
        # actor_optimizer.zero_grad()
        # print("pointer actor model 1")
        # actions = actor_model(right_eye_batch, left_eye_batch)
        #
        # print("pointer critic model 2")
        # critic_value = critic_model(right_eye_batch, left_eye_batch, actions)
        # actor_loss = -critic_value.mean()
        # actor_loss.backward()
        # actor_optimizer.step()

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        print("batch indices:", batch_indices)

        # Convert to tensors
        # right_eye_batch = torch.from_numpy(self.right_eye_buffer[batch_indices])
        # left_eye_batch = torch.from_numpy(self.left_eye_buffer[batch_indices])
        # action_batch = torch.from_numpy(self.action_buffer[batch_indices])
        # reward_batch = torch.from_numpy(self.reward_buffer[batch_indices])

        right_eye_batch = keras.ops.convert_to_tensor(self.right_eye_buffer[batch_indices])
        left_eye_batch = keras.ops.convert_to_tensor(self.left_eye_buffer[batch_indices])
        action_batch = keras.ops.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = keras.ops.convert_to_tensor(self.reward_buffer[batch_indices])

        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        # reward_batch = reward_batch.float()

        print("action batch in learn function:", action_batch.shape)

        # next_right_eye_batch = torch.from_numpy(self.next_right_eye_buffer[batch_indices])
        # next_left_eye_batch = torch.from_numpy(self.next_left_eye_buffer[batch_indices])

        next_right_eye_batch = keras.ops.convert_to_tensor(self.next_right_eye_buffer[batch_indices])
        next_left_eye_batch = keras.ops.convert_to_tensor(self.next_left_eye_buffer[batch_indices])

        print("shapes in Buffer learn:", right_eye_batch.shape, left_eye_batch.shape,
              action_batch.shape, reward_batch.shape,
              next_right_eye_batch.shape, next_left_eye_batch.shape)

        self.update(right_eye_batch, left_eye_batch,
                    action_batch, reward_batch,
                    next_right_eye_batch, next_left_eye_batch)

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.

def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)

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

def policy(state, noise_object):
    # state is the pre-processed video of the right and left eyes

    # TODO: Uncomment to process on state
    # ## Concatenation of inputs starts
    # last_state_concat = np.concatenate((state[0], state[1]), axis=0)
    # print("shape of last_state_concat:", last_state_concat.shape)
    #
    # concat_layer_flatten = last_state_concat.reshape(last_state_concat.shape[0], -1)
    #
    # sampled_actions = torch.squeeze(actor_model(concat_layer_flatten))

    print("pointer actor model 2")
    # print("shape of state received:", state.shape)

    # state = keras.ops.expand_dims(
    #     keras.ops.convert_to_tensor(state), 0
    # )

    # sampled_actions = torch.squeeze(actor_model(state[0], state[1])[0])
    actor_model_output = actor_model([state[0], state[1]])
    print("actor_model_output:", actor_model_output.shape, actor_model_output)

    sampled_actions = keras.ops.squeeze(actor_model_output)
    # sampled_actions = keras.ops.squeeze(actor_model(state[0], state[1]))
    print("sampled_actions from policy before noise:", sampled_actions)

    noise = noise_object()
    print("noise in policy:", noise)

    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise
    # sampled_actions = sampled_actions.numpy() + noise
    print("sampled_actions from policy after noise:", sampled_actions)

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    print("legal_action in policy:", legal_action)

    return [np.squeeze(legal_action)]
    # return legal_action

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

def play_music():
    current_music = None

    while True:
        try:
            # Check if there is a new music file in the queue
            new_music = music_queue.get_nowait()  # Non-blocking check
            print("new_music just after getting from queue:", new_music)

            if new_music != current_music:
                pygame.mixer.music.stop()  # Stop current music
                pygame.mixer.music.load(new_music)
                pygame.mixer.music.play(-1)  # Play indefinitely
                print(f"Playing new music: {new_music}")
                current_music = new_music
        except queue.Empty as e:
            print("mp 1, error in music queue handling in play_music function:", e)
            pass  # No new music in the queue

        # Continue the loop and let the music play
        pygame.time.Clock().tick(10)  # Small delay to prevent CPU overload

    # pygame.mixer.init()
    # pygame.mixer.music.load(file_path)
    # pygame.mixer.music.play(-1)  # -1 means play indefinitely
    # while pygame.mixer.music.get_busy():
    #     time.sleep(1)  # Check periodically if the music is still playing

def concat_images(frames):
    print("frames details:", frames.shape)

    len_frames = frames.shape[0]
    empty_frame = np.zeros((frames.shape[1], frames.shape[2], 3), dtype=np.float64)

    print("empty_frame details:", empty_frame.shape)

    for frame in frames:
        empty_frame += frame/(256*len_frames)

    # empty_frames = np.expand_dims(empty_frames, axis=-1)

    np.clip(empty_frame, 0, 255).astype(np.uint8)

    print("empty_frame details another:", empty_frame.shape)

    return empty_frame

def video_capture_and_stuff():
    app = QApplication(sys.argv)

    window = GifPlayer("/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/animation_from.gif")
    window.show()

    green_sig_init_once = False
    flag_generated_motion_once = False
    flag_movement_demo = False
    flag_sig_movement_demo = False
    sig_movement_demo = None
    flag_do_demo_warn = False
    flag_take_usr_input_rep_demo = False
    flag_user_rep_done = False

    pose_seq_similarity_score = None

    # model_actor_local = EyeVidPre_and_ViViT.Model(5)
    music_folder = "Musics"  # Folder where music files are stored
    music_files = [os.path.join(music_folder, f) for f in os.listdir(music_folder) if f.endswith('.mp3')]

    motion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # count = 1

    right_eye_video = []
    left_eye_video = []

    right_eye_main = []
    right_eye_main_dict = {33: None, 246: None, 161: None, 160: None, 159: None, 158: None, 157: None,
                           173: None, 133: None, 155: None, 154: None, 153: None, 145: None, 144: None,
                           163: None, 7: None}
    r_lobe_coords = {130: None, 247: None, 30: None, 29: None, 27: None, 28: None, 56: None, 190: None, 243: None,
                     25: None,
                     110: None, 24: None, 23: None, 22: None, 26: None, 112: None}
    regions_re = {1: [33, 246, 7], 2: [246, 7, 163, 161], 3: [161, 163, 144, 160], 4: [160, 144, 145, 159],
                  5: [159, 145, 153, 158], 6: [158, 153, 154, 157], 7: [157, 154, 155, 173], 8: [173, 155, 133]}
    right_upper_lobe = {1: [247, 246, 33, 130], 2: [247, 30, 161, 246], 3: [30, 29, 160, 161], 4: [29, 27, 159, 160],
                        5: [27, 28, 158, 159], 6: [28, 56, 157, 158], 7: [56, 190, 173, 157], 8: [190, 243, 133, 173]}
    right_lower_lobe = {1: [130, 33, 7, 25], 2: [7, 163, 110, 25], 3: [163, 144, 24, 110], 4: [144, 145, 23, 24],
                        5: [145, 153, 22, 23], 6: [153, 154, 26, 22], 7: [154, 155, 112, 26], 8: [155, 133, 243, 112]}

    left_eye_main = []
    left_eye_main_dict = {263: None, 466: None, 388: None, 387: None, 386: None, 385: None, 384: None,
                          398: None, 362: None, 382: None, 381: None, 380: None, 374: None, 373: None,
                          390: None, 249: None}
    l_lobe_coords = {463: None, 414: None, 286: None, 258: None, 257: None, 259: None, 260: None, 467: None, 359: None,
                     255: None, 339: None, 254: None, 253: None, 252: None, 256: None, 341: None}
    regions_le = {1: [263, 466, 249], 2: [466, 249, 390, 388], 3: [388, 390, 373, 387], 4: [387, 373, 374, 386],
                  5: [386, 374, 380, 385], 6: [385, 380, 381, 384], 7: [384, 381, 382, 398], 8: [398, 382, 362]}
    left_upper_lobe = {1: [359, 263, 466, 467], 2: [467, 466, 388, 260], 3: [260, 388, 387, 259],
                       4: [259, 387, 386, 257],
                       5: [257, 386, 385, 258], 6: [258, 385, 384, 286], 7: [286, 384, 398, 414],
                       8: [414, 398, 362, 463]}
    left_lower_lobe = {1: [359, 263, 249, 255], 2: [255, 249, 390, 339], 3: [339, 390, 373, 254],
                       4: [254, 373, 374, 253],
                       5: [253, 374, 380, 252], 6: [252, 380, 381, 256], 7: [256, 381, 382, 341],
                       8: [341, 382, 362, 463]}

    def KMeans_Image(image, K=3):
        print("inside KMeans_Image", image.shape)

        Z = image.reshape((-1, 2))
        Z = np.float32(Z)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image.shape))

        return res2

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

    drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    resized_re_t_before = None
    resized_le_t_before = None

    resized_re_OTSU_before = None
    resized_le_OTSU_before = None

    resized_re_KMeans_before = None
    resized_le_KMeans_before = None

    eye_right_cropped_model_input = []
    eye_left_cropped_model_input = []

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

    plain_raw_video = [[], []]  # [[], []] = [(right-eye), (left-eye)]

    cap = cv.VideoCapture(0)

    num_frames = 32

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

    flag_red_sig = False
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

        for ep in range(num_frames*total_episodes):

            regions_re = {1: [33, 246, 7], 2: [246, 7, 163, 161], 3: [161, 163, 144, 160], 4: [160, 144, 145, 159],
                          5: [159, 145, 153, 158], 6: [158, 153, 154, 157], 7: [157, 154, 155, 173], 8: [173, 155, 133]}
            regions_le = {1: [263, 466, 249], 2: [466, 249, 390, 388], 3: [388, 390, 373, 387], 4: [387, 373, 374, 386],
                          5: [386, 374, 380, 385], 6: [385, 380, 381, 384], 7: [384, 381, 382, 398], 8: [398, 382, 362]}

            r_lobe_coords = {130: None, 247: None, 30: None, 29: None, 27: None, 28: None, 56: None, 190: None, 243: None,
                             25: None,
                             110: None, 24: None, 23: None, 22: None, 26: None, 112: None}
            right_upper_lobe = {1: [247, 246, 33, 130], 2: [247, 30, 161, 246], 3: [30, 29, 160, 161],
                                4: [29, 27, 159, 160],
                                5: [27, 28, 158, 159], 6: [28, 56, 157, 158], 7: [56, 190, 173, 157],
                                8: [190, 243, 133, 173]}
            right_lower_lobe = {1: [130, 33, 7, 25], 2: [7, 163, 110, 25], 3: [163, 144, 24, 110], 4: [144, 145, 23, 24],
                                5: [145, 153, 22, 23], 6: [153, 154, 26, 22], 7: [154, 155, 112, 26],
                                8: [155, 133, 243, 112]}

            l_lobe_coords = {463: None, 414: None, 286: None, 258: None, 257: None, 259: None, 260: None, 467: None,
                             359: None,
                             255: None, 339: None, 254: None, 253: None, 252: None, 256: None, 341: None}
            left_upper_lobe = {1: [359, 263, 466, 467], 2: [467, 466, 388, 260], 3: [260, 388, 387, 259],
                               4: [259, 387, 386, 257],
                               5: [257, 386, 385, 258], 6: [258, 385, 384, 286], 7: [286, 384, 398, 414],
                               8: [414, 398, 362, 463]}
            left_lower_lobe = {1: [359, 263, 249, 255], 2: [255, 249, 390, 339], 3: [339, 390, 373, 254],
                               4: [254, 373, 374, 253],
                               5: [253, 374, 380, 252], 6: [252, 380, 381, 256], 7: [256, 381, 382, 341],
                               8: [341, 382, 362, 463]}

            percents_re = []
            percents_le = []

            success, img2 = cap.read()
            # img_for_eye = img.copy()
            # imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            # results = pose.process(imgRGB)

            # flag_re = False
            # flag_le = False
            # flag_n = False
            #
            # right_eye = None
            # left_eye = None

            # focus_points = [None, None, None, None, None]
            # #                ri, ro, li, lo, n
            #
            # face_points = [None, None, None]
            # #              lm,  c, rm
            #
            # bounds_right = [None, None, None, None]
            # #                 ur,   lr,   rc,   lc
            # bounds_left = [None, None, None, None]
            # #                 ur,   lr,   rc,   lc

            if (success) and (img2 is not None):
                img = adjust_brightness(img2)

                # cv.imshow("simple plain video input", img)

                # try:
                if True:
                    if flag_red_sig is False:
                        print("just before the flag_red_sig main if statement", (time.time() - red_signal), flag_red_sig)
                    
                    if ((time.time() - red_signal) < 3) and (flag_red_sig is False):
                        # print("inside the flag_red_sig main if statement")
                        if success:
                            cv.putText(img2, "Please stand around 2-3 meters from the screen", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                            cv.putText(img2, str(3 - (int(time.time() - red_signal))), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                        
                        else:
                            raise Exception("Error in initial video feed")
                    
                    else:
                        flag_red_sig = True

                        if green_sig_init_once is False:
                            green_signal =  time.time()
                            green_sig_init_once = True
                        
                        print("green_signal time init: ", green_signal)
                    
                    user_starting_pose = []

                    # if flag_red_sig is True:
                    #     print("flag_red_sig is true now true now")
                    # else:
                    #     print("flag_red_sig is false now false now")
                    
                    # if flag_green_sig is True:
                    #     print("flag_green_sig is true now true now")
                    # else:
                    #     print("flag_green_sig is false now false now")
                    
                    if (flag_red_sig is True) and (flag_green_sig is False):
                        if (time.time() - green_signal) < 5:
                            if success:
                                cv.putText(img2, "Getting your current position", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                                cv.putText(img2, str(5 - (int(time.time() - green_signal))), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                                # print("flag_green_sig is still False: ", 5 - (int(time.time() - green_signal)), time.time(), green_signal)
                            
                            else:
                                raise Exception("Error in initial video feed")
                        
                        else:
                            flag_green_sig = True
                            print("flag_green_sig is now True")
                # except Exception as e_:
                #     print("the error in boolean setting part of code:", e_)
                
                # if flag_red_sig is True:
                if True:
                    right_eye_main = []
                    left_eye_main = []

                    with mp_face_mesh.FaceMesh(
                            static_image_mode=True,
                            min_detection_confidence=0.5) as face_mesh, \
                        mp_pose.Pose(
                            static_image_mode=False, 
                            model_complexity=1, 
                            smooth_landmarks=True, 
                            enable_segmentation=False, 
                            min_detection_confidence=0.5, 
                            min_tracking_confidence=0.5) as pose:
                        
                        if True:
                            # Convert the BGR image to RGB before processing.
                            results_2 = face_mesh.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                            pose_result = pose.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

                            # Print and draw face mesh landmarks on the image.
                            if not results_2.multi_face_landmarks:
                                continue

                            print(len(results_2.multi_face_landmarks))

                            if pose_result is not None:
                                pose_major_joints = pose_result.pose_landmarks.landmark

                                mpDraw.draw_landmarks(
                                    img,
                                    pose_result.pose_landmarks,
                                    mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                    connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                )

                                print("type of pose_major_joints:", type(pose_major_joints))

                                # right arm: (12, 14, 16) -> shoulder, elbow, wrist
                                # left arm: (11, 13, 15) -> shoulder, elbow, wrist
                                # right leg: (24, 26, 28) -> hip, knee, ankle
                                # left leg: (23, 25, 27) -> hip, knee, ankle

                                right_arm = [
                                    np.array([pose_major_joints[12].x, pose_major_joints[12].y]),
                                    np.array([pose_major_joints[14].x, pose_major_joints[14].y]),
                                    np.array([pose_major_joints[16].x, pose_major_joints[16].y]),
                                ]
                                left_arm = [
                                    np.array([pose_major_joints[11].x, pose_major_joints[11].y]),
                                    np.array([pose_major_joints[13].x, pose_major_joints[13].y]),
                                    np.array([pose_major_joints[15].x, pose_major_joints[15].y]),
                                ]
                                right_leg = [
                                    np.array([pose_major_joints[24].x, pose_major_joints[24].y]),
                                    np.array([pose_major_joints[26].x, pose_major_joints[26].y]),
                                    np.array([pose_major_joints[28].x, pose_major_joints[28].y]),
                                ]
                                left_leg = [
                                    np.array([pose_major_joints[23].x, pose_major_joints[23].y]),
                                    np.array([pose_major_joints[25].x, pose_major_joints[25].y]),
                                    np.array([pose_major_joints[27].x, pose_major_joints[27].y]),
                                ]

                                right_sa = calculate_angle(right_leg[0] - right_arm[0], right_arm[1] - right_arm[0])
                                right_ea = calculate_angle(right_arm[2] - right_arm[1], right_arm[0] - right_arm[1])
                                left_sa = calculate_angle(left_leg[0] - left_arm[0], left_arm[1] - left_arm[0])
                                left_ea = calculate_angle(left_arm[2] - left_arm[1], left_arm[0] - left_arm[1])

                                right_ha = calculate_angle(left_leg[0] - right_leg[0], right_leg[1] - right_leg[0]) - 90
                                right_ka = 180
                                left_ha = calculate_angle(right_leg[0] - left_leg[0], left_leg[1] - left_leg[0]) - 90
                                left_ka = 180

                                print("joint angles:", right_sa, right_ea, left_ea, left_sa)
                                print("joint angles:", right_ka, right_ha, left_ha, left_ka)

                                if (flag_green_sig is False) and (flag_red_sig is True):
                                    # VRNN_pose_input.append([right_ea, right_sa, left_sa, left_ea, 
                                    #                         left_ha, left_ka, right_ka, right_ha])
                                    VRNN_pose_input.append([right_ea, right_sa, left_sa, left_ea])

                                if (flag_do_demo_warn is True) and (flag_user_rep_done is False):
                                    pose_seq_user_rep.append([right_ea, right_sa, left_sa, left_ea])

                                    if len(pose_seq_user_rep) == 60:
                                        flag_user_rep_done = True
                                
                            else:
                                raise Exception("error in getting the limb coordinates")

                            # annotated_image = img.copy()
                            for face_landmarks in results_2.multi_face_landmarks:
                                # mpDraw.draw_landmarks(
                                #     image=img,
                                #     landmark_list=face_landmarks,
                                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                                #     landmark_drawing_spec=drawing_spec,
                                #     connection_drawing_spec=drawing_spec)

                                for _, data_point in enumerate(face_landmarks.landmark):
                                    # print("data inner:", data_point.x, data_point.y)
                                    h, w, c = img.shape
                                    cx, cy = int(data_point.x * w), int(data_point.y * h)

                                    if int(_) == 168:
                                        re_points.append([cx, cy])
                                        le_points.append([cx, cy])
                                        glabella = [cx, cy]
                                    if int(_) == 133:
                                        re_points.append([cx, cy])
                                    if int(_) == 362:
                                        le_points.append([cx, cy])

                                    if int(_) in right_eye_main_dict.keys():
                                        right_eye_main_dict[int(_)] = [cx, cy]
                                    if int(_) in left_eye_main_dict.keys():
                                        left_eye_main_dict[int(_)] = [cx, cy]

                                    if int(_) in [130, 247, 30, 29, 27, 28, 56, 190, 243, 25, 110, 24, 23, 22, 26, 112]:
                                        r_lobe_coords[int(_)] = [cx, cy]

                                    if int(_) in [463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256,
                                                341]:
                                        l_lobe_coords[int(_)] = [cx, cy]
                    

                    # Generate the sequence using this starting pose:
                    if (flag_green_sig is True) and (flag_red_sig is True) and (flag_generated_motion_once is False):
                        print("pointer flag_green_sig model input")
                        initial_pose_user = np.array(VRNN_pose_input)
                        initial_pose_user = np.mean(initial_pose_user, axis=0)
                        initial_pose_user = torch.from_numpy(initial_pose_user).to(dtype=torch.float32)

                        print("initial_pose_user:", initial_pose_user)

                        try:
                            print("just before the main generate_sequence model...")
                            VRNN_seq_gen = modelv1.generate_sequence(vrnn_model=model_seq_gen, initial_pose_original=initial_pose_user)
                            VRNN_seq_gen = VRNN_seq_gen * 180
                            print("just after the main generate_sequence model... with details of VRNN_seq_gen:", VRNN_seq_gen.shape)
                        except Exception as e_motion_gen_1:
                            raise Exception("e_motion_gen_1; Error in generating the sequence...", e_motion_gen_1)
                        
                        # try:
                        #     print("just before making the animation")
                        #     modelv1.create_animation(VRNN_seq_gen, output_path="animation_from.gif", num_reps=2, save_animation=True)
                        #     print("created animation...")
                        # except Exception as e_motion_gen_2:
                        #     raise Exception("e_motion_gen_2; Error in animating the dance sequence...", e_motion_gen_2)
                        
                        # flag_generated_motion_once = True
                        # print("flag_generated_motion_once = True")

                        # After showing the animation, we give the user some time to get ready before performing
                        # the dance motion...

                        # try:
                        #     print("inside the code to show the animation...")

                        #     gif_path = "animation_from.gif"  # Replace with your GIF file path
                        #     gif_path = "/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/animation_from.gif"

                        #     image = Image.open(gif_path)
                        #     frames = [ImageTk.PhotoImage(frame.copy()) for frame in ImageSequence.Iterator(image)]

                        #     frame_idx = 0
                        #     play_count = 0

                        #     label = tk.Label(root)
                        #     label.pack()

                        #     print("just before the playing the function...")
                        #     play_gif(frames, label, frame_idx, play_count)
                        #     root.mainloop()
                        #     print("just after the playing the function...")
                        # except Exception as e_show_animation:
                        #     print("error in showing the animation GIF:", e_show_animation)

                        # Show the animation 10 times ...
                        print("going to show the animation...")
                        try:
                            for gif_c_ in range(3):
                                window.update_gif("/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/animation_from.gif")

                                # time.sleep(3)
                        except Exception as e_GIF:
                            print("Error in showing the animation:", e_GIF)
                        print("done showing the animation...")

                        if flag_sig_movement_demo is False:
                            sig_movement_demo = time.time()
                            flag_sig_movement_demo = True
                        
                        print("pointer before getting user input for demo reaction...")

                        if ((time.time() - sig_movement_demo) < 5) and (flag_do_demo_warn is False):
                            print("inside the demo condition:", time.time() - sig_movement_demo)
                            cv.putText(img2, "Get ready to copy the movement, you will get a small amount of time to replicate the shown dance motion", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                            cv.putText(img2, "Message will disappear in " + str(5 - (int(time.time() - sig_movement_demo))), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                        else:
                            print("inside the demo condition else timings:", time.time(), sig_movement_demo)
                            flag_do_demo_warn = True
                        
                        if flag_do_demo_warn is True:
                            if flag_user_rep_done is False:
                                print("inside the user input for demo repetition condition", len(pose_seq_user_rep), len(VRNN_seq_gen))
                                cv.putText(img2, "Number of frames left to capture...", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                                cv.putText(img2, str(len(VRNN_seq_gen) - len(pose_seq_user_rep)), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                            else:
                                # The code which takes care of the comparison
                                pose_estimator = dance_comp_Dyuthi.PoseEstimator()

                                pose_seq_similarity_score = pose_estimator.compare_kps(pose_seq_user_rep, VRNN_seq_gen)
                                print("pose_seq_similarity_score:", pose_seq_similarity_score)
                                
                    if flag_do_demo_warn is True:
                        for key in right_eye_main_dict:
                            right_eye_main.append(right_eye_main_dict[key])
                        for key in left_eye_main_dict:
                            left_eye_main.append(left_eye_main_dict[key])

                        right_eye_main = np.array(right_eye_main, np.int32)
                        right_eye_main = right_eye_main.reshape((-1, 1, 2))

                        left_eye_main = np.array(left_eye_main, np.int32)
                        left_eye_main = left_eye_main.reshape((-1, 1, 2))

                        isClosed = True
                        color = (255, 0, 0)
                        thickness = 2

                        img = cv.polylines(img, [right_eye_main], isClosed, color, thickness)
                        img = cv.polylines(img, [left_eye_main], isClosed, color, thickness)

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

                        for key in regions_re.keys():
                            temp = regions_re[key]
                            temp_coords = []

                            for index in temp:
                                temp_coords.append(right_eye_main_dict[index])

                            regions_re[key] = temp_coords

                        for key in regions_le.keys():
                            temp = regions_le[key]
                            temp_coords = []

                            for index in temp:
                                temp_coords.append(left_eye_main_dict[index])

                            regions_le[key] = temp_coords

                        for key in right_upper_lobe.keys():
                            temp = right_upper_lobe[key]
                            temp_coords = []

                            for index in temp:
                                if index in r_lobe_coords.keys():
                                    temp_coords.append(r_lobe_coords[index])

                                else:
                                    temp_coords.append(right_eye_main_dict[index])

                            right_upper_lobe[key] = temp_coords

                        for key in right_lower_lobe.keys():
                            temp = right_lower_lobe[key]
                            temp_coords = []

                            for index in temp:
                                if index in r_lobe_coords.keys():
                                    temp_coords.append(r_lobe_coords[index])

                                else:
                                    temp_coords.append(right_eye_main_dict[index])

                            right_lower_lobe[key] = temp_coords

                        for key in left_upper_lobe.keys():
                            temp = left_upper_lobe[key]
                            temp_coords = []

                            for index in temp:
                                if index in l_lobe_coords.keys():
                                    temp_coords.append(l_lobe_coords[index])

                                else:
                                    temp_coords.append(left_eye_main_dict[index])

                            left_upper_lobe[key] = temp_coords

                        for key in left_lower_lobe.keys():
                            temp = left_lower_lobe[key]
                            temp_coords = []

                            for index in temp:
                                if index in l_lobe_coords.keys():
                                    temp_coords.append(l_lobe_coords[index])

                                else:
                                    temp_coords.append(left_eye_main_dict[index])

                            left_lower_lobe[key] = temp_coords

                        print("regions_re:", regions_re)
                        print("regions_le:", regions_le)
                        print("right_upper_lobe:", right_upper_lobe)
                        print("right_lower_lobe:", right_lower_lobe)
                        print("left_upper_lobe:", left_upper_lobe)
                        print("left_lower_lobe:", left_lower_lobe)

                        rect_right = cv.boundingRect(right_eye_main)
                        x, y, w, h = rect_right
                        cropped_re = img[y:y + h, x:x + w].copy()
                        right_eye_main = right_eye_main - right_eye_main.min(axis=0)
                        mask_re = np.zeros(cropped_re.shape[:2], np.uint8)
                        cv.drawContours(mask_re, [right_eye_main], -1, (255, 255, 255), -1, cv.LINE_AA)
                        dst_re = cv.bitwise_and(cropped_re, cropped_re, mask=mask_re)
                        bg = np.ones_like(cropped_re, np.uint8) * 255
                        cv.bitwise_not(bg, bg, mask=mask_re)
                        dst_re = bg + dst_re

                        for key in regions_re.keys():
                            temp = regions_re[key]
                            temp_t = []

                            for i in range(len(temp)):
                                temp_t.append([temp[i][0] - x, temp[i][1] - y])

                            regions_re[key] = temp_t

                        for key in right_upper_lobe.keys():
                            temp = right_upper_lobe[key]
                            temp_t = []

                            for i in range(len(temp)):
                                temp_t.append([temp[i][0] - x, temp[i][1] - y])

                            right_upper_lobe[key] = temp_t

                        for key in right_lower_lobe.keys():
                            temp = right_lower_lobe[key]
                            temp_t = []

                            for i in range(len(temp)):
                                temp_t.append([temp[i][0] - x, temp[i][1] - y])

                            right_lower_lobe[key] = temp_t

                        rect_left = cv.boundingRect(left_eye_main)
                        x, y, w, h = rect_left
                        cropped_le = img[y:y + h, x:x + w].copy()
                        left_eye_main = left_eye_main - left_eye_main.min(axis=0)
                        mask_le = np.zeros(cropped_le.shape[:2], np.uint8)
                        cv.drawContours(mask_le, [left_eye_main], -1, (255, 255, 255), -1, cv.LINE_AA)
                        dst_le = cv.bitwise_and(cropped_le, cropped_le, mask=mask_le)
                        bg = np.ones_like(cropped_le, np.uint8) * 255
                        cv.bitwise_not(bg, bg, mask=mask_le)
                        dst_le = bg + dst_le

                        for key in regions_le.keys():
                            temp = regions_le[key]
                            temp_t = []

                            for i in range(len(temp)):
                                temp_t.append([temp[i][0] - x, temp[i][1] - y])

                            regions_le[key] = temp_t

                        for key in left_upper_lobe.keys():
                            temp = left_upper_lobe[key]
                            temp_t = []

                            for i in range(len(temp)):
                                temp_t.append([temp[i][0] - x, temp[i][1] - y])

                            left_upper_lobe[key] = temp_t

                        for key in left_lower_lobe.keys():
                            temp = left_lower_lobe[key]
                            temp_t = []

                            for i in range(len(temp)):
                                temp_t.append([temp[i][0] - x, temp[i][1] - y])

                            left_lower_lobe[key] = temp_t

                        kernel = np.array([[0, -1, 0], [-1, 6.5, -1], [0, -1, 0]])

                        thresh = 0

                        dst_re = cv.cvtColor(dst_re, cv.COLOR_BGR2GRAY)
                        # dst_re = cv.threshold(dst_re, thresh, 255, cv.THRESH_BINARY)[1]
                        dst_re = cv.filter2D(dst_re, -1, kernel)
                        dst_re = cv.medianBlur(dst_re, 5)
                        # dst_re = cv.threshold(dst_re, thresh, 255, cv.THRESH_BINARY)[1]
                        # dst_re = cv.Canny(image=dst_re, threshold1=50, threshold2=200)

                        dst_le = cv.cvtColor(dst_le, cv.COLOR_BGR2GRAY)
                        dst_le = cv.filter2D(dst_le, -1, kernel)
                        dst_le = cv.medianBlur(dst_le, 5)
                        # dst_le = cv.threshold(dst_le, thresh, 255, cv.THRESH_BINARY)[1]
                        # dst_le = cv.Sobel(src=dst_le, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
                        # dst_le = cv.Canny(image=dst_le, threshold1=50, threshold2=200)

                        ar_re = []
                        ar_le = []

                        for key in regions_re.keys():
                            # mask_re = np.zeros_like(dst_re)
                            #
                            list_coords = np.array(regions_re[key])
                            # # print("list_coords_re:", list_coords)
                            # cv.fillPoly(mask_re, [list_coords], 255)
                            # mask_re = cv.threshold(mask_re, 200, 255, cv.THRESH_BINARY)[1]
                            # masked_image = cv.bitwise_and(dst_re, mask_re)
                            masked_image = dst_re

                            # black_pixel_count_re = 0
                            # white_pixel_count_re = 0

                            pixel_count = 0
                            values = 0

                            for y in range(masked_image.shape[0]):
                                for x in range(masked_image.shape[1]):
                                    if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
                                        pixel_count += 1
                                        values += masked_image[y, x]

                                        # if masked_image[y, x] < 20:
                                        #     black_pixel_count_re += 1
                                        #
                                        # else:
                                        #     white_pixel_count_re += 1

                            # percents_re.append(round(black_pixel_count_re / (black_pixel_count_re + white_pixel_count_re), 2))
                            percents_re.append(round((values / pixel_count), 2))
                            text_temp = ":" + str(pixel_count) + ":"
                            ar_re.append(text_temp)

                        for key in regions_le.keys():
                            # mask_le = np.zeros_like(dst_le)
                            #
                            list_coords = np.array(regions_le[key])
                            # # print("list_coords_le:", list_coords)
                            # cv.fillPoly(mask_le, [list_coords], 255)
                            # # cv.drawContours(mask_le, [list_coords], -1, 255, 1, thickness=cv.LINE_AA)
                            # # mask_le = cv.threshold(mask_le, 200, 255, cv.THRESH_BINARY)[1]
                            # masked_image = cv.bitwise_and(dst_le, mask_le)
                            masked_image = dst_le

                            # black_pixel_count_le = 0
                            # white_pixel_count_le = 0

                            pixel_count = 0
                            values = 0

                            for y in range(masked_image.shape[0]):
                                for x in range(masked_image.shape[1]):
                                    if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
                                        pixel_count += 1
                                        values += masked_image[y, x]

                                        # if masked_image[y, x] < 20:
                                        #     black_pixel_count_le += 1
                                        #
                                        # else:
                                        #     white_pixel_count_le += 1

                            # percents_le.append(round(black_pixel_count_le / (black_pixel_count_le + white_pixel_count_le), 2))
                            percents_le.append(round((values / pixel_count), 2))
                            text_temp = ":" + str(pixel_count) + ":"
                            ar_le.append(text_temp)

                        index = 0
                        for key in right_upper_lobe.keys():
                            list_coords = np.array(right_upper_lobe[key])
                            masked_image = dst_re
                            pixel_count = 0

                            for y in range(masked_image.shape[0]):
                                for x in range(masked_image.shape[1]):
                                    if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
                                        pixel_count += 1

                            ar_re[index] = str(pixel_count) + ar_re[index]
                            index += 1

                        index = 0
                        for key in right_lower_lobe.keys():
                            list_coords = np.array(right_lower_lobe[key])
                            masked_image = dst_re
                            pixel_count = 0

                            for y in range(masked_image.shape[0]):
                                for x in range(masked_image.shape[1]):
                                    if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
                                        pixel_count += 1

                            ar_re[index] = ar_re[index] + str(pixel_count)
                            index += 1

                        index = 0
                        for key in left_upper_lobe.keys():
                            list_coords = np.array(left_upper_lobe[key])
                            masked_image = dst_re
                            pixel_count = 0

                            for y in range(masked_image.shape[0]):
                                for x in range(masked_image.shape[1]):
                                    if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
                                        pixel_count += 1

                            ar_le[index] = str(pixel_count) + ar_le[index]
                            index += 1

                        index = 0
                        for key in left_lower_lobe.keys():
                            list_coords = np.array(left_lower_lobe[key])
                            masked_image = dst_re
                            pixel_count = 0

                            for y in range(masked_image.shape[0]):
                                for x in range(masked_image.shape[1]):
                                    if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
                                        pixel_count += 1

                            ar_le[index] = ar_le[index] + str(pixel_count)
                            index += 1

                        print("percents_re:", percents_re)
                        print("percents_le:", percents_le)

                        # contours, _ = cv.findContours(dst_le, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
                        # contour_image = np.zeros_like(dst_le)
                        # cv.drawContours(contour_image, contours, -1, (255, 255, 255), 2)  # Draw all contours with thickness 2
                        # # print("_:", _)

                        print("cropped shapes:", dst_re.shape, dst_le.shape)

                        resized_re = cv.resize(dst_re, (128, 32), interpolation=cv.INTER_LINEAR)
                        resized_le = cv.resize(dst_le, (128, 32), interpolation=cv.INTER_LINEAR)

                        # TODO: Uncomment this for inference
                        if num_frames != 0:
                            plain_raw_video[0].append(resized_re)
                            plain_raw_video[1].append(resized_le)

                            num_frames -= 1

                        else:
                            # # eye_crop_window = 32
                            # eye_right_cropped_model_input = np.array(eye_right_cropped_model_input)
                            # eye_left_cropped_model_input = np.array(eye_left_cropped_model_input)
                            #
                            # # print("eye_right_cropped_model_input:", eye_right_cropped_model_input.shape)
                            # # print("eye_left_cropped_model_input:", eye_left_cropped_model_input.shape)
                            #
                            # # TODO: Enough for the GPU processing
                            # # last_layer_from_upper_branch, softmaxed_layer = EyeVidPre_and_ViViT.PreProcessandFreezedOutput(
                            # #     eye_right_cropped_model_input, eye_left_cropped_model_input)
                            # # print("last_layer_from_upper_branch:", last_layer_from_upper_branch)

                            for frame_counter in range(len(plain_raw_video[0])):
                                if (frame_counter == (len(plain_raw_video[0]) - 1)) and (music_queue.empty() is False):
                                    try:
                                        music_queue.get()
                                        print("music_queue status:", music_queue.empty())
                                    except Exception as e:
                                        print("mp 2, Error in music queue handling:", e)
                                        raise Exception("mp 2, Error in music queue handling:")

                                resized_re = plain_raw_video[0][frame_counter]
                                resized_le = plain_raw_video[1][frame_counter]

                                print("type of resized re:", type(resized_re))

                                otsu_threshold, resized_re_OTSU = cv.threshold(
                                    resized_re, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
                                )
                                otsu_threshold_2, resized_le_OTSU = cv.threshold(
                                    resized_le, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
                                )

                                resized_re_t = np.expand_dims(resized_re, axis=-1)
                                resized_le_t = np.expand_dims(resized_le, axis=-1)

                                resized_re_OTSU = np.expand_dims(resized_re_OTSU, axis=-1)
                                resized_le_OTSU = np.expand_dims(resized_le_OTSU, axis=-1)

                                resized_re_t = np.concatenate((resized_re_t, resized_re_OTSU), axis=-1)
                                resized_le_t = np.concatenate((resized_le_t, resized_le_OTSU), axis=-1)

                                # Repetition needs to be changed
                                resized_re_KMeans = KMeans_Image(resized_re)
                                resized_le_KMeans = KMeans_Image(resized_le)

                                resized_re_KMeans = np.expand_dims(resized_re_KMeans, axis=-1)
                                resized_le_KMeans = np.expand_dims(resized_le_KMeans, axis=-1)

                                resized_re_t = np.concatenate((resized_re_t, resized_re_KMeans), axis=-1)
                                resized_le_t = np.concatenate((resized_le_t, resized_le_KMeans), axis=-1)

                                print("cropped shapes:", resized_re.shape, resized_le.shape)
                                print("cropped shapes in _t:", resized_re_t.shape, resized_le_t.shape)

                                reward = 0
                                reward_directional = 0

                                # Subtraction for OTSU binarization
                                if resized_re_OTSU_before is not None:
                                    resized_re_OTSU_sub = cv.subtract(resized_re_OTSU, resized_re_OTSU_before)
                                    resized_re_OTSU_sub = np.where(resized_re_OTSU_sub > 0, 255, 0).astype(np.uint8)

                                    wp_c = np.sum(resized_re_OTSU_sub == 255)
                                    r_w = wp_c / (32 * 128)
                                    r_b = 1 - r_w
                                    reward += r_b

                                    num_white_curr = np.where(resized_re_OTSU > 0, 255, 0).astype(np.uint8)
                                    num_white_curr = np.sum(num_white_curr == 255) / (32 * 128)

                                    num_white_before = np.where(resized_re_OTSU_before > 0, 255, 0).astype(np.uint8)
                                    num_white_before = np.sum(num_white_before == 255) / (32 * 128)

                                    direction = -1 if (num_white_curr - num_white_before) < 0 else 1

                                    reward_directional += direction * reward

                                    cv.imshow("right_eye OTSU subtracted", resized_re_OTSU_sub)
                                    resized_re_OTSU_before = resized_re_OTSU
                                else:
                                    resized_re_OTSU_before = resized_re_OTSU

                                if resized_le_OTSU_before is not None:
                                    resized_le_OTSU_sub = cv.subtract(resized_le_OTSU, resized_le_OTSU_before)
                                    resized_le_OTSU_sub = np.where(resized_le_OTSU_sub > 0, 255, 0).astype(np.uint8)

                                    wp_c = np.sum(resized_le_OTSU_sub == 255)
                                    r_w = wp_c / (32 * 128)
                                    r_b = 1 - r_w
                                    reward += r_b

                                    num_white_curr = np.where(resized_le_OTSU > 0, 255, 0).astype(np.uint8)
                                    num_white_curr = np.sum(num_white_curr == 255) / (32 * 128)

                                    num_white_before = np.where(resized_le_OTSU_before > 0, 255, 0).astype(np.uint8)
                                    num_white_before = np.sum(num_white_before == 255) / (32 * 128)

                                    direction = -1 if (num_white_curr - num_white_before) < 0 else 1

                                    reward_directional += direction * reward

                                    cv.imshow("left_eye OTSU subtracted", resized_le_OTSU_sub)
                                    print("something very inner:", "left_eye OTSU subtracted")

                                    resized_le_OTSU_before = resized_le_OTSU
                                else:
                                    resized_le_OTSU_before = resized_le_OTSU

                                reward /= 2
                                reward_directional /= 2

                                print("reward from the eyes:", reward)

                                reward_avg += reward
                                avgs_unit.append(reward_directional)

                                eye_right_cropped_model_input.append(resized_re_t)
                                eye_left_cropped_model_input.append(resized_le_t)

                            eye_right_cropped_model_input = np.array(eye_right_cropped_model_input)
                            eye_left_cropped_model_input = np.array(eye_left_cropped_model_input)

                            num_frames = 32
                            reward_avg = reward_avg / num_frames  # equal weights to all the frames...

                            '''Concatenate images now'''
                            start_2 = time.time()
                            start_3 = time.time()
                            concat_frame_left = concat_images(eye_right_cropped_model_input)
                            end_3 = time.time()

                            concat_frame_left_save = cv.normalize(concat_frame_left, None, 0, 255, cv.NORM_MINMAX)
                            concat_frame_left_save = concat_frame_left_save.astype(np.uint8)

                            start_4 = time.time()
                            concat_frame_right = concat_images(eye_left_cropped_model_input)
                            end_4 = time.time()
                            end_2 = time.time()

                            concat_frame_right_save = cv.normalize(concat_frame_right, None, 0, 255, cv.NORM_MINMAX)
                            concat_frame_right_save = concat_frame_right_save.astype(np.uint8)

                            cv.imwrite(f"ConcatEyeImages/right_eye_{window_num}.png", concat_frame_right_save)
                            cv.imwrite(f"ConcatEyeImages/left_eye_{window_num}.png", concat_frame_left_save)

                            print("time for concat function left:", end_3 - start_3)
                            print("time for concat function right:", end_4 - start_4)
                            print("time for both concat functions:", end_2 - start_2)

                            frame_arr = [concat_frame_right, concat_frame_left, reward_avg]
                            print("frame_arr dimensions:", len(frame_arr))
                            print("concat_frame_right dimensions:", len(concat_frame_right))
                            print("concat_frame_right:", concat_frame_right)
                            print("reward_avg:", reward_avg)
                            print(f"\nreward_avg--- {reward_avg}", file=open(log_file, 'a'))

                            if prev_state is None:
                                print("prev state is None")
                                prev_state = [concat_frame_right, concat_frame_left]
                                print("pointer 1 after reward_avg")
                            else:
                                # The action state calculation using the concatenated images as input
                                start_after_concat = time.time()
                                print("pointer 2 after reward_avg")
                                # output_after_dense, final_output = EyeVidPre_and_ViViT.PreProcessandFreezedOutput_MobileNetV1(
                                #     concat_frame_right, concat_frame_left)

                                curr_state = [concat_frame_right, concat_frame_left]
                                print("pointer 3 after reward_avg:", curr_state[0].shape, prev_state[0].shape)

                                # Process the right eye input for prev
                                inputs = image_processor_1(prev_state[0], return_tensors="pt")
                                print("pointer 4 after reward_avg:")
                                with torch.no_grad():
                                    output_right_eye_prev = model_1(**inputs).last_hidden_state
                                    print("pointer 5 after reward_avg", output_right_eye_prev.shape)
                                # Process the left eye input for prev
                                inputs = image_processor_1(prev_state[1], return_tensors="pt")
                                print("pointer 6 after reward_avg")
                                with torch.no_grad():
                                    output_left_eye_prev = model_1(**inputs).last_hidden_state
                                    print("pointer 7 after reward_avg")

                                output_right_eye_prev = output_right_eye_prev.detach().numpy()
                                print("pointer 8 after reward_avg")
                                output_right_eye_prev = tf.convert_to_tensor(output_right_eye_prev, dtype=tf.float32)
                                print("pointer 9 after reward_avg")
                                output_left_eye_prev = output_left_eye_prev.detach().numpy()
                                print("pointer 10 after reward_avg")
                                output_left_eye_prev = tf.convert_to_tensor(output_left_eye_prev, dtype=tf.float32)
                                print("pointer 11 after reward_avg")

                                # Process the right eye input for curr
                                inputs = image_processor_1(curr_state[0], return_tensors="pt")
                                print("pointer 12 after reward_avg")
                                with torch.no_grad():
                                    output_right_eye_curr = model_1(**inputs).last_hidden_state
                                    print("pointer 13 after reward_avg")
                                # Process the left eye input for curr
                                inputs = image_processor_1(curr_state[1], return_tensors="pt")
                                print("pointer 14 after reward_avg")
                                with torch.no_grad():
                                    output_left_eye_curr = model_1(**inputs).last_hidden_state
                                    print("pointer 15 after reward_avg")

                                output_right_eye_curr = output_right_eye_curr.detach().numpy()
                                print("pointer 16 after reward_avg")
                                output_right_eye_curr = tf.convert_to_tensor(output_right_eye_curr, dtype=tf.float32)
                                print("pointer 17 after reward_avg")
                                output_left_eye_curr = output_left_eye_curr.detach().numpy()
                                print("pointer 18 after reward_avg")
                                output_left_eye_curr = tf.convert_to_tensor(output_left_eye_curr, dtype=tf.float32)
                                print("pointer 19 after reward_avg")

                                # prev_state = [np.expand_dims(output_right_eye_prev, axis=0),
                                #               np.expand_dims(output_left_eye_prev, axis=0)]
                                prev_state = [output_right_eye_prev, output_left_eye_prev]
                                print("pointer 20 after reward_avg")

                                final_output = policy(prev_state, ou_noise)
                                print("pointer 21 after reward_avg")
                                end_after_concat = time.time()
                                print("pointer 22 after reward_avg")

                                print("final_output:", final_output)
                                print("time taken for layer after:", end_after_concat - start_after_concat)

                                action = int(np.argmax(final_output))  # replace with policy function
                                print("pointer 23 after reward_avg")
                                # action = int(torch.argmax(final_output))  # replace with policy function
                                alpha = 1
                                beta = 1
                                gamma = 10
                                
                                # print("reward just before buffer learn setting:", reward)

                                # reward = alpha*reward_avg + beta*calculate_temporal_smoothness_reward(avgs_unit)  # New reward function...
                                if pose_seq_similarity_score is None:
                                    print("some error in pose_seq_similarity_score assignment position...")
                                else:
                                    reward = alpha*reward_avg + beta*calculate_temporal_smoothness_reward(avgs_unit) + gamma*pose_seq_similarity_score  # New reward function..., with dance comparison

                                print(f"\ntotal reward--- {reward}", file=open(log_file, 'a'))

                                reward_avg = 0
                                avgs_unit = []
                                print("pointer 24 after reward_avg")

                                print("After completion of action calculation:", action)
                                action_arr = [0] * num_actions
                                print("pointer 25 after reward_avg")
                                action_arr[action] = 1
                                print("pointer 26 after reward_avg")

                                print("shape of action vector before buffer.record:", action_arr)
                                action_arr = np.array(action_arr)
                                print("pointer 27 after reward_avg")
                                action_arr = action_arr.reshape(1, 6)
                                print("pointer 28 after reward_avg")
                                print("shape of action vector before buffer.record after expanding:", action_arr, action_arr.shape)

                                print("After completion of reward calculation:", reward)

                                # curr_state = [np.expand_dims(output_right_eye_curr, axis=0),
                                #               np.expand_dims(output_left_eye_curr, axis=0)]
                                curr_state_ = [output_right_eye_curr, output_left_eye_curr]
                                print("pointer 29 after reward_avg")

                                print("After completion of curr_state calculation:", type(curr_state_))

                                print("shape of output_right_eye_prev:", output_right_eye_prev.shape)

                                # buffer.record((np.array(prev_state), np.array([action]), np.array([reward]), np.array(curr_state)))
                                # buffer.record((np.array(output_right_eye_prev), np.array(output_left_eye_prev),
                                #                action, reward,
                                #                np.array(output_right_eye_curr), np.array(output_left_eye_curr)))
                                buffer.record((np.array(output_right_eye_prev), np.array(output_left_eye_prev),
                                            action_arr, reward,
                                            np.array(output_right_eye_curr), np.array(output_left_eye_curr)))
                                print("pointer 30 after reward_avg")
                                
                                # plt.imsave("output_right_eye_prev.png", output_right_eye_prev)
                                # plt.imsave("output_left_eye_prev.png", output_left_eye_prev)
                                # plt.imsave("output_right_eye_curr.png", output_right_eye_curr)
                                # plt.imsave("output_left_eye_curr.png", output_left_eye_curr)
                                # print(f"\nreward: {reward}", file=open(log_file, 'a'))

                                buffer.learn()
                                print("pointer 31 after reward_avg")

                                print("pointer actor model 3")
                                update_target(target_actor, actor_model, tau)
                                print("pointer 32 after reward_avg")

                                print("pointer critic model 3")
                                update_target(target_critic, critic_model, tau)
                                print("pointer 33 after reward_avg")

                                prev_state = curr_state
                                print("pointer 34 after reward_avg")

                                print("music_index:", action)

                                # frame_queue.put(frame_arr, block=True)
                                # # num_frames = 32

                                # Muisc playing part
                                # current_music_index = (window_num % 5)
                                current_music_index = action
                                print("pointer 35 after reward_avg:", current_music_index)
                                current_music_file = music_files[current_music_index]
                                print("pointer 36 after reward_avg:", current_music_file)
                                print("current_music_file:", current_music_file)

                                try:
                                    music_queue.put(music_files[current_music_index])
                                    print("pointer 37 after reward_avg")
                                except Exception as e:
                                    print("mp 3, Exception in music queue handling:", e)
                                    raise Exception("mp 3, Exception in music queue handling:")

                                reward = 0









                                # # The action state calculation using the sequence as input: RNN (GRU)
                                # actor_model_GRU(plain_raw_video)

                            frame_arr = []
                            plain_raw_video = [[], []]

                            eye_right_cropped_model_input = []
                            eye_left_cropped_model_input = []

                            print("video window num:", window_num)
                            window_num += 1
                            # window_num = 1

                            # Re-init of the marker variables
                            green_sig_init_once = False
                            flag_generated_motion_once = False
                            flag_movement_demo = False
                            flag_sig_movement_demo = False
                            sig_movement_demo = None
                            flag_do_demo_warn = False
                            flag_take_usr_input_rep_demo = False
                            flag_user_rep_done = False

                            pose_seq_similarity_score = None

                            VRNN_pose_input = []
                            pose_seq_user_rep = []

                cv.imshow("img", img)
                cv.imshow("img2", img2)
                print("shape of image:", img.shape)

                # if (frame_queue.full()) or (cv.waitKey(1) & 0xFF == ord('q')):
                if False or (cv.waitKey(1) & 0xFF == ord('q')):
                    break

        end_time_total = time.time()

        print("total time taken:", end_time_total - start_time_total)

        # for ep in range(total_episodes):
        #     _, __, ___ = EyeCropping.camera_input_2()
        #     print("window num:", window_num)
        #     window_num += 1

            # ret, frame = cap.read()
            # if not ret:
            #     break
            #
            # if num_frames != 0:
            #     frame_arr.append(frame)
            #     num_frames -= 1
            # else:
            #     # Add the frame to the queue for processing, blocking if queue is full
            #     frame_queue.put(frame_arr, block=True)
            #     num_frames = 32
            #     frame_arr = []
            #
            #     # print("window number:", window_num)
            #     #
            #     # window_num += 1
            #
            # # Display the original frame
            # cv.imshow('Original Frame', frame)
            #
            # # Give some time for the threads to process
            # # time.sleep(0.05)  # Adjust this to balance between frame capture and processing
            #
            # if (frame_queue.full()) or (cv.waitKey(1) & 0xFF == ord('q')):
            #     break

        # vid_end_time = time.time()
        # print("total time:", vid_end_time - vid_start_time)

    except Exception as e:
        print("Error in video input threading...:", e)

    finally:
        # Clean up
        cap.release()
        cv.destroyAllWindows()
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
    music_thread = threading.Thread(target=play_music)
    music_thread.daemon = True  # Ensure the thread exits when the main program exits
    music_thread.start()

    video_capture_and_stuff()

    # Stop music after video ends
    pygame.mixer.music.stop()


if __name__ == "__main__":
    main()