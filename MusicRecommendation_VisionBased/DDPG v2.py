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

from transformers import VivitImageProcessor, VivitForVideoClassification, \
    AutoImageProcessor, MobileNetV1Model

import pygame

import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh

pygame.mixer.init()

# Global variable to track the current music
# current_music_file = None
# music_change_event = threading.Event()

# cap = cv.VideoCapture(0)

from CriticNetwork import EyeVidPre_and_ViViT_tf, EyeVidPre_and_ViViT
# from CriticNetwork.EyeVidPre_and_ViViT import KMeans_Image
from CriticNetwork import EyeCropping

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


lower_bound = 0
upper_bound = 5

# Create a Queue for handling music change requests
music_queue = queue.Queue()

image_processor_1 = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
model_1 = MobileNetV1Model.from_pretrained("google/mobilenet_v1_1.0_224")

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

    right_eye = layers.Input(shape=(1024, 7, 7))
    left_eye = layers.Input(shape=(1024, 7, 7))

    dense_layer_1 = layers.Dense(128, input_shape=(2*50176,))
    dense_layer_1_conc = layers.Dense(128)
    dense_layer_2 = layers.Dense(64)
    dense_layer_3 = layers.Dense(6)
    softmax = layers.Softmax()
    sigmoid = layers.Activation('sigmoid')

    last_state_concat = layers.Concatenate()([right_eye, left_eye])
    print("shape of last_state_concat in get_actor_extra:", last_state_concat.shape)

    # concat_layer_flatten = tf.reshape(last_state_concat, (tf.shape(last_state_concat)[0], -1))
    concat_layer_flatten = layers.Reshape((-1,))(last_state_concat)
    print("shape of concat_layer_flatten in get_actor_extra:", concat_layer_flatten.shape)

    output_after_dense = dense_layer_1(concat_layer_flatten)

    # output_after_dense = tf.reshape(output_after_dense, (-1,))
    output_after_dense = layers.Reshape((-1,))(output_after_dense)

    # output_after_dense = tf.expand_dims(output_after_dense, axis=0)
    output_after_dense = layers.Reshape((1, -1))(output_after_dense)

    output_after_dense = dense_layer_1_conc(output_after_dense)

    output_after_dense = dense_layer_2(output_after_dense)
    output_after_dense = dense_layer_3(output_after_dense)

    final_output = softmax(output_after_dense)

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

    right_eye = layers.Input(shape=(1024, 7, 7))
    left_eye = layers.Input(shape=(1024, 7, 7))
    action = layers.Input(shape=(1, 6))

    print("critic model, init done...")

    dense_layer_1 = keras.layers.Dense(128, input_shape=(2*50176,))
    # dense_layer_1 = keras.layers.Dense(128, input_shape=(100352,))
    # dense_layer_1 = keras.layers.Dense(64, input_shape=(50176,))

    dense_layer_1_conc = keras.layers.Dense(128)
    dense_layer_2 = keras.layers.Dense(64)
    dense_layer_3 = keras.layers.Dense(32)
    dense_layer_parallel_1 = keras.layers.Dense(16)
    dense_layer_parallel_2 = keras.layers.Dense(32)
    dense_layer_comb_1 = keras.layers.Dense(32)
    dense_layer_comb_2 = keras.layers.Dense(8)
    dense_layer_final = keras.layers.Dense(1)

    print("critic model, layer definitions done...")

    # Concatenation of the last hidden states
    last_state_concat = layers.Concatenate()([right_eye, left_eye])
    print("shape of last_state_concat:", last_state_concat.shape)

    # Flatten the concatenated states
    # concat_layer_flatten = tf.reshape(last_state_concat, (tf.shape(last_state_concat)[0], -1))
    concat_layer_flatten = layers.Reshape((-1,))(last_state_concat)
    print("shape of concat_layer_flatten:", concat_layer_flatten.shape)

    # Pass through the first dense layer
    output_after_dense = dense_layer_1(concat_layer_flatten)
    # output_after_dense = tf.reshape(output_after_dense, (-1,))
    output_after_dense = layers.Reshape((-1,))(output_after_dense)
    print("shape of output_after_dense:", output_after_dense.shape)

    output_after_dense = dense_layer_1_conc(output_after_dense)

    # Pass through the additional dense layers
    output_after_dense = dense_layer_2(output_after_dense)
    output_after_dense = dense_layer_3(output_after_dense)

    # output_after_dense = tf.reshape(output_after_dense, (1, -1))
    output_after_dense = layers.Reshape((1, -1))(output_after_dense)
    print("shape of output_after_dense 2:", output_after_dense.shape)

    # Process action input through parallel dense layers
    action_inner = dense_layer_parallel_1(action)
    action_inner = dense_layer_parallel_2(action_inner)
    print("shape of action_inner:", action_inner.shape)

    # Combine action and output_after_dense
    # comb = tf.concat([action, output_after_dense], axis=0)
    comb = layers.Concatenate(axis=0)([action_inner, output_after_dense])
    print("shape of comb:", comb.shape)

    # comb = tf.reshape(comb, (tf.shape(comb)[0], -1))
    comb = layers.Reshape((-1,))(comb)
    print("shape of comb 2:", comb.shape)

    # Apply the combined dense layers
    comb = dense_layer_comb_1(comb)
    comb = dense_layer_comb_2(comb)
    print("shape of comb 3:", comb.shape)

    # Final output
    final_output = dense_layer_final(comb)
    print("shape of final_output:", final_output.shape)

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
num_actions = 1

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(6), std_deviation=float(std_dev) * np.ones(6))

actor_model = get_actor_extra()
critic_model = get_critic_extra()

target_actor = get_actor_extra()
target_critic = get_critic_extra()

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
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
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

        print("obs_tuple[4:", obs_tuple[5].shape)
        self.next_left_eye_buffer[index] = obs_tuple[5]
        print("self.next_left_eye_buffer[index] in Buffer done:")

        self.buffer_counter += 1

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
            print("shape of target_critic_output:", target_critic_output.shape, reward_batch.shape)
            print("shapes of objects before critic_model in Buffer update:", right_eye_batch.shape,
                  left_eye_batch.shape, action_batch.shape)
            
            y = reward_batch + gamma * target_critic_output
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

def play_music():
    current_music = None

    while True:
        try:
            # Check if there is a new music file in the queue
            new_music = music_queue.get_nowait()  # Non-blocking check
            if new_music != current_music:
                pygame.mixer.music.stop()  # Stop current music
                pygame.mixer.music.load(new_music)
                pygame.mixer.music.play(-1)  # Play indefinitely
                print(f"Playing new music: {new_music}")
                current_music = new_music
        except queue.Empty:
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

    def angle_of_inclination(p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2

        delta_x = x2 - x1
        delta_y = y2 - y1
        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

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
                # cv.imshow("simple plain video input", img)

                img = adjust_brightness(img2)

                right_eye_main = []
                left_eye_main = []

                with mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        min_detection_confidence=0.5) as face_mesh:
                    if True:
                        # Convert the BGR image to RGB before processing.
                        results_2 = face_mesh.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

                        # Print and draw face mesh landmarks on the image.
                        if not results_2.multi_face_landmarks:
                            continue

                        # print(len(results_2.multi_face_landmarks))

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
                        resized_re = plain_raw_video[0][frame_counter]
                        resized_le = plain_raw_video[1][frame_counter]

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

                        # Subtraction for OTSU binarization
                        if resized_re_OTSU_before is not None:
                            resized_re_OTSU_sub = cv.subtract(resized_re_OTSU, resized_re_OTSU_before)
                            resized_re_OTSU_sub = np.where(resized_re_OTSU_sub > 0, 255, 0).astype(np.uint8)

                            wp_c = np.sum(resized_re_OTSU_sub == 255)
                            r_w = wp_c / (32 * 128)
                            r_b = 1 - r_w
                            reward += r_b

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

                            cv.imshow("left_eye OTSU subtracted", resized_le_OTSU_sub)
                            print("something very inner:", "left_eye OTSU subtracted")

                            resized_le_OTSU_before = resized_le_OTSU
                        else:
                            resized_le_OTSU_before = resized_le_OTSU

                        reward /= 2
                        print("reward from the eyes:", reward)

                        reward_avg += reward

                        eye_right_cropped_model_input.append(resized_re_t)
                        eye_left_cropped_model_input.append(resized_le_t)

                    eye_right_cropped_model_input = np.array(eye_right_cropped_model_input)
                    eye_left_cropped_model_input = np.array(eye_left_cropped_model_input)

                    num_frames = 32
                    reward_avg = reward_avg / num_frames

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

                    if prev_state is None:
                        print("prev state is None")
                        prev_state = [concat_frame_right, concat_frame_left]
                    else:
                        start_after_concat = time.time()
                        # output_after_dense, final_output = EyeVidPre_and_ViViT.PreProcessandFreezedOutput_MobileNetV1(
                        #     concat_frame_right, concat_frame_left)

                        curr_state = [concat_frame_right, concat_frame_left]

                        # Process the right eye input for prev
                        inputs = image_processor_1(prev_state[0], return_tensors="pt")
                        with torch.no_grad():
                            output_right_eye_prev = model_1(**inputs).last_hidden_state
                        # Process the left eye input for prev
                        inputs = image_processor_1(prev_state[1], return_tensors="pt")
                        with torch.no_grad():
                            output_left_eye_prev = model_1(**inputs).last_hidden_state

                        output_right_eye_prev = output_right_eye_prev.detach().numpy()
                        output_right_eye_prev = tf.convert_to_tensor(output_right_eye_prev, dtype=tf.float32)
                        output_left_eye_prev = output_left_eye_prev.detach().numpy()
                        output_left_eye_prev = tf.convert_to_tensor(output_left_eye_prev, dtype=tf.float32)

                        # Process the right eye input for curr
                        inputs = image_processor_1(curr_state[0], return_tensors="pt")
                        with torch.no_grad():
                            output_right_eye_curr = model_1(**inputs).last_hidden_state
                        # Process the left eye input for curr
                        inputs = image_processor_1(curr_state[1], return_tensors="pt")
                        with torch.no_grad():
                            output_left_eye_curr = model_1(**inputs).last_hidden_state

                        output_right_eye_curr = output_right_eye_curr.detach().numpy()
                        output_right_eye_curr = tf.convert_to_tensor(output_right_eye_curr, dtype=tf.float32)
                        output_left_eye_curr = output_left_eye_curr.detach().numpy()
                        output_left_eye_curr = tf.convert_to_tensor(output_left_eye_curr, dtype=tf.float32)

                        # prev_state = [np.expand_dims(output_right_eye_prev, axis=0),
                        #               np.expand_dims(output_left_eye_prev, axis=0)]
                        prev_state = [output_right_eye_prev, output_left_eye_prev]

                        final_output = policy(prev_state, ou_noise)
                        end_after_concat = time.time()

                        print("final_output:", final_output)
                        print("time taken for layer after:", end_after_concat - start_after_concat)

                        action = int(np.argmax(final_output))  # replace with policy function
                        # action = int(torch.argmax(final_output))  # replace with policy function
                        reward = reward_avg

                        print("After completion of action calculation:", action)
                        print("After completion of reward calculation:", reward)

                        # curr_state = [np.expand_dims(output_right_eye_curr, axis=0),
                        #               np.expand_dims(output_left_eye_curr, axis=0)]
                        curr_state = [output_right_eye_curr, output_left_eye_curr]

                        print("After completion of curr_state calculation:", curr_state)

                        print("shape of output_right_eye_prev:", output_right_eye_prev.shape)

                        # buffer.record((np.array(prev_state), np.array([action]), np.array([reward]), np.array(curr_state)))
                        buffer.record((np.array(output_right_eye_prev), np.array(output_left_eye_prev),
                                       action, reward,
                                       np.array(output_right_eye_curr), np.array(output_left_eye_curr)))
                        buffer.learn()

                        print("pointer actor model 3")
                        update_target(target_actor, actor_model, tau)

                        print("pointer critic model 3")
                        update_target(target_critic, critic_model, tau)

                        prev_state = curr_state

                        print("music_index:", action)

                        # frame_queue.put(frame_arr, block=True)
                        # # num_frames = 32

                        # Muisc playing part
                        # current_music_index = (window_num % 5)
                        current_music_index = action
                        current_music_file = music_files[current_music_index]
                        print("current_music_file:", current_music_file)
                        music_queue.put(music_files[current_music_index])

                    frame_arr = []
                    plain_raw_video = [[], []]

                    eye_right_cropped_model_input = []
                    eye_left_cropped_model_input = []

                    print("video window num:", window_num)
                    window_num += 1
                    # window_num = 1

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