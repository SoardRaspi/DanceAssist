'''
This function takes the video of the image for the window as input and gives the last hidden layer's
weights as output. The input is for the gray-scaled cropped video for both the eyes.
'''

import numpy as np
import cv2 as cv
import os

import torch
import torch.nn as nn

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

import tensorflow as tf

from transformers import VivitImageProcessor, VivitForVideoClassification, \
    AutoImageProcessor, MobileNetV1Model

# image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
# model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
# model.config.output_hidden_states = True

image_processor_1 = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
model_1 = MobileNetV1Model.from_pretrained("google/mobilenet_v1_1.0_224")

# Get the total number of parameters
total_params = sum(p.numel() for p in model_1.parameters())

# Print the number of parameters
print(f"Total number of parameters: {total_params}")

# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# model.to(device)

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

def read_video_pyav_2(video_list, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]

    # for i, frame in enumerate(container.decode(video=0)):
    for i, frame in enumerate(video_list):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)

    return np.stack([x for x in frames])

def sample_frame_indices_2(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)

    return indices


class Model(nn.Module):
    def __init__(self, final_num_output):
        super(Model, self).__init__()
        self.dense_layer = nn.Linear(3137 * 768, 128)
        self.dense_layer_2 = nn.Linear(256, final_num_output)

        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.model.config.output_hidden_states = True

    def forward(self, frames_right, frames_left):
        video_listed_2 = list(frames_right)

        inputs_2 = self.image_processor(video_listed_2, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs_2)
            last_state_right = np.array(outputs.hidden_states)[-1]

        video_listed_2 = list(frames_left)

        inputs_2 = self.image_processor(video_listed_2, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs_2)
            last_state_left = np.array(outputs.hidden_states)[-1]

        last_state_concat = np.concatenate((last_state_right, last_state_left), axis=0)
        print("shape of last_state_concat:", last_state_concat.shape)

        # Normalization after concatenation:
        concat_layer_flatten = last_state_concat.reshape(last_state_concat.shape[0], -1)
        print("shape of concat_layer_flatten:", concat_layer_flatten.shape)

        # dense_layer = nn.Linear(3137 * 768, 128)
        output_after_dense = self.dense_layer(torch.from_numpy(concat_layer_flatten))
        output_after_dense = output_after_dense.reshape(-1)

        print("shape of output_after_dense:", output_after_dense.shape)

        softmax = nn.Softmax()
        output_after_dense = self.dense_layer_2(output_after_dense)

        print("shape of output_after_dense 2:", output_after_dense.shape)

        final_output = softmax(output_after_dense)

        print("shape of final_output:", final_output.shape)

        return output_after_dense, final_output

# def PreProcessandFreezedOutput(right_eye_video_gray, left_eye_video_gray):
#     frames_right = []
#     frames_left = []
#
#     print("reached inside PreProcessandFreezedOutput")
#
#     for i, frame in enumerate(right_eye_video_gray):
#         print("frame shape:", frame.shape)
#
#         normal = frame
#         otsu_threshold, otsu_binarized = cv.threshold(
#             frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
#         )
#         kmeans = KMeans_Image(frame)
#
#         normal = np.expand_dims(normal, axis=-1)
#         otsu_binarized = np.expand_dims(otsu_binarized, axis=-1)
#         kmeans = np.expand_dims(kmeans, axis=-1)
#
#         normal = np.concatenate((normal, otsu_binarized), axis=-1)
#         normal = np.concatenate((normal, kmeans), axis=-1)
#
#         frames_right.append(normal)
#
#     for i, frame in enumerate(left_eye_video_gray):
#         normal = frame
#         otsu_threshold, otsu_binarized = cv.threshold(
#             frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
#         )
#         kmeans = KMeans_Image(frame)
#
#         normal = np.expand_dims(normal, axis=-1)
#         otsu_binarized = np.expand_dims(otsu_binarized, axis=-1)
#         kmeans = np.expand_dims(kmeans, axis=-1)
#
#         normal = np.concatenate((normal, otsu_binarized), axis=-1)
#         normal = np.concatenate((normal, kmeans), axis=-1)
#
#         frames_left.append(normal)
#
#     frames_right = np.array(frames_right)
#     frames_left = np.array(frames_left)
#
#     # indices_2 = sample_frame_indices_2(clip_len=32, frame_sample_rate=4, seg_len=len(frames_right))
#     # video_2 = read_video_pyav_2(video_list=frames_right, indices=indices_2)
#
#     # video_listed_2 = list(video_2)
#     video_listed_2 = list(frames_right)
#
#     inputs_2 = image_processor(video_listed_2, return_tensors="pt")
#
#     with torch.no_grad():
#         outputs = model(**inputs_2)
#         last_state_right = np.array(outputs.hidden_states)[-1]
#
#     # indices_2 = sample_frame_indices_2(clip_len=32, frame_sample_rate=4, seg_len=len(frames_left))
#     # video_2 = read_video_pyav_2(video_list=frames_left, indices=indices_2)
#
#     # video_listed_2 = list(video_2)
#     video_listed_2 = list(frames_left)
#
#     inputs_2 = image_processor(video_listed_2, return_tensors="pt")
#
#     with torch.no_grad():
#         outputs = model(**inputs_2)
#         last_state_left = np.array(outputs.hidden_states)[-1]
#
#     ## Concatenation of inputs starts
#     last_state_concat = np.concatenate((last_state_right, last_state_left), axis=0)
#     print("shape of last_state_concat:", last_state_concat.shape)
#
#     # Normalization after concatenation:
#     concat_layer_flatten = last_state_concat.reshape(last_state_concat.shape[0], -1)
#     print("shape of concat_layer_flatten:", concat_layer_flatten.shape)
#     ## Concatenation of inputs ends
#
#     dense_layer = nn.Linear(3137 * 768, 128)
#     output_after_dense = dense_layer(torch.from_numpy(concat_layer_flatten))
#     output_after_dense = output_after_dense.reshape(-1)
#
#     print("shape of output_after_dense:", output_after_dense.shape)
#
#     dense_layer = nn.Linear(256, 128)
#     softmax = nn.Softmax()
#     output_after_dense = dense_layer(output_after_dense)
#
#     print("shape of output_after_dense 2:", output_after_dense.shape)
#
#     final_output = softmax(output_after_dense)
#
#     print("shape of final_output:", final_output.shape)
#
#     # return output_after_dense
#     return output_after_dense, final_output

def PreProcessandFreezedOutput_MobileNetV1(right_eye, left_eye):
    inputs = image_processor_1(right_eye, return_tensors="pt")
    with torch.no_grad():
        output_right_eye = model_1(**inputs)

    inputs = image_processor_1(left_eye, return_tensors="pt")
    with torch.no_grad():
        output_left_eye = model_1(**inputs)

    output_right_eye = output_right_eye.last_hidden_state
    output_left_eye = output_left_eye.last_hidden_state

    ## Concatenation of inputs starts
    last_state_concat = np.concatenate((output_right_eye, output_left_eye), axis=0)
    print("shape of last_state_concat:", last_state_concat.shape)

    # Normalization after concatenation:
    concat_layer_flatten = last_state_concat.reshape(last_state_concat.shape[0], -1)
    print("shape of concat_layer_flatten:", concat_layer_flatten.shape)
    ## Concatenation of inputs ends

    # dense_layer = nn.Linear(3137 * 768, 128)
    dense_layer = nn.Linear(50176, 128)
    output_after_dense = dense_layer(torch.from_numpy(concat_layer_flatten))
    output_after_dense = output_after_dense.reshape(-1)

    print("shape of output_after_dense:", output_after_dense.shape)

    dense_layer_1 = nn.Linear(256, 128)
    dense_layer_2 = nn.Linear(128, 64)
    dense_layer_3 = nn.Linear(64, 5)
    softmax = nn.Softmax()
    sigmoid = nn.Sigmoid()
    output_after_dense = dense_layer_1(output_after_dense)
    output_after_dense = dense_layer_2(output_after_dense)
    output_after_dense = dense_layer_3(output_after_dense)

    print("shape of output_after_dense 2:", output_after_dense.shape)

    final_output = softmax(output_after_dense)
    # final_output = sigmoid(output_after_dense)

    print("shape of final_output:", final_output.shape)

    # return output_after_dense
    return output_after_dense, final_output

class Actor_PreProcessandFreezedOutput_MobileNetV1_Model_Tf(keras.Model):
    def __init__(self):
        super(Actor_PreProcessandFreezedOutput_MobileNetV1_Model_Tf, self).__init__()

        # Define the image processor and model
        self.image_processor_1 = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
        self.model_1 = MobileNetV1Model.from_pretrained("google/mobilenet_v1_1.0_224")

        # Define the dense layers
        self.dense_layer_1 = layers.Dense(128, input_shape=(50176,))
        self.dense_layer_1_conc = layers.Dense(128)
        self.dense_layer_2 = layers.Dense(64)
        self.dense_layer_3 = layers.Dense(5)
        self.softmax = layers.Softmax()
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, right_eye, left_eye):
        print("data from Actor Model tf starts")

        # Process the right eye input
        inputs = self.image_processor_1(right_eye, return_tensors="pt")
        with torch.no_grad():
            output_right_eye = self.model_1(**inputs).last_hidden_state

        output_right_eye = output_right_eye.detach().numpy()
        output_right_eye = tf.convert_to_tensor(output_right_eye, dtype=tf.float32)
        print("output_right_eye in Actor model tf type:", type(output_right_eye))

        # Process the left eye input
        inputs = self.image_processor_1(left_eye, return_tensors="pt")
        with torch.no_grad():
            output_left_eye = self.model_1(**inputs).last_hidden_state

        output_left_eye = output_left_eye.detach().numpy()
        output_left_eye = tf.convert_to_tensor(output_left_eye, dtype=tf.float32)
        print("output_left_eye in Actor model tf type:", type(output_left_eye))

        # Concatenation of the last hidden states
        last_state_concat = tf.concat([output_right_eye, output_left_eye], axis=0)
        print("shape of last_state_concat:", last_state_concat.shape)

        # Flatten the concatenated states
        concat_layer_flatten = tf.reshape(last_state_concat, (tf.shape(last_state_concat)[0], -1))
        print("shape of concat_layer_flatten:", concat_layer_flatten.shape)

        # Pass through the first dense layer
        output_after_dense = self.dense_layer_1(concat_layer_flatten)
        output_after_dense = tf.reshape(output_after_dense, (-1,))
        output_after_dense = tf.expand_dims(output_after_dense, axis=0)
        print("shape of output_after_dense:", output_after_dense.shape)

        output_after_dense = self.dense_layer_1_conc(output_after_dense)

        # Pass through the additional dense layers
        output_after_dense = self.dense_layer_2(output_after_dense)
        output_after_dense = self.dense_layer_3(output_after_dense)
        # output_after_dense = tf.reshape(output_after_dense, (1, -1))
        print("shape of output_after_dense 2:", output_after_dense.shape, output_after_dense)

        # Apply softmax or sigmoid for the final output
        final_output = self.softmax(output_after_dense)
        # Alternatively, use sigmoid
        # final_output = self.sigmoid(output_after_dense)

        print("shape of final_output:", final_output.shape, final_output)
        print("data from Actor Model ends")

        return output_after_dense, final_output


class Actor_PreProcessandFreezedOutput_MobileNetV1_Model(nn.Module):
    def __init__(self):
        super(Actor_PreProcessandFreezedOutput_MobileNetV1_Model, self).__init__()

        # Define the dense layers
        self.dense_layer_1 = nn.Linear(50176, 128)
        self.dense_layer_1_conc = nn.Linear(256, 128)
        self.dense_layer_2 = nn.Linear(128, 64)
        self.dense_layer_3 = nn.Linear(64, 5)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.image_processor_1 = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
        self.model_1 = MobileNetV1Model.from_pretrained("google/mobilenet_v1_1.0_224")

    # def forward(self, model_input):
    def forward(self, right_eye, left_eye):
        print("data from Actor Model starts")
        # right_eye = model_input[0]
        # left_eye = model_input[1]

        print("early input shapes in Actor model:", right_eye.shape, left_eye.shape)

        # Process the right eye input
        inputs = self.image_processor_1(right_eye, return_tensors="pt")
        with torch.no_grad():
            output_right_eye = self.model_1(**inputs).last_hidden_state

        print("output_right_eye in Actor model type:", type(output_right_eye))

        # Process the left eye input
        inputs = self.image_processor_1(left_eye, return_tensors="pt")
        with torch.no_grad():
            output_left_eye = self.model_1(**inputs).last_hidden_state

        # Concatenation of the last hidden states
        last_state_concat = np.concatenate((output_right_eye, output_left_eye), axis=0)
        print("shape of last_state_concat:", last_state_concat.shape)

        # Flatten the concatenated states
        concat_layer_flatten = last_state_concat.reshape(last_state_concat.shape[0], -1)
        print("shape of concat_layer_flatten:", concat_layer_flatten.shape)

        # Convert numpy array to torch tensor
        concat_layer_flatten = torch.from_numpy(concat_layer_flatten).float()

        # Pass through the first dense layer
        output_after_dense = self.dense_layer_1(concat_layer_flatten)
        output_after_dense = output_after_dense.reshape(-1)
        print("shape of output_after_dense:", output_after_dense.shape)

        output_after_dense = self.dense_layer_1_conc(output_after_dense)

        # Pass through the additional dense layers
        output_after_dense = self.dense_layer_2(output_after_dense)
        output_after_dense = self.dense_layer_3(output_after_dense)
        output_after_dense = output_after_dense.reshape(1, -1)
        print("shape of output_after_dense 2:", output_after_dense.shape, output_after_dense)

        # Apply softmax or sigmoid for the final output
        final_output = self.softmax(output_after_dense)
        # Alternatively, use sigmoid
        # final_output = self.sigmoid(output_after_dense)

        print("shape of final_output:", final_output.shape, final_output)
        print("data from Actor Model ends")

        return output_after_dense, final_output

class Critic_PreProcessandFreezedOutput_MobileNetV1_Model_Tf(keras.Model):
    def __init__(self):
        super(Critic_PreProcessandFreezedOutput_MobileNetV1_Model_Tf, self).__init__()

        # Define the image processor and model
        self.image_processor_1 = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
        self.model_1 = MobileNetV1Model.from_pretrained("google/mobilenet_v1_1.0_224")

        # Define the dense layers
        self.dense_layer_1 = keras.layers.Dense(128, input_shape=(50176,))
        self.dense_layer_1_conc = keras.layers.Dense(128)
        self.dense_layer_2 = keras.layers.Dense(64)
        self.dense_layer_3 = keras.layers.Dense(32)
        self.dense_layer_parallel_1 = keras.layers.Dense(16)
        self.dense_layer_parallel_2 = keras.layers.Dense(32)
        self.dense_layer_comb_1 = keras.layers.Dense(32)
        self.dense_layer_comb_2 = keras.layers.Dense(8)
        self.dense_layer_final = keras.layers.Dense(1)

    def call(self, right_eye, left_eye, action):
        print("data from Critic Model starts")

        print("early input shapes in Actor model:", right_eye.shape, left_eye.shape)

        # Process the right eye input
        inputs = self.image_processor_1(right_eye, return_tensors="pt")
        with torch.no_grad():
            output_right_eye = self.model_1(**inputs).last_hidden_state

        # Process the left eye input
        inputs = self.image_processor_1(left_eye, return_tensors="pt")
        with torch.no_grad():
            output_left_eye = self.model_1(**inputs).last_hidden_state

        # Concatenation of the last hidden states
        last_state_concat = tf.concat([output_right_eye, output_left_eye], axis=0)
        print("shape of last_state_concat:", last_state_concat.shape)

        # Flatten the concatenated states
        concat_layer_flatten = tf.reshape(last_state_concat, (tf.shape(last_state_concat)[0], -1))
        print("shape of concat_layer_flatten:", concat_layer_flatten.shape)

        # Pass through the first dense layer
        output_after_dense = self.dense_layer_1(concat_layer_flatten)
        output_after_dense = tf.reshape(output_after_dense, (-1,))
        print("shape of output_after_dense:", output_after_dense.shape)

        output_after_dense = self.dense_layer_1_conc(output_after_dense)

        # Pass through the additional dense layers
        output_after_dense = self.dense_layer_2(output_after_dense)
        output_after_dense = self.dense_layer_3(output_after_dense)
        output_after_dense = tf.reshape(output_after_dense, (1, -1))
        print("shape of output_after_dense 2:", output_after_dense.shape)

        # Process action input through parallel dense layers
        action = self.dense_layer_parallel_1(action)
        action = self.dense_layer_parallel_2(action)

        # Combine action and output_after_dense
        comb = tf.concat([action, output_after_dense], axis=0)
        comb = tf.reshape(comb, (tf.shape(comb)[0], -1))

        # Apply the combined dense layers
        comb = self.dense_layer_comb_1(comb)
        comb = self.dense_layer_comb_2(comb)

        # Final output
        final_output = self.dense_layer_final(comb)

        print("shape of final_output:", final_output.shape)
        print("data from Critic Model ends")

        return final_output

class Critic_PreProcessandFreezedOutput_MobileNetV1_Model(nn.Module):
    def __init__(self):
        super(Critic_PreProcessandFreezedOutput_MobileNetV1_Model, self).__init__()

        # Define the dense layers
        self.dense_layer_1 = nn.Linear(50176, 128)
        self.dense_layer_1_conc = nn.Linear(256, 128)
        self.dense_layer_2 = nn.Linear(128, 64)
        self.dense_layer_3 = nn.Linear(64, 32)
        self.dense_layer_parallel_1 = nn.Linear(5, 16)
        self.dense_layer_parallel_2 = nn.Linear(16, 32)
        self.dense_layer_comb_1 = nn.Linear(64, 32)
        self.dense_layer_comb_2 = nn.Linear(32, 8)
        self.dense_layer_final = nn.Linear(8, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.image_processor_1 = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
        self.model_1 = MobileNetV1Model.from_pretrained("google/mobilenet_v1_1.0_224")

    # def forward(self, state_batch, action):
    def forward(self, right_eye, left_eye, action):
        print("data from Critic Model starts")
        # right_eye = state_batch[0]
        # left_eye = state_batch[1]

        print("early input shapes in Actor model:", right_eye.shape, left_eye.shape)

        # Process the right eye input
        inputs = self.image_processor_1(right_eye, return_tensors="pt")
        with torch.no_grad():
            output_right_eye = self.model_1(**inputs).last_hidden_state

        # Process the left eye input
        inputs = self.image_processor_1(left_eye, return_tensors="pt")
        with torch.no_grad():
            output_left_eye = self.model_1(**inputs).last_hidden_state

        # Concatenation of the last hidden states
        last_state_concat = np.concatenate((output_right_eye, output_left_eye), axis=0)
        print("shape of last_state_concat:", last_state_concat.shape)

        # Flatten the concatenated states
        concat_layer_flatten = last_state_concat.reshape(last_state_concat.shape[0], -1)
        print("shape of concat_layer_flatten:", concat_layer_flatten.shape)

        # Convert numpy array to torch tensor
        concat_layer_flatten = torch.from_numpy(concat_layer_flatten).float()

        # Pass through the first dense layer
        output_after_dense = self.dense_layer_1(concat_layer_flatten)
        output_after_dense = output_after_dense.reshape(-1)
        print("shape of output_after_dense:", output_after_dense.shape)

        output_after_dense = self.dense_layer_1_conc(output_after_dense)

        # Pass through the additional dense layers
        output_after_dense = self.dense_layer_2(output_after_dense)
        output_after_dense = self.dense_layer_3(output_after_dense)
        output_after_dense = output_after_dense.reshape(1, -1)
        print("shape of output_after_dense 2:", output_after_dense.shape)

        action = self.dense_layer_parallel_1(action)
        action = self.dense_layer_parallel_2(action)

        comb = np.concatenate((action, output_after_dense), axis=0)
        comb = comb.reshape(last_state_concat.shape[0], -1)

        comb = self.dense_layer_comb_1(comb)
        comb = self.dense_layer_comb_2(comb)

        # Apply softmax or sigmoid for the final output
        # final_output = self.softmax(output_after_dense)
        final_output = self.dense_layer_final(comb)
        # Alternatively, use sigmoid
        # final_output = self.sigmoid(output_after_dense)

        print("shape of final_output:", final_output.shape)
        print("data from Critic Model ends")

        return final_output
