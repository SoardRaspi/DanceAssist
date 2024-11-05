import numpy as np
import cv2 as cv
import tensorflow as tf
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

from transformers import VivitImageProcessor, VivitForVideoClassification

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
model.config.output_hidden_states = True

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

def PreProcessandFreezedOutput(right_eye_video_gray, left_eye_video_gray):
    frames_right = []
    frames_left = []

    print("reached inside PreProcessandFreezedOutput")

    for i, frame in enumerate(right_eye_video_gray):
        print("frame shape:", frame.shape)

        normal = frame
        otsu_threshold, otsu_binarized = cv.threshold(
            frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
        )
        kmeans = KMeans_Image(frame)

        normal = np.expand_dims(normal, axis=-1)
        otsu_binarized = np.expand_dims(otsu_binarized, axis=-1)
        kmeans = np.expand_dims(kmeans, axis=-1)

        normal = np.concatenate((normal, otsu_binarized), axis=-1)
        normal = np.concatenate((normal, kmeans), axis=-1)

        frames_right.append(normal)

    for i, frame in enumerate(left_eye_video_gray):
        normal = frame
        otsu_threshold, otsu_binarized = cv.threshold(
            frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
        )
        kmeans = KMeans_Image(frame)

        normal = np.expand_dims(normal, axis=-1)
        otsu_binarized = np.expand_dims(otsu_binarized, axis=-1)
        kmeans = np.expand_dims(kmeans, axis=-1)

        normal = np.concatenate((normal, otsu_binarized), axis=-1)
        normal = np.concatenate((normal, kmeans), axis=-1)

        frames_left.append(normal)

    frames_right = np.array(frames_right)
    frames_left = np.array(frames_left)

    # indices_2 = sample_frame_indices_2(clip_len=32, frame_sample_rate=4, seg_len=len(frames_right))
    # video_2 = read_video_pyav_2(video_list=frames_right, indices=indices_2)

    # video_listed_2 = list(video_2)
    video_listed_2 = list(frames_right)

    inputs_2 = image_processor(video_listed_2, return_tensors="pt")

    with tf.no_gradient():
        outputs = model(**inputs_2)
        last_state_right = np.array(outputs.hidden_states)[-1]

    # indices_2 = sample_frame_indices_2(clip_len=32, frame_sample_rate=4, seg_len=len(frames_left))
    # video_2 = read_video_pyav_2(video_list=frames_left, indices=indices_2)

    # video_listed_2 = list(video_2)
    video_listed_2 = list(frames_left)

    inputs_2 = image_processor(video_listed_2, return_tensors="pt")

    with tf.no_gradient:
        outputs = model(**inputs_2)
        last_state_left = np.array(outputs.hidden_states)[-1]

    last_state_concat = np.concatenate((last_state_right, last_state_left), axis=0)
    print("shape of last_state_concat:", last_state_concat.shape)

    # Normalization after concatenation:
    concat_layer_flatten = last_state_concat.reshape(last_state_concat.shape[0], -1)
    print("shape of concat_layer_flatten:", concat_layer_flatten.shape)

    # dense_layer = nn.Linear(3137 * 768, 128)
    # dense_layer = layers.Dense((3137 * 768, 128), activation="relu")
    dense_layer = layers.Dense(128, activation="relu")

    # output_after_dense = dense_layer(torch.from_numpy(concat_layer_flatten))
    output_after_dense = dense_layer(tf.convert_to_tensor(concat_layer_flatten, dtype=tf.float32))

    output_after_dense = output_after_dense.reshape(-1)

    print("shape of output_after_dense:", output_after_dense.shape)

    # dense_layer = nn.Linear(256, 128)
    # dense_layer = layers.Dense((256, 128), activation="relu")
    dense_layer = layers.Dense(128, activation="softmax")

    # softmax = nn.Softmax()

    output_after_dense = dense_layer(output_after_dense)

    print("shape of output_after_dense 2:", output_after_dense.shape)

    # return output_after_dense
    return output_after_dense, dense_layer
