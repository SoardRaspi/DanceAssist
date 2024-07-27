##  Goal 0: Create a basic transformer structure..., but why?
##  Goal 1: Given a final position, the model should be able to generate a sequence
##  Goal 2: Given a starting position, the model should be able to generate a sequence
##  Goal 3: Combine the two
#
#  The goal is to see the areas where the motions are similar. This can be done by considering a window.
#  Compare in that window. See the similarity and try to make it perfect.

#  Set an initial frequency, and then change it with iterations

#  The model is trying to find the best beat parameters for the given motion sequence. The loss function is the DTW
#  distance. Finally, the model should give out the beat parameters when an input motion sequence is given.

import json

import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from PIL import Image
from dtw import dtw
from music_generator import play_tick

# filename = "right_arm_focus.json"
input_loc = "correlation_images_2"

plt_1 = input_loc + "/done_seq_1.png"
plt_2 = input_loc + "/done_seq_2.png"

plt_1 = Image.open(plt_1)
plt_1 = asarray(plt_1)
plt_1 = plt_1 / 256

plt_2 = Image.open(plt_2)
plt_2 = asarray(plt_2)
plt_2 = plt_2 / 256

# print("plt_1:", plt_1)

# # defining all 3 axis
# z_1 = []
# x_1 = []
# y_1 = []
#
# z_2 = []
# x_2 = []
# y_2 = []
#
# #  x, y, z
# # sa,ea,wa
#
# for row in plt_1:
#     x_1.append(row[0])
#     y_1.append(row[1])
#     z_1.append(row[2])
#
# for row in plt_2:
#     x_2.append(row[0])
#     y_2.append(row[1])
#     z_2.append(row[2])
#
# fig = plt.figure()
#
# # syntax for 3-D projection
# ax = plt.axes(projection='3d')
#
# # plotting
# ax.plot3D(x_1, y_1, z_1, 'green')
# ax.plot3D(x_2, y_2, z_2, 'blue')
# ax.set_title('3D line plot geeks for geeks')
# plt.show()

# Function to compute the Euclidean distance between two points
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

# # Sample data: sequences of joint angles (shoulder, elbow, wrist)
# # Each sequence is an array of records, each record is an array of [shoulder_angle, elbow_angle, wrist_angle]
# sequence1 = np.array([
#     [30, 45, 60],
#     [31, 46, 61],
#     [32, 47, 62],
#     [33, 48, 63],
#     [34, 49, 64]
# ])
#
# sequence2 = np.array([
#     [35, 50, 65],
#     [36, 51, 66],
#     [37, 52, 67],
#     [38, 53, 68]
# ])

# Perform Dynamic Time Warping
# alignment = dtw(sequence1, sequence2)
# alignment = dtw(plt_1, plt_1)
# alignment = dtw(plt_1, plt_2)

def dtw_score(plt_1_inner, plt_2_inner):
    return dtw(plt_1_inner, plt_2_inner).distance

# # Print the alignment distance and the mapping of sequences
# print(f'DTW distance: {alignment.distance}')
# print('Index mapping:')
# for index1, index2 in zip(alignment.index1, alignment.index2):
#     print(f'Sequence1[{index1}] -> Sequence2[{index2}]')

# # Metronome settings
# frequency = 1000  # Frequency of the tick sound in Hz
# duration = 0.1  # Duration of the tick sound in seconds
# bpm = 120  # Beats per minute
# interval = 60 / bpm  # Interval between ticks in seconds
#
# # Run the metronome
# # while True:
# #     play_tick(frequency, duration, interval)