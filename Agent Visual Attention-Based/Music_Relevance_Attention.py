# # # from pydub import AudioSegment
# # # import librosa
# # # import numpy as np
# # #
# # # # Load the first audio file
# # # audio1 = AudioSegment.from_file("Chaleya.mp3")
# # #
# # # # Load the second audio file
# # # audio2 = AudioSegment.from_file("Queen - We Will Rock You (Official Video).mp3")
# # #
# # # # Optional: Adjust volumes (e.g., reduce volume of the second track by 6dB)
# # # audio2 = audio2 - 6
# # #
# # # # Optional: Adjust timing (e.g., add silence to the beginning of the second track)
# # # silence = AudioSegment.silent(duration=2000)  # 2 seconds of silence
# # # audio2 = silence + audio2
# # #
# # # # Combine the two tracks (overlay the second track on the first)
# # # combined = audio1.overlay(audio2)
# # #
# # # # Export the combined audio to a file
# # # combined.export("combined_audio.mp3", format="mp3")
# # #
# # # print("Combined audio saved as 'combined_audio.mp3'")
# #
# # mfcc = [-0.3007914 ,  0.5769816 , -0.03873457,  0.35950872,  0.04860171,
# #         0.14289656,  0.14038755,  0.20856857,  0.17212985,  0.3575108 ,
# #         0.05666408, -0.01705325, -0.03032584,  0.18391235,  0.10758709,
# #        -0.11695638, -0.15475251,  0.11984038,  0.11715946,  0.03541001,
# #        -0.20745099, -0.04511094, -0.23520747, -0.06589869, -0.03102105,
# #        -0.07771166, -0.03916495,  0.06946871,  0.01059791, -0.15406805,
# #        -0.1028441 ,  0.19268863, -0.1785376 , -0.06809582, -0.09669289,
# #        -0.03544096, -0.23908912, -0.02975397,  0.18561341, -0.06280752]
# #
# # import numpy as np
# # import librosa
# # import sounddevice as sd
# #
# # # Function to generate dummy audio features (e.g., MFCCs) and convert to waveform
# # def generate_audio_from_features(features, sr=22050):
# #     # Inverting MFCCs back to waveform (simplified for demonstration purposes)
# #     # This method is highly lossy and is not the correct way to invert MFCCs
# #     # Normally, you would need to have a proper inversion method based on your feature extraction process
# #     D = librosa.feature.inverse.mfcc_to_mel(features)
# #     y = librosa.feature.inverse.mel_to_audio(D, sr=sr)
# #     return y
# #
# # # generated_features = np.reshape(np.random.rand(40), (1, 40))
# # generated_features = np.reshape(mfcc, (1, 40))
# #
# # print(generated_features.shape)
# # print(generated_features)
# #
# # # import tensorflow as tf
# # # from tensorflow.python import keras
# # #
# # # # Load the model from the .h5 file
# # # model = keras.models.load_model('Music_by_genre/DA_MG_v1.h5')
# #
# # # Convert the features to audio waveform
# # audio_waveform = generate_audio_from_features(generated_features)
# #
# # # Play the audio waveform
# # sd.play(audio_waveform, samplerate=22050)
# # sd.wait()
#
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import cv2 as cv
#
# vx, vy, vz = [], [], []
#
# def print_face_landmarks(vx, vy, vz):
#     mpPose = mp.solutions.pose
#     pose = mpPose.Pose()
#     mpDraw = mp.solutions.drawing_utils
#
#     v_landmarks = {}
#     landmarks_dict = {}
#     delta_time = 1
#
#     cap = cv.VideoCapture(0)
#
#     main_face = [i for i in range(9)]
#
#     while True:
#         start_time = time.time()
#         success, img = cap.read()
#
#         if not success:
#             print("error opening camera")
#             break
#
#         imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#         results = pose.process(imgRGB)
#
#         ### 32 imp body landmarks ###
#         # cv.imshow("Original", img)
#
#         if results.pose_landmarks:
#             # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
#
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 # print("inner_data:", id, lm)
#                 # print("inner_data types:", type(id), type(lm))
#                 id_temp = int(id)
#
#                 if id_temp in main_face:
#                     cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
#                     cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
#
#                     if id not in v_landmarks:
#                         v_landmarks[id] = 0
#                     else:
#                         # print(v_landmarks)
#                         # print(landmarks_dict)
#
#                         v_landmarks[id] = [(cx - landmarks_dict[id_temp][0]) / delta_time,
#                                            (cy - landmarks_dict[id_temp][1]) / delta_time,
#                                            (cz - landmarks_dict[id_temp][2]) / delta_time]
#
#                     landmarks_dict[id_temp] = [cx, cy, cz]
#
#             try:
#                 vx.append(sum(v_landmarks[iddd][0] for iddd in range(9))/9)
#                 vy.append(sum(v_landmarks[iddd][1] for iddd in range(9))/9)
#                 vz.append(sum(v_landmarks[iddd][2] for iddd in range(9))/9)
#             except:
#                 vx.append(0)
#                 vy.append(0)
#                 vz.append(0)
#
#             # print(landmarks_dict)
#
#         cv.imshow("Landmarks", img)
#
#         end_time = time.time()
#         delta_time = end_time - start_time
#
#         k = cv.waitKey(30) & 0xff
#         if k == 27:
#             break
#
#     cap.release()
#     cv.destroyAllWindows()
#
#     return vx, vy, vz
#
#
# vx, vy, vz = print_face_landmarks(vx, vy, vz)
#
# print(vx)
# print(vy)
# print(vz)
#
# ## Grouping the landmarks
# # Generate data
# x = np.linspace(0, 10, len(vx))
# y1 = np.array(vx)
# y1 = (y1 - np.min(y1)) / (np.max(y1) - np.min(y1))
#
# y2 = np.array(vx)
# y2 = (y2 - np.min(y2)) / (np.max(y2) - np.min(y2))
#
# y3 = np.array(vx)
# y3 = (y3 - np.min(y3)) / (np.max(y3) - np.min(y3))
#
# # Create a figure and a 3x1 grid of subplots
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
#
# # Plot vx on the first subplot
# ax1.plot(y1, 'b-')
# ax1.set_title('vx')
# ax1.set_xlabel('Index')
# ax1.set_ylabel('Value')
#
# # Plot vy on the second subplot
# ax2.plot(y2, 'r-')
# ax2.set_title('vy')
# ax2.set_xlabel('Index')
# ax2.set_ylabel('Value')
#
# # Plot vz on the third subplot
# ax3.plot(y3, 'g-')
# ax3.set_title('vz')
# ax3.set_xlabel('Index')
# ax3.set_ylabel('Value')
#
# # Adjust layout to prevent overlap
# plt.tight_layout()
#
# # Show the plot
# plt.show()

# import tensorflow as tf
# from tensorflow.python import keras
# from tensorflow.python.keras import Model, Input
# from tensorflow.python.keras.layers import Conv2D, Concatenate, Dense

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import Counter

from sklearn.cluster import MeanShift

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='mediapipe_face',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'mediapipe_face':
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 7),
                             (0, 4), (4, 5), (5, 6), (6, 8)]
            self.edge = self_link + neighbor_link
            self.center = 0
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        # self.graph = Graph(**graph_args)
        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):
        print("shape of x:", x.size())

        # N: batch size.
        # C: original node features, i.e.triplet of(Coordinate - X / Coordinate - Y / Confidence - Level).
        # T: time steps.
        # V: number of nodes in Graph.
        # M: number of skeletons in a data record.

        # # data normalization
        # V, T, C = x.size()
        # N, M = 1, 1
        # x = x.permute(0, 4, 3, 1, 2).contiguous()
        # x = x.view(N * M, V * C, T)
        # x = self.data_bn(x)
        # x = x.view(N, M, V, C, T)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = x.view(N * M, C, T, V)

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        # return output, feature
        return output

# print(Model)
# input_test = torch.tensor([[[ 0.5680,  0.6881, -1.1487],
#          [ 0.5680,  0.6880, -1.0757],
#          [ 0.5682,  0.6879, -1.0984],
#          [ 0.5685,  0.6888, -1.0954],
#          [ 0.5687,  0.6860, -1.0257],
#          [ 0.5688,  0.6879, -1.0477],
#          [ 0.5688,  0.6889, -1.0965],
#          [ 0.5688,  0.6904, -1.0809],
#          [ 0.5690,  0.6905, -1.1033],
#          [ 0.5692,  0.6914, -1.1011]],
#         [[ 0.5936,  0.5895, -1.0901],
#          [ 0.5936,  0.5897, -1.0102],
#          [ 0.5939,  0.5896, -1.0305],
#          [ 0.5939,  0.5916, -1.0263],
#          [ 0.5940,  0.5900, -0.9659],
#          [ 0.5941,  0.5918, -0.9868],
#          [ 0.5942,  0.5923, -1.0362],
#          [ 0.5942,  0.5934, -1.0319],
#          [ 0.5945,  0.5934, -1.0474],
#          [ 0.5949,  0.5942, -1.0425]],
#         [[ 0.6104,  0.5901, -1.0902],
#          [ 0.6104,  0.5904, -1.0104],
#          [ 0.6106,  0.5903, -1.0306],
#          [ 0.6106,  0.5924, -1.0263],
#          [ 0.6107,  0.5908, -0.9660],
#          [ 0.6107,  0.5926, -0.9871],
#          [ 0.6109,  0.5930, -1.0362],
#          [ 0.6110,  0.5940, -1.0317],
#          [ 0.6113,  0.5939, -1.0475],
#          [ 0.6117,  0.5946, -1.0425]],
#         [[ 0.6262,  0.5908, -1.0900],
#          [ 0.6262,  0.5915, -1.0102],
#          [ 0.6264,  0.5913, -1.0304],
#          [ 0.6264,  0.5936, -1.0261],
#          [ 0.6264,  0.5920, -0.9658],
#          [ 0.6265,  0.5938, -0.9869],
#          [ 0.6268,  0.5941, -1.0358],
#          [ 0.6269,  0.5949, -1.0313],
#          [ 0.6272,  0.5948, -1.0472],
#          [ 0.6276,  0.5954, -1.0423]],
#         [[ 0.5364,  0.5876, -1.1028],
#          [ 0.5365,  0.5876, -1.0290],
#          [ 0.5371,  0.5875, -1.0465],
#          [ 0.5377,  0.5885, -1.0413],
#          [ 0.5380,  0.5872, -0.9805],
#          [ 0.5382,  0.5884, -1.0011],
#          [ 0.5382,  0.5895, -1.0466],
#          [ 0.5383,  0.5909, -1.0411],
#          [ 0.5385,  0.5911, -1.0597],
#          [ 0.5387,  0.5916, -1.0552]],
#         [[ 0.5170,  0.5874, -1.1025],
#          [ 0.5172,  0.5874, -1.0289],
#          [ 0.5178,  0.5873, -1.0463],
#          [ 0.5187,  0.5880, -1.0409],
#          [ 0.5189,  0.5868, -0.9802],
#          [ 0.5192,  0.5878, -1.0009],
#          [ 0.5193,  0.5892, -1.0463],
#          [ 0.5195,  0.5905, -1.0406],
#          [ 0.5197,  0.5907, -1.0594],
#          [ 0.5199,  0.5912, -1.0549]],
#         [[ 0.4992,  0.5891, -1.1029],
#          [ 0.4994,  0.5891, -1.0294],
#          [ 0.5002,  0.5890, -1.0468],
#          [ 0.5014,  0.5896, -1.0412],
#          [ 0.5018,  0.5881, -0.9806],
#          [ 0.5021,  0.5891, -1.0014],
#          [ 0.5022,  0.5906, -1.0466],
#          [ 0.5024,  0.5920, -1.0409],
#          [ 0.5025,  0.5922, -1.0599],
#          [ 0.5027,  0.5926, -1.0553]],
#         [[ 0.6502,  0.6209, -0.6601],
#          [ 0.6502,  0.6217, -0.5778],
#          [ 0.6504,  0.6214, -0.5833],
#          [ 0.6503,  0.6235, -0.5819],
#          [ 0.6503,  0.6229, -0.5550],
#          [ 0.6503,  0.6241, -0.5658],
#          [ 0.6510,  0.6244, -0.6092],
#          [ 0.6512,  0.6246, -0.6058],
#          [ 0.6516,  0.6246, -0.6278],
#          [ 0.6521,  0.6246, -0.6208]],
#         [[ 0.4687,  0.6271, -0.7067],
#          [ 0.4688,  0.6271, -0.6488],
#          [ 0.4694,  0.6269, -0.6526],
#          [ 0.4699,  0.6271, -0.6502],
#          [ 0.4701,  0.6258, -0.6145],
#          [ 0.4704,  0.6262, -0.6239],
#          [ 0.4708,  0.6267, -0.6526],
#          [ 0.4710,  0.6274, -0.6484],
#          [ 0.4711,  0.6274, -0.6791],
#          [ 0.4712,  0.6275, -0.6711]]])
#
# input_test_2 = torch.tensor([[[ 0.5680,  0.5936,  0.6104,  0.6262,  0.5364,  0.5170,  0.4992,
#            0.6502,  0.4687],
#          [ 0.5680,  0.5936,  0.6104,  0.6262,  0.5365,  0.5172,  0.4994,
#            0.6502,  0.4688],
#          [ 0.5682,  0.5939,  0.6106,  0.6264,  0.5371,  0.5178,  0.5002,
#            0.6504,  0.4694],
#          [ 0.5685,  0.5939,  0.6106,  0.6264,  0.5377,  0.5187,  0.5014,
#            0.6503,  0.4699],
#          [ 0.5687,  0.5940,  0.6107,  0.6264,  0.5380,  0.5189,  0.5018,
#            0.6503,  0.4701],
#          [ 0.5688,  0.5941,  0.6107,  0.6265,  0.5382,  0.5192,  0.5021,
#            0.6503,  0.4704],
#          [ 0.5688,  0.5942,  0.6109,  0.6268,  0.5382,  0.5193,  0.5022,
#            0.6510,  0.4708],
#          [ 0.5688,  0.5942,  0.6110,  0.6269,  0.5383,  0.5195,  0.5024,
#            0.6512,  0.4710],
#          [ 0.5690,  0.5945,  0.6113,  0.6272,  0.5385,  0.5197,  0.5025,
#            0.6516,  0.4711],
#          [ 0.5692,  0.5949,  0.6117,  0.6276,  0.5387,  0.5199,  0.5027,
#            0.6521,  0.4712]],
#
#         [[ 0.6881,  0.5895,  0.5901,  0.5908,  0.5876,  0.5874,  0.5891,
#            0.6209,  0.6271],
#          [ 0.6880,  0.5897,  0.5904,  0.5915,  0.5876,  0.5874,  0.5891,
#            0.6217,  0.6271],
#          [ 0.6879,  0.5896,  0.5903,  0.5913,  0.5875,  0.5873,  0.5890,
#            0.6214,  0.6269],
#          [ 0.6888,  0.5916,  0.5924,  0.5936,  0.5885,  0.5880,  0.5896,
#            0.6235,  0.6271],
#          [ 0.6860,  0.5900,  0.5908,  0.5920,  0.5872,  0.5868,  0.5881,
#            0.6229,  0.6258],
#          [ 0.6879,  0.5918,  0.5926,  0.5938,  0.5884,  0.5878,  0.5891,
#            0.6241,  0.6262],
#          [ 0.6889,  0.5923,  0.5930,  0.5941,  0.5895,  0.5892,  0.5906,
#            0.6244,  0.6267],
#          [ 0.6904,  0.5934,  0.5940,  0.5949,  0.5909,  0.5905,  0.5920,
#            0.6246,  0.6274],
#          [ 0.6905,  0.5934,  0.5939,  0.5948,  0.5911,  0.5907,  0.5922,
#            0.6246,  0.6274],
#          [ 0.6914,  0.5942,  0.5946,  0.5954,  0.5916,  0.5912,  0.5926,
#            0.6246,  0.6275]],
#
#         [[-1.1487, -1.0901, -1.0902, -1.0900, -1.1028, -1.1025, -1.1029,
#           -0.6601, -0.7067],
#          [-1.0757, -1.0102, -1.0104, -1.0102, -1.0290, -1.0289, -1.0294,
#           -0.5778, -0.6488],
#          [-1.0984, -1.0305, -1.0306, -1.0304, -1.0465, -1.0463, -1.0468,
#           -0.5833, -0.6526],
#          [-1.0954, -1.0263, -1.0263, -1.0261, -1.0413, -1.0409, -1.0412,
#           -0.5819, -0.6502],
#          [-1.0257, -0.9659, -0.9660, -0.9658, -0.9805, -0.9802, -0.9806,
#           -0.5550, -0.6145],
#          [-1.0477, -0.9868, -0.9871, -0.9869, -1.0011, -1.0009, -1.0014,
#           -0.5658, -0.6239],
#          [-1.0965, -1.0362, -1.0362, -1.0358, -1.0466, -1.0463, -1.0466,
#           -0.6092, -0.6526],
#          [-1.0809, -1.0319, -1.0317, -1.0313, -1.0411, -1.0406, -1.0409,
#           -0.6058, -0.6484],
#          [-1.1033, -1.0474, -1.0475, -1.0472, -1.0597, -1.0594, -1.0599,
#           -0.6278, -0.6791],
#          [-1.1011, -1.0425, -1.0425, -1.0423, -1.0552, -1.0549, -1.0553,
#           -0.6208, -0.6711]]])
# # input_test_2 = input_test_2.unsqueeze(-1).unsqueeze(0)
#
# # print(input_test_2.shape)
# # model = Model(in_channels=3, num_class=16, graph_args={}, edge_importance_weighting=True)
# # result = model.extract_feature(input_test_2)
# # result = result.squeeze(-1).squeeze(0)

class ST_GCN_Post(nn.Module):
    def __init__(self):
        super(ST_GCN_Post, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(6 * 9, 10)  # Adjust the input size according to the flattened dimension
        self.ST_GCN = Model(in_channels=3, num_class=16, graph_args={}, edge_importance_weighting=True)

    def forward(self, x):
        x = x.permute(2, 1, 0)

        x = x.unsqueeze(-1).unsqueeze(0)
        x = self.ST_GCN.extract_feature(x)
        x = x.squeeze(0)

        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(-1)
        return x

class MRA(nn.Module):
    def __init__(self):
        super(MRA, self).__init__()
        self.face_landmark = ST_GCN_Post()
        self.fc_1 = nn.Linear(160, 80)
        self.fc_2 = nn.Linear(80, 40)
        self.music_vec_1 = nn.Linear(10, 40)
        self.fc_3 = nn.Linear(40, 10)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x, music_vec):
        print("x:", x)
        # music_vec = x[-1]
        # x = x[0]

        x = self.face_landmark(x)
        x = self.fc_1(x)
        x = self.fc_2(x)

        music_vec = self.music_vec_1(music_vec)

        x = torch.cat((x, music_vec), 0)
        x = x.view(-1)
        x = self.fc_2(x)

        x = self.fc_3(x)
        x = self.soft(x)
        return x

# Example input tensor of shape (16, 3, 9, 1)
# input_tensor = result.unsqueeze(-1)

# # Ensure the input tensor is in the expected shape for Conv2d
# model = ST_GCN_Post()
# # output = model(input_tensor)
# output = model(input_test)
# # output = output.view(-1)
#
# # print("result", result)
# print("result", output)
# print("result shape", output.size())

def calculate_center_ity(after_playing_response):
    center_extents = []

    landmarks_dict = np.array(after_playing_response)
    print("landmarks_dict:", landmarks_dict)

    for row in range(len(landmarks_dict[0])):

        # draw triangle:
        a = list(landmarks_dict[0][row][:2])
        b = list(landmarks_dict[2][row][:2])
        c = list(landmarks_dict[6][row][:2])

        print("a, b, c:", a, b, c)

        pts = [a, b, c]
        pts = np.array(pts)
        # pts = pts.reshape((-1, 1, 2))
        # cv.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        print("pts:", pts)

        A = np.sum(pts, axis=0) / 3

        a = list(landmarks_dict[1][row][:2])
        b = list(landmarks_dict[3][row][:2])
        c = list(landmarks_dict[7][row][:2])

        pts = [a, b, c]
        pts = np.array(pts)
        # pts = pts.reshape((-1, 1, 2))
        # cv.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        B = np.sum(pts, axis=0) / 3

        a = list(landmarks_dict[4][row][:2])
        b = list(landmarks_dict[6][row][:2])
        c = list(landmarks_dict[8][row][:2])

        pts = [a, b, c]
        pts = np.array(pts)
        # pts = pts.reshape((-1, 1, 2))
        # cv.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        C = np.sum(pts, axis=0) / 3

        print("A, B, C:", A, B, C)

        M = (B + C) / 2
        BC = C - B
        AM = M - A
        BC_norm = BC / np.linalg.norm(BC)
        AM_norm = AM / np.linalg.norm(AM)
        dot_product = np.dot(BC_norm, AM_norm)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        print("angle_deg before:", angle_deg)

        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        print("angle_deg after:", angle_deg)

        extent_center = angle_deg / 90
        print("extent of center:", extent_center)
        center_extents.append(extent_center)

    X = center_extents
    X = np.expand_dims(X, axis=1)
    print(X.shape)

    # Apply Mean Shift clustering
    mean_shift = MeanShift().fit(X)
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_

    # Count the number of points in each cluster
    cluster_counts = Counter(labels)

    # Sort clusters by size and get the top two clusters
    top_two_clusters = [cluster for cluster, _ in cluster_counts.most_common(2)]
    top_two_centers = cluster_centers[top_two_clusters]

    print("Top two cluster centers:")
    print(top_two_centers)
    top_1 = top_two_centers[0][0]

    return top_1

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, next_state):
        # state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        # action = torch.tensor(action, dtype=torch.long)
        # reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        where = calculate_center_ity(next_state)

        # if len(state.shape) == 1:
        #     # (1, x)
        #     state = torch.unsqueeze(state, 0)
        #     next_state = torch.unsqueeze(next_state, 0)
        #     action = torch.unsqueeze(action, 0)
        #     reward = torch.unsqueeze(reward, 0)
        #     # done = (done,)

        # 1: predicted Q values with current state
        # pred = self.model(state)

        # # target = pred.clone()
        # # for idx in range(len(done)):
        # if True:
        #     # Q_new = reward
        #     # if not done[idx]:
        #     if True:
        #         # Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
        #         # Q_new = reward + self.gamma * torch.max(self.model(next_state))
        #
        #     # target[torch.argmax(action).item()] = Q_new
        #     # print("target:", target)

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        # loss = self.criterion(target, pred)
        loss = torch.tensor([[1 - where]], requires_grad=True)
        loss.backward()

        self.optimizer.step()