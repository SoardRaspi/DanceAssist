import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import GAT
import csv
import os
import numpy as np
import librosa

import matplotlib.pyplot as plt

from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

#  The model is trying to find the best beat parameters for the given motion sequence. The loss function is the DTW
#  distance. Finally, the model should give out the beat parameters when an input motion sequence is given.

#  What is the ultimate goal: The ultimate goal is to see where the user is having trouble to move. We play different
#  music and try to improve the movement.

filename = None


def print_function():
    print(filename)

def calc_correlation(show_motion, actual_motion):
    show_motion_t = np.array(show_motion)
    actual_motion_t = np.array(actual_motion).T

    # print("show_motion_t:", show_motion_t)
    # print("actual_motion_t:", actual_motion_t)

    # print(show_motion_t)
    # print(actual_motion_t)

    correlation_matrix = np.dot(show_motion_t, actual_motion_t)

    return correlation_matrix

# device = torch.device('cpu')
#
# path = "musics"
# audio_files = os.listdir(path)
#
# filename = "right_arm_data.csv"
# right_arm_sa = []
# right_arm_ea = []
# right_arm_wa = []
#
# with open(filename, 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#
#     for _, lines in enumerate(csvreader):
#         if _ > 0:
#             sa, ea, wa = lines
#             right_arm_sa.append(float(sa))
#             right_arm_ea.append(float(ea))
#             right_arm_wa.append(float(wa))
#
# # edges = np.array([[0, 1], [1, 2]])
# # adjacenct = [[0 for i in range(len(np.unique(edges.flatten())))] for ii in range(len(np.unique(edges.flatten())))]
# #
# # for edge in edges:
# #     a = edge[0]
# #     b = edge[1]
# #
# #     adjacenct[a][b] = 1
# #     adjacenct[b][a] = 1
# #
# # print(adjacenct)
#
# # # audio_file = path + "/" + audio_files[length_angles_motion_counter]
# # audio_file = path + "/" + audio_files[0]
# # y, sr = librosa.load(audio_file)
# # stft = librosa.stft(y)
# # stft_db = librosa.amplitude_to_db(np.abs(stft))
# #
# # class PositionalEncoding(nn.Module):
# #     def __init__(self, d_model, max_sequence_length):
# #         super().__init__()
# #         self.max_sequence_length = max_sequence_length
# #         self.d_model = d_model
# #
# #     def forward(self):
# #         even_i = torch.arange(0, self.d_model, 2).float()
# #         denominator = torch.pow(10000, even_i / self.d_model)
# #         position = (torch.arange(self.max_sequence_length)
# #                     .reshape(self.max_sequence_length, 1))
# #         even_PE = torch.sin(position / denominator)
# #         odd_PE = torch.cos(position / denominator)
# #         stacked = torch.stack([even_PE, odd_PE], dim=2)
# #         PE = torch.flatten(stacked, start_dim=1, end_dim=2)
# #         return PE
# #
# # features = []
# # args = []
# #
# # # gat_net = GAT(
# # #         in_features=features.shape[1],          # Number of input features per node
# # #         n_hidden=args.hidden_dim,               # Output size of the first Graph Attention Layer
# # #         n_heads=args.num_heads,                 # Number of attention heads in the first Graph Attention Layer
# # #         num_classes=labels.max().item() + 1,    # Number of classes to predict for each node
# # #         concat=args.concat_heads,               # Wether to concatinate attention heads
# # #         dropout=args.dropout_p,                 # Dropout rate
# # #         leaky_relu_slope=0.2                    # Alpha (slope) of the leaky relu activation
# # #     ).to(device)
# #
#
# window = 10
#
# temp = []
# for i in range(window):
#     sa_temp = np.ceil(((right_arm_sa[i] + 180) * 256) / 360)
#     ea_temp = np.ceil(((right_arm_ea[i] + 180) * 256) / 360)
#     wa_temp = np.ceil(((right_arm_wa[i] + 180) * 256) / 360)
#
#     t = [sa_temp, ea_temp, wa_temp]
#     temp.append(t)
#
# temp = np.array(temp)
#
# plt.imshow(temp, cmap='viridis', interpolation='nearest')
# plt.colorbar()  # Add a colorbar to show the scale
# plt.title('2D Array Visualization')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()

############################################
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x
############################################

# G1: The model will take the motion sequence and the beat parameters as input and the output will be the DTW distance
#     Finally, the model will be used to determine the final similarity between the shown and the actual motion sequence
#     for the given beat.

class Linear_QNet(nn.Module):
    def __init__(self, window_size, beat_params_size, beat_params, num_layers,
                 d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        # self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, output_size)

        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                          for _ in range(num_layers)])
        self.encoder_layer = EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)

    def forward(self, motion_seq):
        # x = F.relu(self.linear1(x))
        # x = self.linear2(x)

        x = self.encoder_layer(motion_seq)

        return x

    def save(self, file_name='model_DAUM1.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
