s = """
What is the aim of this model?
The model will take the input of the genre of music that the user likes and then try to come with some custom music.

G1 : This model will take the motion sequence window as the input and give out a single frequency to be played and that
input will be given to the imitator model to give out the DTW score.
"""

# # # import tkinter
# # # from tkinter import *
# # #
# # # master = Tk()
# # #
# # # var1 = IntVar()
# # # Checkbutton(master, text='male', variable=var1).grid(row=0, sticky=W)
# # #
# # # var2 = IntVar()
# # # Checkbutton(master, text='female', variable=var2).grid(row=1, sticky=W)
# # #
# # # mainloop()
# #
# # # import numpy as np
# # # data = np.load('../Ballet/video00000_0_1461_0.npy')
# # #
# # # print(data)
# #
# # import librosa
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# # # def generate_spectrogram(audio_path):
# # #     y, sr = librosa.load(audio_path)
# # #     S = librosa.feature.melspectrogram(y=y, sr=sr)
# # #     S_DB = librosa.amplitude_to_db(S, ref=np.max)
# # #
# # #     plt.figure(figsize=(10, 4))
# # #     spectrogram = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
# # #
# # #     plt.colorbar(format='%+2.0f dB')
# # #     plt.title('Mel-frequency spectrogram')
# # #     plt.tight_layout()
# # #     plt.show()
# # #
# # # # Example usage
# # # generate_spectrogram('../Chaleya.mp3')
# #
# # import librosa
# # import librosa.display
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# #
# # def generate_spectrogram(audio_path):
# #     y, sr = librosa.load(audio_path)
# #     S = librosa.feature.melspectrogram(y=y, sr=sr)
# #     S_DB = librosa.amplitude_to_db(S, ref=np.max)
# #
# #     # print(type(S_DB))
# #     # print(S_DB.shape)
# #     # print(len(S_DB))
# #     # print(len(S_DB[0]))
# #     # print(S_DB[0])
# #
# #     # for row in S_DB:
# #     #     row_temp = []
# #     #
# #     #     for item in row:
# #     #         row_temp.append(item)
# #     #
# #     #     print(row_temp)
# #
# #     return S_DB, sr
# #
# #
# # def plot_spectrogram(S_DB, sr):
# #     plt.figure(figsize=(10, 4))
# #     plt.imshow(S_DB, aspect='auto', origin='lower', cmap='viridis', extent=[0, S_DB.shape[1] / sr, 0, sr / 2])
# #     plt.colorbar(format='%+2.0f dB')
# #     plt.title('Mel-frequency spectrogram')
# #     plt.xlabel('Time (s)')
# #     plt.ylabel('Frequency (Hz)')
# #     plt.tight_layout()
# #     plt.show()
# #
# #
# # # Example usage
# # audio_path = '../Chaleya.mp3'
# # S_DB, sr = generate_spectrogram(audio_path)
# # plot_spectrogram(S_DB, sr)
# #
# # plt.show()
#
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# import soundfile as sf
#
#
# def generate_spectrogram(audio_path):
#     y, sr = librosa.load(audio_path)
#     S = librosa.feature.melspectrogram(y=y, sr=sr)
#     S_DB = librosa.amplitude_to_db(S, ref=np.max)
#
#     return S_DB, sr
#
#
# def plot_spectrogram(S_DB, sr):
#     plt.figure(figsize=(10, 4))
#     plt.imshow(S_DB, aspect='auto', origin='lower', cmap='viridis', extent=[0, S_DB.shape[1] / sr, 0, sr / 2])
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Mel-frequency spectrogram')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Frequency (Hz)')
#     plt.tight_layout()
#     plt.show()
#
#
# def spectrogram_to_audio(S_DB, sr):
#     # Convert dB spectrogram to linear
#     S = librosa.db_to_amplitude(S_DB)
#
#     # Invert the mel spectrogram to a linear spectrogram
#     S_inv = librosa.feature.inverse.mel_to_stft(S, sr=sr)
#
#     # Use the Griffin-Lim algorithm to invert the spectrogram to an audio signal
#     y_inv = librosa.griffinlim(S_inv)
#
#     return y_inv
#
#
# def save_audio(y, sr, output_path):
#     sf.write(output_path, y, sr)
#
#
# # Example usage
# audio_path = '../Chaleya.mp3'
# output_path = 'reconstructed_audio.wav'
# S_DB, sr = generate_spectrogram(audio_path)
# # plot_spectrogram(S_DB, sr)
#
# # Convert spectrogram back to audio
# y_inv = spectrogram_to_audio(S_DB, sr)
#
# # Save the reconstructed audio
# save_audio(y_inv, sr, output_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

def get_device():
    # return torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    return torch.device('cpu')

############################################

# def batch_tokenize(self, batch, start_token, end_token):
#
#     def tokenize(sentence, start_token, end_token):
#         sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
#         if start_token:
#             sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
#         if end_token:
#             sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
#         for _ in range(len(sentence_word_indicies), self.max_sequence_length):
#             sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
#         return torch.tensor(sentence_word_indicies)
#
#     tokenized = []
#     for sentence_num in range(len(batch)):
#         tokenized.append(tokenize(batch[sentence_num], start_token, end_token))
#     tokenized = torch.stack(tokenized)
#     return tokenized.to(get_device())
class PositionalEncoding():
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        # even_i = torch.arange(0, self.d_model, 2).float()
        i = torch.arange(0, self.d_model, 1).float()

        # denominator = torch.pow(10000, even_i / self.d_model)
        denominator = torch.pow(10000, i / self.d_model)

        position = (torch.arange(self.max_sequence_length)
                    .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        # odd_PE = torch.cos(position / denominator)

        # stacked = torch.stack([even_PE, odd_PE], dim=2)
        stacked = torch.stack([even_PE], dim=2)
        # print(stacked)

        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        # print(PE)

        return PE

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)

    # print("v from scaled_dot_product:", v)

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
        # print("x:", x)

        if len(x.size()) == 4:
            x = x[0]

        # print("x_size:", x.size(), len(x.size()))

        batch_size, sequence_length, d_model = x.size()

        # print(batch_size, sequence_length, d_model)

        qkv = self.qkv_layer(x.to(torch.float32))
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)

        # print(qkv)

        q, k, v = qkv.chunk(3, dim=-1)

        # print(q, k, v)

        values, attention = scaled_dot_product(q, k, v, mask)

        # print(values, attention)
        # print(values.shape, attention.shape)

        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)

        # print(values)

        out = self.linear_layer(values)
        # out = values
        return out

# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
#         super(EncoderLayer, self).__init__()
#         self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
#         self.norm1 = LayerNormalization(parameters_shape=[d_model])
#         self.dropout1 = nn.Dropout(p=drop_prob)
#         self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
#         self.norm2 = LayerNormalization(parameters_shape=[d_model])
#         self.dropout2 = nn.Dropout(p=drop_prob)
#
#     def forward(self, x, self_attention_mask):
#         residual_x = x.clone()
#         x = self.attention(x, mask=self_attention_mask)
#         x = self.dropout1(x)
#         x = self.norm1(x + residual_x)
#         residual_x = x.clone()
#         x = self.ffn(x)
#         x = self.dropout2(x)
#         x = self.norm2(x + residual_x)
#         return x

class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

############################################

d_model = 3
num_heads = 1
max_sequence_length = 10
ffn_hidden = 16
drop_prob = 0.2
num_layers = 12

class EncoderLayer(nn.Module):
    def __init__(self, d_model, max_sequence_length, num_heads, drop_prob, ffn_hidden):
        super(EncoderLayer, self).__init__()
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        # pos_2 = torch.stack
        # pos = position_encoder.to(get_device())
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.pos = self.position_encoder.forward()
        self.layer_drop_1 = nn.Dropout(drop_prob)
        self.layer_norm_1 = LayerNormalization(parameters_shape=[d_model])
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_drop_2 = nn.Dropout(drop_prob)
        self.layer_norm_2 = LayerNormalization(parameters_shape=[d_model])

    def forward(self, img_arr, attention_mask):
        # dropout = nn.Dropout(p=0.1)
        # img_array = dropout(img_array + pos)
        # print("img_array from EncoderLayer:", img_array)
        # print("attention_mask from EncoderLayer:", attention_mask)
        # print("self.pos from EncoderLayer:", self.pos)

        img_array = attention_mask

        img_array = img_array + self.pos
        # print(img_array)
        # print(pos)

        img_array_2 = []
        img_array_2.append(torch.tensor(np.array(img_array.clone())))
        img_array_2 = torch.stack(img_array_2)
        img_array = img_array_2.clone()

        # print(img_array)

        mask = None
        img_array = self.mha.forward(img_array, mask)
        # print(img_array)

        img_array = self.layer_drop_1(img_array)
        img_array = self.layer_norm_1(img_array_2 + img_array)

        # print(img_array)
        img_array_2 = img_array.clone()

        img_array = self.ffn(img_array.to(torch.float32))
        img_array = self.layer_drop_2(img_array)
        img_array = self.layer_norm_2(img_array + img_array_2)

        # print(img_array)

        return img_array


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, max_sequence_length, num_heads, drop_prob, ffn_hidden):
        super().__init__()

        self.layers = SequentialEncoder(*[EncoderLayer(d_model, max_sequence_length, num_heads, drop_prob, ffn_hidden)
                                          for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x, x)

        return x

class Combiner(nn.Module):
    def __init__(self, input_dim, mid_output_dtw_1, mid_output_dtw_2, final_output):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, mid_output_dtw_1)
        self.linear2 = nn.Linear(mid_output_dtw_1, mid_output_dtw_2)
        self.linear3 = nn.Linear(mid_output_dtw_2, final_output)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # print("x from Combiner:", x)
        # print("music_params from Combiner:", music_params)

        # x = torch.cat((x, music_params), 0)
        # print("concat:", x)

        x = self.linear1(x.to(torch.float32))
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)

        # x = self.sigmoid(x)
        x = F.softmax(x, dim=-1)

        return x

class MUSIC1(nn.Module):
    def __init__(self, num_layers, d_model, max_sequence_length, num_heads,
                 drop_prob, ffn_hidden, mid_output_dtw_1, mid_output_dtw_2, final_output_dtw, input_dim):
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, max_sequence_length,
                               num_heads, drop_prob, ffn_hidden)
        self.linear = nn.Linear(d_model, mid_output_dtw_1)
        self.combiner = Combiner(input_dim, mid_output_dtw_1, mid_output_dtw_2, final_output_dtw)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.encoder(x)
        print("x after encoder: ", x)

        x = self.layer_norm(x)
        print("x after normalization: ", x)

        # x = self.linear(x.to(torch.float32))
        # print("x second:", x)
        # print("music_params insider:", music_params)

        x = torch.flatten(x)
        print("x after flatten: ", x)


        # # print("music_params from MUSIC1:", x, music_params)
        x = self.combiner(x)
        print("x after combiner: ", x)

        return x


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state):
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
            # done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        # for idx in range(len(done)):
        if True:
            # Q_new = reward
            # if not done[idx]:
            if True:
                # Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                Q_new = reward + self.gamma * torch.max(self.model(next_state))

            target[torch.argmax(action).item()] = Q_new
            print("target:", target)

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
# class QTrainer:
#     def __init__(self, model, lr, gamma):
#         self.lr = lr
#         self.gamma = gamma
#         self.model = model
#         self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
#         self.criterion = nn.MSELoss()
#
#     def train_step(self, state, action, reward):
#         state = torch.tensor(state)
#         action = torch.tensor(action)
#         reward = torch.tensor(reward)
#         # (n, x)
#
#         if len(state.shape) == 1:
#             # (1, x)
#             state = torch.unsqueeze(state, 0)
#             action = torch.unsqueeze(action, 0)
#             reward = torch.unsqueeze(reward, 0)
#
#         print("data from train_step:", state, action, reward)
#
#         # 1: predicted Q values with current state
#         pred = self.model(state)
#
#         target = pred.clone()
#         print("target in model_v1:", target)
#         # for idx in range(len(done)):
#         if True:
#             # Q_new = reward[idx]
#             Q_new = reward
#             print("Q_new, pred:", 1000 - Q_new, pred.item(), type(Q_new), type(pred))
#             # if not done[idx]:
#             #     Q_new = reward[idx] + self.gamma * torch.max(attention)
#
#             # target[0][torch.argmax(action[0]).item()] = Q_new
#             target = (1000 - float(Q_new)) + pred.item()
#             print("target before new in model_v1:", target)
#
#             target = 1 / (1 + np.exp(-target))
#             target = torch.tensor(target - 0.3)
#             print("target new in model_v1:", target)
#
#         for idx in range(len(done)):
#             Q_new = reward[idx]
#             if not done[idx]:
#                 Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
#
#             target[idx][torch.argmax(action[idx]).item()] = Q_new
#             print("target:", target)
#
#         # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
#         # pred.clone()
#         # preds[argmax(action)] = Q_new
#         self.optimizer.zero_grad()
#         print("done optimizer:", type(target), type(pred))
#
#         loss = self.criterion(target, pred)
#         print("done loss self.criterion")
#
#         # loss = 2000 - reward.item()
#         loss.backward()
#         print("done loss backward")
#
#         self.optimizer.step()
#         print("done optimizer")


# music1 = MUSIC1(num_layers, d_model, max_sequence_length, num_heads, drop_prob, ffn_hidden, 16, 8, 1, 30)
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
# print(matrix_test_full)
# print(matrix_test_full.size())
#
# # while True:
# if True:
#     # print("Enter frequency: ")
#     # freq = int(input())
#     # print("Enter BPM: ")
#     # bpm = int(input())
#
#     # music_test = [torch.tensor(freq), torch.tensor(bpm)]
#     # music_test = torch.stack(music_test)
#
#     DTW_pred = music1(matrix_test_full)
#
#     print(DTW_pred)