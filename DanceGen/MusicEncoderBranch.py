import cv2 as cv
import os
import mediapipe as mp
import librosa
from pydub import AudioSegment
import io
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
from torchsummary import summary

# filename = "DanceAssist DanceGen/gBR_sBM_c01_d04_mBR1_ch01.mp4"

def run_posenet():
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    main_face = [i for i in range(9)]
    main_pose = [12, 14, 16]

    cap = cv.VideoCapture(filename)

    while (cap.isOpened()):
        ret, frame = cap.read()
        cv.imshow('frame', frame)

        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # cv.imshow('imgRGB', frame)

        results = pose.process(frame)
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
                id_temp = int(id)

                # if id_temp in main_face:
                if id_temp in main_pose:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv.circle(frame, (cx, cy), 5, (255, 0, 0), cv.FILLED)

        cv.imshow('landmarks', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def spectogram_from_video(video_path):
    audio = AudioSegment.from_file(video_path)

    # Convert audio to raw audio data
    audio_bytes = io.BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    y, sr = librosa.load(audio_bytes, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    S_db = np.array(S_db)
    print(S_db.shape)

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-frequency spectrogram')
    # plt.show()

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)

    if mask is not None:
        scaled += mask

    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dimension = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3*d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, max_seq_length, d_model = x.sisze()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, max_seq_length, self.num_heads, 3 * self.head_dimension)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, max_seq_length, self.num_heads * self.head_dimension)
        out = self.linear_layer(values)

        return out

class LayerNorm(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean)**2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma*y + self.beta

        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, inputs):
        inputs = self.linear1(inputs)
        inputs = self.dropout(inputs)
        inputs = self.relu(inputs)
        inputs = self.linear2(inputs)

        return inputs

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm_1 = LayerNorm(parameters_shape=[d_model])
        self.drop_1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm_2 = LayerNorm(parameters_shape=[d_model])
        self.drop_2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x
        x = self.attention(x)
        x = self.drop_1(x)
        x += residual_x
        x = self.norm_1(x)
        residual_x = x

        x = self.ffn(x)
        x = self.drop_2(x)
        x += residual_x
        x = self.norm_2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, drop_prob, num_encoder_layers=3):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, num_heads, ffn_hidden, drop_prob)
                                      for i in range(num_encoder_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x

class ResNet_Inner(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, drop_prob, in_channels, out_channels, kernel_size,
                 num_encoder_layers=3, num_gelu_conv=2, num_transformer_encoder=1):
        super().__init__()
        self.num_gelu_conv = num_gelu_conv
        self.gelu_conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        )
        self.trans_encoder = nn.Sequential(*[TransformerEncoder(d_model, num_heads, ffn_hidden, drop_prob,
                                                                num_encoder_layers)
                                             for i in range(num_transformer_encoder)])
        self.drop_1 = nn.Dropout(p=0.2)

    def forward(self, x):
        residual_x = x

        for ii in range(self.num_gelu_conv):
            x = self.gelu_conv(x)
            x = self.gelu_conv(x)
        self.drop_1(x)

        x = self.trans_encoder(x)
        self.drop_1(x)

        x += residual_x

        return x

class ResNet_Outer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, drop_prob, in_channels, out_channels, kernel_size,
                 num_encoder_layers=3, num_gelu_conv=2, num_transformer_encoder=1, num_resnet_inner=3):
        super().__init__()
        self.conv_2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.resnet_inner = nn.Sequential(*[ResNet_Inner(d_model, num_heads, ffn_hidden, drop_prob, in_channels,
                                                         out_channels, kernel_size, num_encoder_layers,
                                                         num_gelu_conv, num_transformer_encoder)
                                            for i in range(num_resnet_inner)])

    def forward(self, x):
        x = self.conv_2d(x)
        x = self.resnet_inner(x)

        return x

class SpectogramEncoder(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, drop_prob, in_channels, out_channels, kernel_size,
                 num_encoder_layers=3, num_gelu_conv=2, num_transformer_encoder=1, num_resnet_inner=3,
                 num_resnet_outer=3):
        super().__init__()
        self.resnet_outer = nn.Sequential(*[ResNet_Outer(d_model, num_heads, ffn_hidden, drop_prob, in_channels,
                                                         out_channels, kernel_size, num_encoder_layers, num_gelu_conv,
                                                         num_transformer_encoder, num_resnet_inner)
                                            for i in range(num_resnet_outer)])
        self.conv_2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x):
        residual_x = x
        x = self.resnet_outer(x)

        x += residual_x
        x = self.conv_2d(x)

        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # return summary(model)

model_spectogram_branch = SpectogramEncoder(128, 8, 256, 0.2, 1, 16, 5)
num_params = count_parameters(model_spectogram_branch)
print(f'Total number of parameters: {num_params}')

# spectogram_from_video(filename)

