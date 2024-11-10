import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2 as cv
from PIL import Image
import csv

import cubical

img = cv.imread("correlation_images_2/done_seq_1.png")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def get_device():
    # return torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    return torch.device('cpu')

############################################

def batch_tokenize(self, batch, start_token, end_token):

    def tokenize(sentence, start_token, end_token):
        sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
        if start_token:
            sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
        if end_token:
            sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
        for _ in range(len(sentence_word_indicies), self.max_sequence_length):
            sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
        return torch.tensor(sentence_word_indicies)

    tokenized = []
    for sentence_num in range(len(batch)):
        tokenized.append(tokenize(batch[sentence_num], start_token, end_token))
    tokenized = torch.stack(tokenized)
    return tokenized.to(get_device())
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

def prep_motion_window(img, window=None):
    img_array = np.asarray(img) / 256
    final = []

    if window:
        for row_i in range(window):
            final.append(torch.tensor(img_array[row_i], device=get_device()))
        final = torch.stack(final)
    else:
        for row in img_array:
            final.append(torch.tensor(row))
        final = torch.stack(final)

    return final.to(get_device())

# print(img_array)
img_array = prep_motion_window(img, window=10)

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
        img_array_2_temp = img_array.clone()
        img_array_2.append(torch.tensor(np.array(img_array_2_temp)))
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
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, music_params):
        # print("x from Combiner:", x)
        # print("music_params from Combiner:", music_params)

        x = torch.cat((x, music_params), 0)
        # print("concat:", x)

        x = self.linear1(x.to(torch.float32))
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)

        return x

class DAUM1(nn.Module):
    def __init__(self, num_layers, d_model, max_sequence_length, num_heads,
                 drop_prob, ffn_hidden, mid_output_dtw_1, mid_output_dtw_2, final_output_dtw, input_dim):
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, max_sequence_length,
                               num_heads, drop_prob, ffn_hidden)
        self.linear = nn.Linear(d_model, mid_output_dtw_1)
        self.combiner = Combiner(input_dim, mid_output_dtw_1, mid_output_dtw_2, final_output_dtw)

    def forward(self, x, music_params):
        x = self.encoder(x)
        # print("x after encoder: ", x)

        # x = self.linear(x.to(torch.float32))
        # print("x second:", x)
        # print("music_params insider:", music_params)

        x = torch.flatten(x)

        # # print("music_params from DAUM1:", x, music_params)
        x = self.combiner(x, music_params)

        return x


daum1 = DAUM1(num_layers, d_model, max_sequence_length, num_heads, drop_prob, ffn_hidden, 16, 8, 1, 32)
# print(img_array)

optim = torch.optim.Adam(daum1.parameters(), lr=1e-4)

class MotionDataset(Dataset):

    def __init__(self, motion_window, music_params, DTW_score):
        self.motion_window = motion_window
        self.music_params = music_params
        self.DTW_score = DTW_score

    def __len__(self):
        return len(self.music_params)

    def __getitem__(self, idx):
        return self.motion_window[idx], self.music_params[idx], self.DTW_score[idx]

motion_windows = []
music_params = []
DTW_scores = []

def prepare_dataset(location_motion_images, location_music_params, window_size=10):
    location_motion_images_shown = location_motion_images + "/shown_seq_1.png"
    location_motion_images_done = location_motion_images + "/done_seq_1.png"
    music_params = location_music_params + "/music_params_1.csv"

    music_params_arr = []

    with open(music_params, mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            music_params_arr.append(lines)
    music_params_arr = music_params_arr[1:]
    music_params = music_params_arr
    del music_params_arr

    done_img = Image.open(location_motion_images_done)
    shown_img = Image.open(location_motion_images_shown)

    done_img = np.asarray(done_img)
    shown_img = np.asarray(shown_img)
    img_len, img_width = done_img.shape

    input_data, output_data = [[], []], []

    if img_width == 3:
        num_windows = img_len // window_size
        if num_windows != 0:
            print("Seq length not perfect multiple of window size")

        print("num_windows:", num_windows)

        for i in range(num_windows):
            temp_done = done_img[i*window_size:(i+1)*window_size]
            temp_shown = shown_img[i*window_size:(i+1)*window_size]
            temp_music_params = music_params[i]

            input_data[0].append(temp_shown)
            input_data[1].append(temp_music_params)
            output_data.append(cubical.dtw_score(temp_done / 256, temp_shown / 256))

        motion_windows, music_params = input_data
        # print("motion_windows:", motion_windows)
        # print("music_params:", music_params)
        # print("output_data:", output_data)

        dataset_temp = MotionDataset(motion_windows, music_params, output_data)

    else:
        dataset_temp = None

    return dataset_temp


location_total = "correlation_images_3"
music_total = "correlation_images_3"
dataset = prepare_dataset(location_total, music_total)

frequency = 1000
bpm = 120
music_params = []

music_params.append(torch.tensor(frequency))
music_params.append(torch.tensor(bpm))
music_params = torch.stack(music_params)

train_loader = DataLoader(dataset, 1)
# iterator = iter(train_loader)

# print("dataset:", dataset)

# for batch_num, batch in enumerate(iterator):
#     print("batch num:", batch_num, batch)
#
#     if batch_num > 3:
#         break

result_1 = daum1(img_array, music_params)
# print(result_1)

#  TRAIN THE MODEL
def train():
    device = get_device()
    daum1.train()
    daum1.to(device)
    total_loss = 0
    num_epochs = 100
    running_loss = 0
    last_loss = 0

    # criterion = nn.CrossEntropyLoss(ignore_index=kannada_to_index[PADDING_TOKEN], reduction='none')
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        iterator = iter(train_loader)
        for batch_num, batch in enumerate(iterator):
            daum1.train()

            print("batch num:", batch_num)
            # print(batch)
            # for iii in range(len(batch)):
            #     print(batch[iii])

            seq_shown, music_params, DTW_actual = batch[0], batch[1], batch[2]

            # encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
            #     eng_batch, kn_batch)
            optim.zero_grad()
            # kn_predictions = daum1(eng_batch,
            #                              kn_batch,
            #                              encoder_self_attention_mask.to(device),
            #                              decoder_self_attention_mask.to(device),
            #                              decoder_cross_attention_mask.to(device),
            #                              enc_start_token=False,
            #                              enc_end_token=False,
            #                              dec_start_token=True,
            #                              dec_end_token=True)

            # print(seq_shown, music_params, DTW_actual)
            # print("shapes:", seq_shown.size(), len(music_params))

            music_params = [torch.tensor(int(item[0])) for item in music_params]
            music_params = torch.stack(music_params)

            # seq_done = seq_shown
            DTW_pred = daum1(seq_shown, music_params)
            # print(DTW_pred, DTW_actual)

            # print(seq_shown)

            loss = criterion(DTW_pred, DTW_actual).to(device)
            loss.backward()

            optim.step()
            running_loss += loss.item()

            if batch_num % 100 == 0:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(batch_num + 1, last_loss))
                # tb_x = epoch * len(training_loader) + batch_num + 1
                # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

            # print("DTW_pred:", DTW_pred)
            #
            # actual_DTW = cubical.dtw_score(seq_shown[0] / 256, seq_done[0] / 256)
            # print("actual DTW:", actual_DTW)

            # print(seq_shown, music_params, DTW_pred)
            # labels = daum1.decoder.sentence_embedding.batch_tokenize(kn_batch, start_token=False, end_token=True)
            # loss = criterion(
            #     kn_predictions.view(-1, kn_vocab_size).to(device),
            #     labels.view(-1).to(device)
            # ).to(device)
            # valid_indicies = torch.where(labels.view(-1) == kannada_to_index[PADDING_TOKEN], False, True)
            # loss = loss.sum() / valid_indicies.sum()
            # loss.backward()
            # optim.step()

            # # train_losses.append(loss.item())
            # if batch_num % 100 == 0:
            #     print(f"Iteration {batch_num} : {loss.item()}")
            #     print(f"English: {eng_batch[0]}")
            #     print(f"Kannada Translation: {kn_batch[0]}")
            #     kn_sentence_predicted = torch.argmax(kn_predictions[0], axis=1)
            #     predicted_sentence = ""
            #     for idx in kn_sentence_predicted:
            #         if idx == kannada_to_index[END_TOKEN]:
            #             break
            #         predicted_sentence += index_to_kannada[idx.item()]
            #     print(f"Kannada Prediction: {predicted_sentence}")
            #
            #     daum1.eval()
            #     kn_sentence = ("",)
            #     eng_sentence = ("should we go to the mall?",)
            #     for word_counter in range(max_sequence_length):
            #         encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
            #             eng_sentence, kn_sentence)
            #         predictions = daum1(eng_sentence,
            #                                   kn_sentence,
            #                                   encoder_self_attention_mask.to(device),
            #                                   decoder_self_attention_mask.to(device),
            #                                   decoder_cross_attention_mask.to(device),
            #                                   enc_start_token=False,
            #                                   enc_end_token=False,
            #                                   dec_start_token=True,
            #                                   dec_end_token=False)
            #         next_token_prob_distribution = predictions[0][word_counter]  # not actual probs
            #         next_token_index = torch.argmax(next_token_prob_distribution).item()
            #         next_token = index_to_kannada[next_token_index]
            #         kn_sentence = (kn_sentence[0] + next_token,)
            #         if next_token == END_TOKEN:
            #             break
            #
            #     print(f"Evaluation translation (should we go to the mall?) : {kn_sentence}")
            #     print("-------------------------------------------")

# train()
#
# torch.save(daum1.state_dict(), "correlation_images_3/DAUM1")

def use_model(motion_seq, music_params):
    ans = []

    # for i in range(len(motion_seq)):
    if True:
        motion = motion_seq
        music = music_params

        ans.append(daum1(motion, music))

    return ans

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
# # print(matrix_test_full)
# # print(matrix_test_full.size())
#
# while True:
#     print("Enter frequency: ")
#     freq = int(input())
#     print("Enter BPM: ")
#     bpm = int(input())
#
#     music_test = [torch.tensor(freq), torch.tensor(bpm)]
#     music_test = torch.stack(music_test)
#
#     DTW_pred = daum1(matrix_test_full, music_test)
#
#     print(DTW_pred)