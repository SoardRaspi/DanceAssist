import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from st_gcn_st_gcn.net.st_gcn import ST_GCN_Model, ST_GCN_Model_Light

from DanceGenerator.FaceKps_EyeCroppedNet.eye_mode_1 import eye_mode_1

class actor_model_RU(nn.Module):
    def __init__(self, head_mov_args, eye_proc_args, eye_proc_hidden_dim, common_hidden_dim, 
                 prev_iter_music_shape, block_size, eye_mov_mode=0):
        super(actor_model_RU, self).__init__()

        self.RU = TheRU(head_mov_args, eye_proc_args, eye_proc_hidden_dim, common_hidden_dim,
                        prev_iter_music_shape, eye_mov_mode=eye_mov_mode)
        
        self.block_size = block_size
        self.common_hidden_dim = common_hidden_dim

        self.eye_mov_mode = eye_mov_mode
    
    def forward(self, head_mov_x, music_prev_x, ht__0=None, eye_crop_x=None):
        '''
        head_mov_x: input of the head movement data in graph format
        eye_crop_x: input of the eye-cropped data data in form of frames
        music_prev_x: input of the music recommendations in from the previous iteration

        head_mov_x: (1, 4, 60, 9, 1) -> (batch size, num features, time steps, num node points, num persons)
        eye_crop_x: (1, 60, (<height>, <width>)) -> (batch size, time steps, (frame height, frame width))
        music_prev_x: mode 1 => (1, 60, 7) -> (batch size, time steps, music recom probs)
        '''

        print("head_mov_x data in actor_model_RU:", type(head_mov_x), head_mov_x.shape, head_mov_x[0].shape)
        print("music_prev_x data in actor_model_RU:", type(music_prev_x), music_prev_x.shape, music_prev_x[0].shape)

        headmov_batch_size, headmov_num_feats, headmov_steps, headmov_num_nodes, headmov_num_persons = head_mov_x.shape

        if self.eye_mov_mode != 0:
            assert eye_crop_x is not None
            eyecrop_batch_size, eyecrop_steps, eye_frame_height, eye_frame_width = eye_crop_x.shape
        
        music_batch_size, music_steps, music_vec_len = music_prev_x.shape

        if self.eye_mov_mode != 0:
            assert headmov_steps == eyecrop_steps

        num_blocks = np.ceil(headmov_steps / self.block_size)
        expected_length = num_blocks * self.block_size

        print("some inner data:", num_blocks, expected_length)

        ht_1s = []
        music_rec_probs_batch = []

        print("headmov_batch_size:", headmov_batch_size)

        for index_in_batch in range(headmov_batch_size):
            music_rec_probs = []
            # print("index_in_batch:", index_in_batch)

            if ht__0 is None:
                # ht_1 = np.zeros((self.common_hidden_dim,))
                ht_1 = torch.zeros(self.common_hidden_dim)
            else:
                ht_1 = ht__0

            for l in range(0, headmov_steps, self.block_size):
                # print("inside the for loop in forward of actor_model_RU:", index_in_batch)

                # xt__head_mov = head_mov_x[:, :, l:(l + self.block_size), :, :]
                xt__head_mov = head_mov_x[index_in_batch, :, l:(l + self.block_size), :, :]
                # print("shape of xt__head_mov:", xt__head_mov.shape)

                # xt__eye_crop = None
                # if self.eye_mov_mode != 0:
                #     xt__eye_crop = eye_crop_x[:, l:(l + self.block_size), :, :]
                xt__eye_crop = None
                if self.eye_mov_mode != 0:
                    xt__eye_crop = eye_crop_x[index_in_batch, l:(l + self.block_size), :, :]
                
                # if headmov_steps == music_steps:
                #     xt__music = music_prev_x[:, l:(l + self.block_size), :]
                # elif music_steps == num_blocks:
                #     xt__music = music_prev_x[:, l // self.block_size, :]
                if headmov_steps == music_steps:
                    xt__music = music_prev_x[index_in_batch, l:(l + self.block_size), :]
                elif music_steps == num_blocks:
                    xt__music = music_prev_x[index_in_batch, l // self.block_size, :]
                # print("shape of xt__music:", xt__music.shape)
                
                # if the eye branch is activated...
                if self.eye_mov_mode != 0:
                    # processing the eye crop data
                    eye_processed = []
                    xt__eye_crop.squeeze(0)

                    for eye_frame in xt__eye_crop:
                        right_eye_temp, left_eye_temp = eye_mode_1(eye_frame, )
                
                try:
                    yt, ht = self.RU(xt__head_mov, xt__music, ht_1, eye_crop_x=xt__eye_crop)
                except Exception as e__RU_inference:
                    print("error in using RU in actor_model_RU:", e__RU_inference)
                
                music_rec_probs.append(torch.squeeze(yt, dim=0))

                ht_1 = ht
            
            try:
                ht_1s.append(ht_1)
                music_rec_probs_batch.append(torch.stack(music_rec_probs, dim=0))
            except Exception as e__main_model_stacking_error:
                print("error in stacking the torch tensor:", e__main_model_stacking_error)

        # return music_rec_probs, ht_1
        # print("music_rec_probs_batch data:", music_rec_probs_batch, type(music_rec_probs_batch))

        print("ht_1s:", len(ht_1s), ht_1s[0].shape)
        print("music_rec_probs_batch:", len(music_rec_probs_batch), music_rec_probs_batch[0].shape)

        ht_1s = torch.stack(ht_1s, dim=0)
        music_rec_probs_batch = torch.stack(music_rec_probs_batch, dim=0)

        ht_1s = torch.squeeze(ht_1s, dim=1)

        print("shapes of returning tensors from actor_model_RU:", ht_1s.shape, music_rec_probs_batch.shape)

        return music_rec_probs_batch, ht_1s

class TheRU(nn.Module):
    '''
    head_mov_args: args for the Light ST-GCN to capture features from the head-movement data
    eye_mov_mode: 0 -> to disable the branch
                  1 -> to consider just the values of the position of the iris in the eye, will require some iris processing using MediaPipe
                  2 -> to consider the whole eye-cropped frame with affine transformation as pre-processing and then feature extraction (too heavy and unstable for DDPG)
                  3 -> to consider only the eye input
    prev_iter_music_one_hot: one-hot encoded vector of the music selected by the block in the previous iteration
    '''
    def __init__(self, head_mov_args, eye_proc_args, eye_proc_hidden_dim, common_hidden_dim, prev_iter_music_shape, eye_mov_mode=0):
        super(TheRU, self).__init__()

        self.eye_move_mode = eye_mov_mode
        self.head_mov_feats = ST_GCN_Model_Light(**head_mov_args)

        if eye_mov_mode == 1:
            self.eye_GRU_mode_1 = nn.GRU(**eye_proc_args[1])
        
        if eye_mov_mode != 0:
            self.fc = nn.Linear(eye_proc_hidden_dim, common_hidden_dim)

        if eye_mov_mode != 0:
            self.fc_comb = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
            )
        else:
            self.fc_comb = nn.Linear(32, 16)

        self.fc_comb2_1 = nn.Linear(32, 16)
        self.fc_comb2_2 = nn.Linear(16, 8)

        # self.prev_iter_music_probs = prev_iter_music_probs
        self.prev_iter_music_shape = prev_iter_music_shape

        self.music_encode = nn.Sequential(
            nn.Linear(prev_iter_music_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.music_comb1 = nn.Linear(16, 8)
        self.music_comb2 = nn.Linear(8, prev_iter_music_shape)
    
    def forward(self, head_mov_x, music_prev_x, ht_1, eye_crop_x=None):
        '''
        x: 3 items -> [graph data of head movements, 
                       frames for eye processing,
                       music from previous iteration by the same block]
        '''

        # ST_GCN_Light_out = self.head_mov_feats(x[0])  # 32
        # eye_based_output = self.fc(self.eye_GRU_mode_1(x[1]))  # 32

        # STGCN_eye_concat = torch.cat((ST_GCN_Light_out, eye_based_output), axis=0)  # 64
        # x = self.fc_comb(STGCN_eye_concat)

        # x__ht_1 = torch.cat((x, ht_1))  # 32

        # ht = nn.ReLU()(self.fc_comb2_1(x__ht_1))
        # x = nn.ReLU()(self.fc_comb2_2(ht))

        # music_encoded__t_1 = self.music_encode(self.prev_iter_music_probs)

        # x = torch.concat((x, music_encoded__t_1))  # 16
        # x = self.music_comb1(x)
        # yt = self.music_comb2(x)

        # yt = nn.Sigmoid()(yt)


        # using actual inputs.....
        head_mov_x = torch.unsqueeze(head_mov_x, dim=0)
        music_prev_x = torch.unsqueeze(music_prev_x, dim=0)
        ht_1 = torch.unsqueeze(ht_1, dim=0)

        # print("head_mov_x shape:", head_mov_x.shape)
        # print("music_prev_x shape:", music_prev_x.shape)
        # print("ht_1 shape:", ht_1.shape)

        ST_GCN_Light_out = self.head_mov_feats(head_mov_x)
        if self.eye_move_mode != 0:
            assert eye_crop_x is not None

            eye_based_output = self.fc(eye_crop_x)
            STGCN_eye_concat = torch.cat((ST_GCN_Light_out, eye_based_output), axis=0)

            x = self.fc_comb(STGCN_eye_concat)
        
        else:
            x = self.fc_comb(ST_GCN_Light_out)
        
        # print("ST_GCN_Light_out: output in actor.py", ST_GCN_Light_out.shape)
        
        # print("shapes before before:", x.shape, ht_1.shape)

        if x.shape != ht_1.shape:
            x = x.view(-1)
        ht_1 = torch.tensor(ht_1)
        
        # print("shapes before 1:", x.shape, ht_1.shape)
        x = torch.squeeze(x)
        ht_1 = torch.squeeze(ht_1)
        # print("shapes before 2:", x.shape, ht_1.shape)

        x__ht_1 = torch.cat((x, ht_1))
        x__ht_1 = torch.unsqueeze(x__ht_1, 0)
        x__ht_1 = x__ht_1.float()
        # print("x__ht_1:", x__ht_1.shape)

        try:
            x__ht_1 = self.fc_comb2_1(x__ht_1)
            # print("x__ht_1 after:", x__ht_1.shape)
        except Exception as eee:
            print("eee:", eee)

        ht = nn.ReLU()(x__ht_1)
        # print("ht:", ht.shape)

        x = nn.ReLU()(self.fc_comb2_2(ht))
        # print("x:", x.shape)

        music_prev_x = torch.tensor(music_prev_x, dtype=torch.float32)
        # print("music_prev_x shape before:", music_prev_x.shape)
        music_encoded__t_1 = self.music_encode(music_prev_x)
        # print("music_encoded__t_1:", music_encoded__t_1.shape)

        x = torch.cat((x, music_encoded__t_1), dim=-1).float()  # 16
        # print("x:", x.shape)

        x = self.music_comb1(x)
        yt = self.music_comb2(x)

        yt = nn.Sigmoid()(yt)

        # print("yt and ht:", yt.shape, ht.shape)

        return yt, ht

