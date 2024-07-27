import torch
import random
import numpy as np
from collections import deque
# from game import SnakeGameAI, Direction, Point
# from . import model_v1
from Music_Relevance_Attention import MRA, QTrainer
# from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()

        self.d_model = 3
        self.num_heads = 1
        self.max_sequence_length = 60
        self.ffn_hidden = 16
        self.drop_prob = 0.2
        self.num_layers = 12

        # self.model = MUSIC1(11, 256, 3)
        # self.model = MUSIC1(self.num_layers, self.d_model, self.max_sequence_length, self.num_heads, self.drop_prob,
        #                     self.ffn_hidden, 16, 8, 10, 30)
        self.model = MRA()

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, motion_window):
        # head = game.snake[0]
        # point_l = Point(head.x - 20, head.y)
        # point_r = Point(head.x + 20, head.y)
        # point_u = Point(head.x, head.y - 20)
        # point_d = Point(head.x, head.y + 20)
        #
        # dir_l = game.direction == Direction.LEFT
        # dir_r = game.direction == Direction.RIGHT
        # dir_u = game.direction == Direction.UP
        # dir_d = game.direction == Direction.DOWN
        #
        # state = [
        #     # Danger straight
        #     (dir_r and game.is_collision(point_r)) or
        #     (dir_l and game.is_collision(point_l)) or
        #     (dir_u and game.is_collision(point_u)) or
        #     (dir_d and game.is_collision(point_d)),
        #
        #     # Danger right
        #     (dir_u and game.is_collision(point_r)) or
        #     (dir_d and game.is_collision(point_l)) or
        #     (dir_l and game.is_collision(point_u)) or
        #     (dir_r and game.is_collision(point_d)),
        #
        #     # Danger left
        #     (dir_d and game.is_collision(point_r)) or
        #     (dir_u and game.is_collision(point_l)) or
        #     (dir_r and game.is_collision(point_u)) or
        #     (dir_l and game.is_collision(point_d)),
        #
        #     # Move direction
        #     dir_l,
        #     dir_r,
        #     dir_u,
        #     dir_d,
        #
        #     # Food location
        #     game.food.x < game.head.x,  # food left
        #     game.food.x > game.head.x,  # food right
        #     game.food.y < game.head.y,  # food up
        #     game.food.y > game.head.y  # food down
        # ]
        #
        # return np.array(state, dtype=int)

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

        something = motion_window
        matrix_test_full = []

        for row in something:
            sa_temp = np.ceil(((row[0] + 180) * 256) / 360).astype(np.uint8)
            ea_temp = np.ceil(((row[1] + 180) * 256) / 360).astype(np.uint8)
            wa_temp = np.ceil(((row[2] + 180) * 256) / 360).astype(np.uint8)
            # denom_temp = np.sqrt((sa_temp ** 2) + (ea_temp ** 2) + (wa_temp ** 2))

            matrix_test_full.append(torch.tensor([sa_temp, ea_temp, wa_temp]))

        matrix_test_full = [torch.stack(matrix_test_full)]
        matrix_test_full = torch.stack(matrix_test_full)

        return matrix_test_full

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))  # popleft if MAX_MEMORY is reached

    # def train_long_memory(self):
    #     if len(self.memory) > BATCH_SIZE:
    #         mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
    #     else:
    #         mini_sample = self.memory
    #
    #     states, actions, rewards, next_states, dones = zip(*mini_sample)
    #     self.trainer.train_step(states, actions, rewards, next_states, dones)
    #     # for state, action, reward, nexrt_state, done in mini_sample:
    #     #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, next_state):
        print("reached the train_short_memory in music_agent:", next_state)
        # print("reached the train_short_memory in music_agent with get_state:", self.get_state(state), action, reward)
        self.trainer.train_step(next_state)

    # def get_action(self, state):
    #     # random moves: tradeoff exploration / exploitation
    #     self.epsilon = 80 - self.n_games
    #     final_move = [0, 0, 0]
    #     if random.randint(0, 200) < self.epsilon:
    #         move = random.randint(0, 2)
    #         final_move[move] = 1
    #     else:
    #         state0 = torch.tensor(state, dtype=torch.float)
    #         prediction = self.model(state0)
    #         move = torch.argmax(prediction).item()
    #         final_move[move] = 1
    #
    #     return final_move

    def get_music(self, state, music):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_music = []
        random_decider = random.randint(0, 200)

        print("get_music in music_agent:", self.epsilon, random_decider)

        if random_decider < self.epsilon:
            random_note = np.random.randint(10)
            final_music = random_note
            # final_music = [0 for ___ in range(9)]
            # final_music[random_note] = 1
            # final_music.append(1 / (1 + np.exp(-(2000 + random_decider))))
        else:
            # state0 = state
            # print("state0 in music_agent:", state, music)

            prediction = self.model(state, music)
            print("prediction in music_agent:", prediction)

            final_music = torch.argmax(prediction).item()
            # final_move[move] = 1

            # final_music.append(prediction.item())

        # print("final_music after deciding:", final_music)
        return final_music