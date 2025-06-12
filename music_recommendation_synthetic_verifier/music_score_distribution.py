import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv

##### THE FOLLOWING CODE IS FOR RANDOM MUSIC PREFERENCE SELECTION (MUSIC PREFERENCE BEHAVIOUR) FOR THIS SYNTHETIC USER
def normalized_gaussian(g):
    g_min = np.min(g)
    g_max = np.max(g)

    return (g - g_min) / (g_max - g_min)

def gaussian(x, mu, sig, normalise=True):
    gaussian_temp = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)

    if normalise is True:
        return normalized_gaussian(gaussian_temp)
    return gaussian_temp

def get_music_pref_dist():
    music_length = 7
    random_music_idx = np.random.randint(music_length, size=1)[0]

    sig_random = np.random.uniform(2, 7, 1)[0]
    # sig_random_eye_plot = np.random.uniform(2, 7, 1)[0]

    print("random_music_idx:", random_music_idx)
    print("sig_random:", sig_random)

    x_values = np.linspace(-8, 8, 1000)
    user_music_preference_gaussian = gaussian(x_values, random_music_idx, sig_random)

    x_values_eye_plot = np.linspace(-8, 8, 1000)
    # static_eye_plot_gaussian = gaussian(x_values_eye_plot, 0, sig_random_eye_plot)

    block_size_user_music_pref = 1000 / 16
    user_music_pref_0 = lambda x: int((8 + x) * block_size_user_music_pref)

    random_music_pref_mapping = np.random.permutation(7)
    random_music_pref_mapping_dict = {i: val for i, val in enumerate(random_music_pref_mapping)}
    random_music_pref_mean_dict = {i: 0 for i, val in enumerate(random_music_pref_mapping)}

    user_music_prefs_blocks = [user_music_preference_gaussian[user_music_pref_0(i):user_music_pref_0(i + 1)] for i in range(7)]
    # user_music_prefs_blocks_idxs =[np.linspace(user_music_pref_0(i), user_music_pref_0(i + 1), len(user_music_prefs_blocks[i])) for i in range(7)]

    # for idx_temp, user_music_pref_block in enumerate(user_music_prefs_blocks):
    #     plt.plot(user_music_prefs_blocks_idxs[idx_temp], user_music_pref_block)

    for i in range(7):
        random_music_pref_mapping_dict[i] = user_music_prefs_blocks[random_music_pref_mapping_dict[i]]
        random_music_pref_mean_dict[i] = np.mean(random_music_pref_mapping_dict[i])
        # random_music_pref_mapping[i] = i
    
    return random_music_pref_mapping_dict, random_music_pref_mean_dict, random_music_idx, sig_random

# # print statements for printing the music preference distribution for each 
# print("random_music_pref_mapping_dict after:", random_music_pref_mapping_dict)
# print("random_music_pref_mean_dict after:", random_music_pref_mean_dict)

# # For plotting results
# plt.plot(x_values, user_music_preference_gaussian)
# plt.plot(x_values_eye_plot, static_eye_plot_gaussian)

# plt.show()
