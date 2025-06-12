import numpy as np
import pandas as pd
import pygame
import matplotlib.pyplot as plt
import cv2 as cv
import ast
import time

from music_recommendation_synthetic_verifier.music_score_distribution import get_music_pref_dist

random_music_pref_mapping_dict, random_music_pref_mean_dict, random_music_idx, sig_random = get_music_pref_dist()
random_music_pref_ring_radii = {}

assert random_music_pref_mapping_dict is not None

print(random_music_pref_mapping_dict)
print(random_music_pref_mean_dict)

chosen_music = max(random_music_pref_mean_dict, key=random_music_pref_mean_dict.get)

faces = pd.read_csv("/Users/soardr/PycharmProjects/ReinforcementLearningSnakeGame/music_recommendation_synthetic_verifier/face_images.csv")
len_dataset = faces.shape[0]
num_cols = faces.shape[1]

nose_points = faces["0"]
nose_points_coords = []

def find_distance_to_closest_value(gaussian_map, target_value):
    abs_diff = np.abs(gaussian_map - target_value)
    closest_index = np.unravel_index(np.argmin(abs_diff), gaussian_map.shape)

    center_y, center_x = np.array(gaussian_map.shape) // 2
    closest_y, closest_x = closest_index

    distance = np.sqrt((closest_x - center_x)**2 + (closest_y - center_y)**2)

    return distance, (closest_x, closest_y)

for str_coords in nose_points:
    temp = ast.literal_eval(str_coords)
    nose_points_coords.append(temp[:2])

# GETTING THE MEAN OF THE POSITIONS
nose_points_mean_x = np.mean([x for x, _ in nose_points_coords])
nose_points_mean_y = np.mean([y for _, y in nose_points_coords])

# select random_start_point
nose_plotting_point = np.random.randint(len_dataset, size=1)[0]
print(f"{nose_plotting_point} : {nose_points_coords[nose_plotting_point]}")

# USER PROPERTY TO REGULATE THE DELAY BETWEEN HEAD MOVES
head_mov_delay_time = 2.0

def generate_2d_gaussian(kernel_size=5, sigma=1, muu=0):
    # k = kernel_size // 2
    k = 1

    x, y = np.meshgrid(np.linspace(-8, 8, kernel_size),
                       np.linspace(-8, 8, kernel_size))
    
    d = np.sqrt((x - muu)**2 + (y - muu)**2)
    gaussian = np.exp(-(d**2 / (2.0 * sigma**2)))
    gaussian_normalized = gaussian / np.sum(gaussian)  # Optional normalization

    return gaussian_normalized

kernel_size = 1000

sigma = 1.0
sigma = sig_random

physical_music_pref_gaussian_2d_map = generate_2d_gaussian(kernel_size=kernel_size, sigma=sigma)
physical_music_pref_gaussian_2d_map = (physical_music_pref_gaussian_2d_map - np.min(physical_music_pref_gaussian_2d_map)) / (np.max(physical_music_pref_gaussian_2d_map) - np.min(physical_music_pref_gaussian_2d_map))

# get the rings radii from here
for key in random_music_pref_mapping_dict:
    temp_range = random_music_pref_mapping_dict[key]
    min_radius, min_coords = find_distance_to_closest_value(physical_music_pref_gaussian_2d_map, min(temp_range))
    max_radius, max_coords = find_distance_to_closest_value(physical_music_pref_gaussian_2d_map, max(temp_range))

    random_music_pref_ring_radii[key] = [[min_radius, max_radius], [min_coords, max_coords]]

# print("random_music_pref_ring_radii:", random_music_pref_ring_radii)

physical_music_pref_gaussian_2d_map = (physical_music_pref_gaussian_2d_map * 255).astype(np.uint8)

# print("shapes:", physical_music_pref_gaussian_2d_map.shape)
 
# physical_music_pref_gaussian_2d_map_heatmap = (physical_music_pref_gaussian_2d_map * 255).astype(np.uint8)
physical_music_pref_gaussian_2d_map_heatmap = np.dstack((physical_music_pref_gaussian_2d_map, np.zeros_like(physical_music_pref_gaussian_2d_map), np.zeros_like(physical_music_pref_gaussian_2d_map)))
# physical_music_pref_gaussian_2d_map_heatmap = cv.applyColorMap(physical_music_pref_gaussian_2d_map, cv.COLORMAP_JET)

# print("shapes:", physical_music_pref_gaussian_2d_map.shape, physical_music_pref_gaussian_2d_map_heatmap.shape)

def tester_alone():
    cap = cv.VideoCapture(0)

    nose_point_start_time = None
    nose_point_start_flag = False

    while True:
        success, img = cap.read()

        if (success) and (img is not None):
            if nose_point_start_flag is False:
                nose_point_start_flag = True
                nose_point_start_time = time.time()

            image_x = int(img.shape[1] / 2)
            image_y = int(img.shape[0] / 2)

            gh, gw, _ = physical_music_pref_gaussian_2d_map_heatmap.shape

            top = image_y - gh // 2
            left = image_x - gw // 2

            top = max(0, top)
            left = max(0, left)
            bottom = top + gh
            right = left + gw

            # Ensure overlay doesn't exceed frame size
            if bottom > img.shape[0] or right > img.shape[1]:
                continue

            roi = img[top:bottom, left:right]

            # print("shapes for gaussian overlay:", roi.shape, physical_music_pref_gaussian_2d_map_heatmap.shape, img.shape)

            # Blend heatmap over region of interest
            blended = cv.addWeighted(roi, 0.5, physical_music_pref_gaussian_2d_map_heatmap, 0.5, 0)
            img[top:bottom, left:right] = blended

            for key in random_music_pref_ring_radii:
                min_radius_temp, max_radius_temp = random_music_pref_ring_radii[key][0]
                
                # cv.circle(img, (image_x, image_y), int(min_radius_temp), (0, 0, 255), int(max_radius_temp - min_radius_temp))
                overlay = img.copy()
                cv.circle(overlay, (image_x, image_y), int(min_radius_temp), (0, 0, 255), int(max_radius_temp - min_radius_temp))
                alpha = 0.4
                cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # # DISPLAY ALL THE POINTS IN THE FACES DATASET ON THE FRAME
            # for x, y in nose_points_coords:
            #     x *= img.shape[1]
            #     x = int(x)

            #     y *= img.shape[0]
            #     y = int(y)

            #     cv.circle(img, (x, y), 3, (255, 0, 0), 3)

            # # CODE FOR RANDOMLY SELECTING THE NEXT NOSE POINT
            # x, y = nose_points_coords[nose_plotting_point]

            # x *= img.shape[1]
            # x = int(x)

            # y *= img.shape[0]
            # y = int(y)

            # cv.circle(img, (x, y), 3, (255, 0, 0), 3)

            # CODE FOR PLOTTING ALL THE 9 POINTS FOR THE FACE
            keypoints_9_row = faces.iloc[nose_plotting_point]
            keypoints_9_row = keypoints_9_row[1:]
            keypoints_9_row = keypoints_9_row.to_dict()

            # print("type of keypoints_9_row:", type(keypoints_9_row))

            for key in keypoints_9_row:
                x, y = ast.literal_eval(keypoints_9_row[key])[:2]

                x *= img.shape[1]
                x = int(x)

                y *= img.shape[0]
                y = int(y)

                keypoints_9_row[key] = (x, y)

                # cv.circle(img, (x, y), 3, (128, 128, 128), 3)
                cv.circle(img, (x, y), 3, (255, 255, 255), 3)
            
            line_connecting_color = (128, 128, 128)
            cv.line(img, keypoints_9_row["7"], keypoints_9_row["3"], line_connecting_color, 3)
            cv.line(img, keypoints_9_row["3"], keypoints_9_row["2"], line_connecting_color, 3)
            cv.line(img, keypoints_9_row["2"], keypoints_9_row["1"], line_connecting_color, 3)

            cv.line(img, keypoints_9_row["1"], keypoints_9_row["0"], line_connecting_color, 3)

            cv.line(img, keypoints_9_row["8"], keypoints_9_row["6"], line_connecting_color, 3)
            cv.line(img, keypoints_9_row["6"], keypoints_9_row["5"], line_connecting_color, 3)
            cv.line(img, keypoints_9_row["5"], keypoints_9_row["4"], line_connecting_color, 3)

            cv.line(img, keypoints_9_row["4"], keypoints_9_row["0"], line_connecting_color, 3)

            # PLOTTING THE MEAN NOSE POINT
            x_nose_mean = int(nose_points_mean_x * img.shape[1])
            y_nose_mean = int(nose_points_mean_y * img.shape[0])
            cv.circle(img, (x_nose_mean, y_nose_mean), 3, (0, 0, 255), 3)

            cv.circle(img, (image_x, image_y), 3, (0, 255, 0), 3)
            
            cv.imshow("img", img)

            if time.time() - nose_point_start_time >= head_mov_delay_time:
                nose_plotting_point = np.random.randint(len_dataset, size=1)[0]
                nose_point_start_flag = False

            if False or (cv.waitKey(1) & 0xFF == ord('q')):
                break

    cap.release()
    cv.destroyAllWindows()

    # print("physical_music_pref_gaussian_2d_map_heatmap:", physical_music_pref_gaussian_2d_map_heatmap)
    # print("physical_music_pref_gaussian_2d_map:", physical_music_pref_gaussian_2d_map)

    # # Display it using matplotlib
    # plt.imshow(physical_music_pref_gaussian_2d_map, interpolation='nearest', cmap='viridis')
    # plt.title(f"2D Gaussian Distribution ({kernel_size}x{kernel_size})")
    # plt.colorbar(label='Intensity')
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.show()


# def tester_call_function(xf, yf):
def tester_call_function(music_index__s0: int):
    # replace this with the index chosen from the distribution
    nose_plotting_point = np.random.randint(len_dataset, size=1)[0]

    # image_x = int(img.shape[1] / 2)
    # image_y = int(img.shape[0] / 2)

    # gh, gw, _ = physical_music_pref_gaussian_2d_map_heatmap.shape

    # top = image_y - gh // 2
    # left = image_x - gw // 2

    # top = max(0, top)
    # left = max(0, left)
    # bottom = top + gh
    # right = left + gw

    # # Ensure overlay doesn't exceed frame size
    # if bottom > img.shape[0] or right > img.shape[1]:
    #     continue

    # roi = img[top:bottom, left:right]

    # # print("shapes for gaussian overlay:", roi.shape, physical_music_pref_gaussian_2d_map_heatmap.shape, img.shape)

    # # Blend heatmap over region of interest
    # blended = cv.addWeighted(roi, 0.5, physical_music_pref_gaussian_2d_map_heatmap, 0.5, 0)
    # img[top:bottom, left:right] = blended

    # for key in random_music_pref_ring_radii:
    #     min_radius_temp, max_radius_temp = random_music_pref_ring_radii[key][0]
        
    #     # cv.circle(img, (image_x, image_y), int(min_radius_temp), (0, 0, 255), int(max_radius_temp - min_radius_temp))
    #     overlay = img.copy()
    #     cv.circle(overlay, (image_x, image_y), int(min_radius_temp), (0, 0, 255), int(max_radius_temp - min_radius_temp))
    #     alpha = 0.4
    #     cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # # DISPLAY ALL THE POINTS IN THE FACES DATASET ON THE FRAME
    # for x, y in nose_points_coords:
    #     x *= img.shape[1]
    #     x = int(x)

    #     y *= img.shape[0]
    #     y = int(y)

    #     cv.circle(img, (x, y), 3, (255, 0, 0), 3)

    # # CODE FOR RANDOMLY SELECTING THE NEXT NOSE POINT
    # x, y = nose_points_coords[nose_plotting_point]

    # x *= img.shape[1]
    # x = int(x)

    # y *= img.shape[0]
    # y = int(y)

    # cv.circle(img, (x, y), 3, (255, 0, 0), 3)

    # CODE FOR PLOTTING ALL THE 9 POINTS FOR THE FACE
    keypoints_9_row = faces.iloc[nose_plotting_point]
    keypoints_9_row = keypoints_9_row[1:]
    keypoints_9_row = keypoints_9_row.to_dict()

    # print("type of keypoints_9_row:", type(keypoints_9_row))

    for key in keypoints_9_row:
        x, y, z, vis = ast.literal_eval(keypoints_9_row[key])

        # x *= xf
        # x = int(x)

        # y *= yf
        # y = int(y)

        keypoints_9_row[key] = (x, y, z, vis)
    
    return keypoints_9_row

def tester_call_function__reward(music_index__s0: int):
    return chosen_music, random_music_pref_mean_dict[music_index__s0]

def tester_get_chosen_music():
    return chosen_music

# tester_alone()