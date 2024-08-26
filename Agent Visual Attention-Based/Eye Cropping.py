# import cv2 as cv
# import mediapipe as mp
# import numpy as np
# import time
# import math
# from scipy.spatial import ConvexHull
#
# header = ["sno",
#           "re_1", "re_2", "re_3", "re_4", "re_5", "re_6", "re_7", "re_8",
#           "ar_re_1", "ar_re_2", "ar_re_3", "ar_re_4", "ar_re_5", "ar_re_6", "ar_re_7", "ar_re_8",
#           "rea",
#           "le_1", "le_2", "le_3", "le_4", "le_5", "le_6", "le_7", "le_8",
#           "ar_le_1", "ar_le_2", "ar_le_3", "ar_le_4", "ar_le_5", "ar_le_6", "ar_le_7", "ar_le_8",
#           "lea",
#           "x", "y"]
#
# # model_left_eye = load_model('cnn_model_1_left_eye.h5')
# filename_eye_data = "eye_data_percents_2.csv"
#
# rows = []
#
# # Task1: get the eye point
# # Task2: see if the eye points have enough resolution..., if not, zoom in, do resolution
# # Task3: calculate position based on this data
#
# # OR
#
# # Task2: just tell if the person is looking at the instructor or not // first this: ...
#
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils
#
# mp_face_mesh = mp.solutions.face_mesh
#
# flag_up = False
# time_up_start = None
# pTime = 0
#
# cap = cv.VideoCapture(0)
#
# def calculate_angle(p1, p2, p3):
#     # ang1 = np.arctan2(*p1[::-1])
#     # ang2 = np.arctan2(*p2[::-1])
#     # return np.rad2deg((ang1 - ang2) % (2 * np.pi))
#
#     # Calculate the vectors between the points
#     vector1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
#     vector2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
#
#     # Calculate the dot product
#     dot_product = np.dot(vector1, vector2)
#
#     # Calculate the magnitudes of the vectors
#     magnitude1 = np.linalg.norm(vector1)
#     magnitude2 = np.linalg.norm(vector2)
#
#     # Calculate the angle in radians
#     angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
#
#     # Convert radians to degrees
#     angle_deg = np.degrees(angle_rad)
#
#     if p3[0] < p1[0]:
#         return -angle_deg
#
#     return angle_deg
#
# motion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# # count = 1
#
# # right_eye_main = []
# # right_eye_main_dict = {33: None, 246: None, 161: None, 160: None, 159: None, 158: None, 157: None,
# #                        173: None, 133: None, 155: None, 154: None, 153: None, 145: None, 144: None,
# #                        163: None, 7: None}
# r_lobe_coords = {130: None, 247: None, 30: None, 29: None, 27: None, 28: None, 56: None, 190: None, 243: None, 25: None,
#                  110: None, 24: None, 23: None, 22: None, 26: None, 112: None}
# regions_re = {1: [33, 246, 7], 2: [246, 7, 163, 161], 3: [161, 163, 144, 160], 4: [160, 144, 145, 159],
#               5: [159, 145, 153, 158], 6: [158, 153, 154, 157], 7: [157, 154, 155, 173], 8: [173, 155, 133]}
# # right_upper_lobe = {1: [247, 246, 33, 130], 2: [247, 30, 161, 246], 3: [30, 29, 160, 161], 4: [29, 27, 159, 160],
# #                     5: [27, 28, 158, 159], 6: [28, 56, 157, 158], 7: [56, 190, 173, 157], 8: [190, 243, 133, 173]}
# # right_lower_lobe = {1: [130, 33, 7, 25], 2: [7, 163, 110, 25], 3: [163, 144, 24, 110], 4: [144, 145, 23, 24],
# #                     5: [145, 153, 22, 23], 6: [153, 154, 26, 22], 7: [154, 155, 112, 26], 8: [155, 133, 243, 112]}
#
# # left_eye_main = []
# # left_eye_main_dict = {263: None, 466: None, 388: None, 387: None, 386: None, 385: None, 384: None,
# #                       398: None, 362: None, 382: None, 381: None, 380: None, 374: None, 373: None,
# #                       390: None, 249: None}
# l_lobe_coords = {463: None, 414: None, 286: None, 258: None, 257: None, 259: None, 260: None, 467: None, 359: None,
#                  255: None, 339: None, 254: None, 253: None, 252: None, 256: None, 341: None}
# regions_le = {1: [263, 466, 249], 2: [466, 249, 390, 388], 3: [388, 390, 373, 387], 4: [387, 373, 374, 386],
#               5: [386, 374, 380, 385], 6: [385, 380, 381, 384], 7: [384, 381, 382, 398], 8: [398, 382, 362]}
# # left_upper_lobe = {1: [359, 263, 466, 467], 2: [467, 466, 388, 260], 3: [260, 388, 387, 259], 4: [259, 387, 386, 257],
# #                    5: [257, 386, 385, 258], 6: [258, 385, 384, 286], 7: [286, 384, 398, 414], 8: [414, 398, 362, 463]}
# # left_lower_lobe = {1: [359, 263, 249, 255], 2: [255, 249, 390, 339], 3: [339, 390, 373, 254], 4: [254, 373, 374, 253],
# #                    5: [253, 374, 380, 252], 6: [252, 380, 381, 256], 7: [256, 381, 382, 341], 8: [341, 382, 362, 463]}
#
# def angle_of_inclination(p1, p2):
#     (x1, y1) = p1
#     (x2, y2) = p2
#
#     delta_x = x2 - x1
#     delta_y = y2 - y1
#     angle_rad = math.atan2(delta_y, delta_x)
#     angle_deg = math.degrees(angle_rad)
#     return angle_deg
#
# # Task1:
#
# drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
#
# count_temp = 1
# last = [None, None]
#
# time_overall = time.time()
#
# regions_R = {'ul': [200, 480], 'lr': [600, 640]}
# regions_G = {'ul': [200, 320], 'lr': [600, 480]}
# regions_B = {'ul': [200, 160], 'lr': [600, 320]}
#
# iter_count = 1
#
#
# def find_rectangle_corners(center_points):
#     center_points = np.array(center_points)
#
#     midpoints = [
#         (center_points[0] + center_points[2]) / 2,
#         (center_points[1] + center_points[3]) / 2
#     ]
#
#     corners = []
#     for i in range(4):
#         x_center, y_center = center_points[i]
#         x_mid, y_mid = midpoints[i % 2]
#         dx = x_mid - x_center
#         dy = y_mid - y_center
#
#         corner1 = [int(x_center + dy), int(y_center - dx)]
#         corner2 = [int(x_center - dy), int(y_center + dx)]
#
#         corners.extend([corner1, corner2])
#
#     # Combine centers and corners into one list
#     all_points = center_points.tolist() + corners
#     return all_points
#
# run = True
# # while run:
#
# while iter_count <= 4:
#     re = None
#     le = None
#     # p_195 = None
#     # p_94 = None
#     # nose = None
#     # glabella = None
#
#     num = None
#     denom = None
#
#     re_points = []
#     le_points = []
#
#     count = 1
#
#     while count <= 500:
#         regions_re = {1: [33, 246, 7], 2: [246, 7, 163, 161], 3: [161, 163, 144, 160], 4: [160, 144, 145, 159],
#                       5: [159, 145, 153, 158], 6: [158, 153, 154, 157], 7: [157, 154, 155, 173], 8: [173, 155, 133]}
#         regions_le = {1: [263, 466, 249], 2: [466, 249, 390, 388], 3: [388, 390, 373, 387], 4: [387, 373, 374, 386],
#                       5: [386, 374, 380, 385], 6: [385, 380, 381, 384], 7: [384, 381, 382, 398], 8: [398, 382, 362]}
#
#         # r_lobe_coords = {130: None, 247: None, 30: None, 29: None, 27: None, 28: None, 56: None, 190: None, 243: None,
#         #                  25: None,
#         #                  110: None, 24: None, 23: None, 22: None, 26: None, 112: None}
#         # right_upper_lobe = {1: [247, 246, 33, 130], 2: [247, 30, 161, 246], 3: [30, 29, 160, 161], 4: [29, 27, 159, 160],
#         #                     5: [27, 28, 158, 159], 6: [28, 56, 157, 158], 7: [56, 190, 173, 157], 8: [190, 243, 133, 173]}
#         # right_lower_lobe = {1: [130, 33, 7, 25], 2: [7, 163, 110, 25], 3: [163, 144, 24, 110], 4: [144, 145, 23, 24],
#         #                     5: [145, 153, 22, 23], 6: [153, 154, 26, 22], 7: [154, 155, 112, 26], 8: [155, 133, 243, 112]}
#
#         # l_lobe_coords = {463: None, 414: None, 286: None, 258: None, 257: None, 259: None, 260: None, 467: None, 359: None,
#         #                  255: None, 339: None, 254: None, 253: None, 252: None, 256: None, 341: None}
#         # left_upper_lobe = {1: [359, 263, 466, 467], 2: [467, 466, 388, 260], 3: [260, 388, 387, 259],
#         #                    4: [259, 387, 386, 257],
#         #                    5: [257, 386, 385, 258], 6: [258, 385, 384, 286], 7: [286, 384, 398, 414],
#         #                    8: [414, 398, 362, 463]}
#         # left_lower_lobe = {1: [359, 263, 249, 255], 2: [255, 249, 390, 339], 3: [339, 390, 373, 254],
#         #                    4: [254, 373, 374, 253],
#         #                    5: [253, 374, 380, 252], 6: [252, 380, 381, 256], 7: [256, 381, 382, 341],
#         #                    8: [341, 382, 362, 463]}
#
#         percents_re = []
#         percents_le = []
#
#         random_x = None
#         random_y = None
#
#         # if count_temp > 10:
#         #     # random_x = int(np.random.uniform(regions_screen[(x_screen, y_screen)][0][0],
#         #     #                                  regions_screen[(x_screen, y_screen)][1][0], 1))
#         #     # random_y = int(np.random.uniform(regions_screen[(x_screen, y_screen)][0][1],
#         #     #                                  regions_screen[(x_screen, y_screen)][1][1], 1))
#         #
#         #     random_x = int(np.random.uniform(0, SCREEN_WIDTH, 1))
#         #     random_y = int(np.random.uniform(0, SCREEN_HEIGHT, 1))
#         #
#         #     last[0] = random_x
#         #     last[1] = random_y
#         #
#         #     count_temp = 0
#         #
#         # else:
#         #     if last[0] is None:
#         #         # random_x = int(np.random.uniform(regions_screen[(x_screen, y_screen)][0][0],
#         #         #                                  regions_screen[(x_screen, y_screen)][1][0], 1))
#         #         # random_y = int(np.random.uniform(regions_screen[(x_screen, y_screen)][0][1],
#         #         #                                  regions_screen[(x_screen, y_screen)][1][1], 1))
#         #
#         #         random_x = int(np.random.uniform(0, SCREEN_WIDTH, 1))
#         #         random_y = int(np.random.uniform(0, SCREEN_HEIGHT, 1))
#         #
#         #         last[0] = random_x
#         #         last[1] = random_y
#         #
#         #     else:
#         #         random_x = last[0]
#         #         random_y = last[1]
#         #
#         #     count_temp += 1
#
#         # print([random_x, random_y])
#         # pygame.draw.circle(screen, (255, 0, 0), [random_x, random_y], 10, 0)
#
#         success, img = cap.read()
#         img_for_eye = img.copy()
#         imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#         results = pose.process(imgRGB)
#
#         flag_re = False
#         flag_le = False
#         flag_n = False
#
#         right_eye = None
#         left_eye = None
#
#         focus_points = [None, None, None, None, None]
#         #                ri, ro, li, lo, n
#
#         face_points = [None, None, None]
#         #              lm,  c, rm
#
#         bounds_right = [None, None, None, None]
#         #                 ur,   lr,   rc,   lc
#         bounds_left = [None, None, None, None]
#         #                 ur,   lr,   rc,   lc
#
#         if success:
#             right_eye_main = []
#             left_eye_main = []
#
#             with mp_face_mesh.FaceMesh(
#                     static_image_mode=True,
#                     min_detection_confidence=0.5) as face_mesh:
#                 if True:
#                     # Convert the BGR image to RGB before processing.
#                     results_2 = face_mesh.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#
#                     # Print and draw face mesh landmarks on the image.
#                     if not results_2.multi_face_landmarks:
#                         continue
#
#                     # print(len(results_2.multi_face_landmarks))
#
#                     # annotated_image = img.copy()
#                     for face_landmarks in results_2.multi_face_landmarks:
#                         # mpDraw.draw_landmarks(
#                         #     image=img,
#                         #     landmark_list=face_landmarks,
#                         #     connections=mp_face_mesh.FACEMESH_CONTOURS,
#                         #     landmark_drawing_spec=drawing_spec,
#                         #     connection_drawing_spec=drawing_spec)
#
#                         for _, data_point in enumerate(face_landmarks.landmark):
#                             # print("data inner:", data_point.x, data_point.y)
#                             h, w, c = img.shape
#                             cx, cy = int(data_point.x * w), int(data_point.y * h)
#
#                             # if int(_) == 168:
#                             #     re_points.append([cx, cy])
#                             #     le_points.append([cx, cy])
#                             #     glabella = [cx, cy]
#                             # if int(_) == 133:
#                             #     re_points.append([cx, cy])
#                             # if int(_) == 362:
#                             #     le_points.append([cx, cy])
#                             # if int(_) == 195:
#                             #     p_195 = [cx, cy]
#                             # if int(_) == 94:
#                             #     p_94 = [cx, cy]
#                             # if int(_) == 4:
#                             #     nose = [cx, cy]
#
#                             if int(_) == 223:
#                                 bounds_right[0] = [cx, cy]
#                             if int(_) == 230:
#                                 bounds_right[1] = [cx, cy]
#                             if int(_) == 35:
#                                 bounds_right[2] = [cx, cy]
#                                 focus_points[1] = [cx, cy]
#                             if int(_) == 245:
#                                 bounds_right[3] = [cx, cy]
#                                 focus_points[0] = [cx, cy]
#
#                             if int(_) == 443:
#                                 bounds_left[0] = [cx, cy]
#                             if int(_) == 450:
#                                 bounds_left[1] = [cx, cy]
#                             if int(_) == 465:
#                                 bounds_left[2] = [cx, cy]
#                                 focus_points[2] = [cx, cy]
#                             if int(_) == 265:
#                                 bounds_left[3] = [cx, cy]
#                                 focus_points[3] = [cx, cy]
#
#                             if int(_) == 0:
#                                 focus_points[4] = [cx, cy]
#
#                             # if int(_) in right_eye_main_dict.keys():
#                             #     right_eye_main_dict[int(_)] = [cx, cy]
#                             # if int(_) in left_eye_main_dict.keys():
#                             #     left_eye_main_dict[int(_)] = [cx, cy]
#
#                             # if int(_) in [130, 247, 30, 29, 27, 28, 56, 190, 243, 25, 110, 24, 23, 22, 26, 112]:
#                             #     r_lobe_coords[int(_)] = [cx, cy]
#                             #
#                             # if int(_) in [463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341]:
#                             #     l_lobe_coords[int(_)] = [cx, cy]
#
#                     for cx, cy in bounds_right:
#                         cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED)
#
#                     for cx, cy in bounds_left:
#                         cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)
#
#                     bounds_right.append(bounds_right[0])
#                     bounds_left.append(bounds_left[0])
#
#                     bounds_right = find_rectangle_corners(bounds_right)
#                     bounds_left = find_rectangle_corners(bounds_left)
#
#                     print("bounds_right:", bounds_right)
#                     print("bounds_left:", bounds_left)
#
#                     hull_right = ConvexHull(bounds_right)
#                     for i in range(len(hull_right.vertices)):
#                         curr_point = bounds_right[hull_right.vertices[i % len(hull_right.vertices)]]
#                         next_point = bounds_right[hull_right.vertices[(i+1) % len(hull_right.vertices)]]
#
#                         cv.line(img, curr_point, next_point, (0, 0, 255), 2)
#
#                     hull_left = ConvexHull(bounds_left)
#                     for i in range(len(hull_left.vertices)):
#                         curr_point = bounds_left[hull_left.vertices[i % len(hull_left.vertices)]]
#                         next_point = bounds_left[hull_left.vertices[(i+1) % len(hull_left.vertices)]]
#
#                         cv.line(img, curr_point, next_point, (0, 255, 0), 2)
#
#
#             # if results.pose_landmarks:
#             #     # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
#             #     joints_dict = {}
#             #
#             #     for id, lm in enumerate(results.pose_landmarks.landmark):
#             #         h, w, c = img.shape
#             #         cx, cy = int(lm.x * w), int(lm.y * h)
#             #         cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
#             #
#             #         id_temp = int(id)
#             #
#             #         joints_dict[id_temp] = (cx, cy)
#             #
#             #         if 5 in joints_dict:
#             #             face_points[2] = np.array(joints_dict[5])
#             #
#             #         if 2 in joints_dict:
#             #             face_points[0] = np.array(joints_dict[2])
#             #
#             #         if (4 in joints_dict) and (6 in joints_dict):
#             #             focus_points[0] = np.array(joints_dict[4])
#             #             focus_points[1] = np.array(joints_dict[6])
#             #
#             #             flag_re = True
#             #
#             #         else:
#             #             flag_re = False
#             #
#             #         if (1 in joints_dict) and (3 in joints_dict):
#             #             focus_points[2] = np.array(joints_dict[1])
#             #             focus_points[3] = np.array(joints_dict[3])
#             #
#             #             flag_le = True
#             #
#             #         else:
#             #             flag_le = False
#             #
#             #         if 0 in joints_dict:
#             #             focus_points[4] = np.array(joints_dict[0])
#             #             face_points[1] = np.array(joints_dict[0])
#             #
#             #             flag_n = True
#             #
#             #         else:
#             #             flag_n = False
#             #
#             #     # Crop the right and left eyes position
#             #
#             #     if flag_re is True:
#             #         top_left = None
#             #         bot_right = None
#             #         center = (focus_points[0] + focus_points[1]) / 2
#             #         center[0] = int(center[0])
#             #         center[1] = int(center[1])
#             #
#             #         height = int(abs(np.linalg.norm(focus_points[0] - focus_points[1])) / 2)
#             #         right_eye = img_for_eye[min(focus_points[1][1] - height, focus_points[0][1] - height) - 5:max(focus_points[1][1] + height, focus_points[0][1] + height) + 5,
#             #                                 focus_points[1][0] - 5:focus_points[0][0] + 5]
#             #
#             #     if flag_le is True:
#             #         top_left = None
#             #         bot_right = None
#             #         center = (focus_points[2] + focus_points[3]) / 2
#             #         center[0] = int(center[0])
#             #         center[1] = int(center[1])
#             #
#             #         height = int(abs(np.linalg.norm(focus_points[2] - focus_points[3])) / 2)
#             #         left_eye = img_for_eye[min(focus_points[2][1] - height, focus_points[3][1] - height) - 5:max(focus_points[2][1] + height, focus_points[3][1] + height) + 5,
#             #                                focus_points[2][0] - 5:focus_points[3][0] + 5]
#             #
#             #     # Display cropped image
#             #
#             #     if (right_eye is not None) and (flag_re is True):
#             #         right_eye = cv.cvtColor(right_eye, cv.COLOR_BGR2GRAY)
#             #         # (thresh, right_eye) = cv.threshold(right_eye, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#             #         contours, hierarchy = cv.findContours(right_eye,
#             #                                               cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#             #         cv.drawContours(right_eye, contours, -1, (0, 255, 0), 3)
#             #         cv.imshow("right_eye", right_eye)
#             #         # cv.imwrite("re/re_" + str(count) + ".jpg", right_eye)
#             #
#             #     if (left_eye is not None) and (flag_le is True):
#             #         left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)
#             #         # (thresh, left_eye) = cv.threshold(left_eye, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#             #         contours, hierarchy = cv.findContours(left_eye,
#             #                                               cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#             #         cv.drawContours(left_eye, contours, -1, (0, 255, 0), 3)
#             #         cv.imshow("left_eye", left_eye)
#             #         # cv.imwrite("le/le_" + str(count) + ".jpg", left_eye)
#
#             # for key in right_eye_main_dict:
#             #     right_eye_main.append(right_eye_main_dict[key])
#             #
#             # for key in left_eye_main_dict:
#             #     left_eye_main.append(left_eye_main_dict[key])
#
#             # right_eye_main = np.array(right_eye_main, np.int32)
#             # right_eye_main = right_eye_main.reshape((-1, 1, 2))
#
#             # left_eye_main = np.array(left_eye_main, np.int32)
#             # left_eye_main = left_eye_main.reshape((-1, 1, 2))
#
#             # isClosed = True
#             # color = (255, 0, 0)
#             # thickness = 2
#             #
#             # img = cv.polylines(img, [right_eye_main], isClosed, color, thickness)
#             # img = cv.polylines(img, [left_eye_main], isClosed, color, thickness)
#
#             if bounds_right[0] is not None:
#                 flag_re = True
#
#             if bounds_left[0] is not None:
#                 flag_le = True
#
#             if flag_re is True:
#                 right_eye = img_for_eye[bounds_right[0][1]:bounds_right[1][1], bounds_right[2][0]:bounds_right[3][0]]
#
#             if flag_le is True:
#                 left_eye = img_for_eye[bounds_left[0][1]:bounds_left[1][1], bounds_left[2][0]:bounds_left[3][0]]
#
#             # for key in regions_re.keys():
#             #     temp = regions_re[key]
#             #     temp_coords = []
#             #
#             #     for index in temp:
#             #         temp_coords.append(right_eye_main_dict[index])
#             #
#             #     regions_re[key] = temp_coords
#             #
#             # for key in regions_le.keys():
#             #     temp = regions_le[key]
#             #     temp_coords = []
#             #
#             #     for index in temp:
#             #         temp_coords.append(left_eye_main_dict[index])
#             #
#             #     regions_le[key] = temp_coords
#
#             # for key in right_upper_lobe.keys():
#             #     temp = right_upper_lobe[key]
#             #     temp_coords = []
#             #
#             #     for index in temp:
#             #         if index in r_lobe_coords.keys():
#             #             temp_coords.append(r_lobe_coords[index])
#             #
#             #         else:
#             #             temp_coords.append(right_eye_main_dict[index])
#             #
#             #     right_upper_lobe[key] = temp_coords
#             #
#             # for key in right_lower_lobe.keys():
#             #     temp = right_lower_lobe[key]
#             #     temp_coords = []
#             #
#             #     for index in temp:
#             #         if index in r_lobe_coords.keys():
#             #             temp_coords.append(r_lobe_coords[index])
#             #
#             #         else:
#             #             temp_coords.append(right_eye_main_dict[index])
#             #
#             #     right_lower_lobe[key] = temp_coords
#             #
#             # for key in left_upper_lobe.keys():
#             #     temp = left_upper_lobe[key]
#             #     temp_coords = []
#             #
#             #     for index in temp:
#             #         if index in l_lobe_coords.keys():
#             #             temp_coords.append(l_lobe_coords[index])
#             #
#             #         else:
#             #             temp_coords.append(left_eye_main_dict[index])
#             #
#             #     left_upper_lobe[key] = temp_coords
#             #
#             # for key in left_lower_lobe.keys():
#             #     temp = left_lower_lobe[key]
#             #     temp_coords = []
#             #
#             #     for index in temp:
#             #         if index in l_lobe_coords.keys():
#             #             temp_coords.append(l_lobe_coords[index])
#             #
#             #         else:
#             #             temp_coords.append(left_eye_main_dict[index])
#             #
#             #     left_lower_lobe[key] = temp_coords
#
#             print("regions_re:", regions_re)
#             print("regions_le:", regions_le)
#             # print("right_upper_lobe:", right_upper_lobe)
#             # print("right_lower_lobe:", right_lower_lobe)
#             # print("left_upper_lobe:", left_upper_lobe)
#             # print("left_lower_lobe:", left_lower_lobe)
#
#             # rect_right = cv.boundingRect(right_eye_main)
#             # x, y, w, h = rect_right
#             # cropped_re = img[y:y + h, x:x + w].copy()
#             # # right_eye_main = right_eye_main - right_eye_main.min(axis=0)
#             # mask_re = np.zeros(cropped_re.shape[:2], np.uint8)
#             # cv.drawContours(mask_re, [right_eye_main], -1, (255, 255, 255), -1, cv.LINE_AA)
#             # dst_re = cv.bitwise_and(cropped_re, cropped_re, mask=mask_re)
#             # bg = np.ones_like(cropped_re, np.uint8) * 255
#             # cv.bitwise_not(bg, bg, mask=mask_re)
#             # dst_re = bg + dst_re
#             #
#             # for key in regions_re.keys():
#             #     temp = regions_re[key]
#             #     temp_t = []
#             #
#             #     for i in range(len(temp)):
#             #         temp_t.append([temp[i][0] - x, temp[i][1] - y])
#             #
#             #     regions_re[key] = temp_t
#
#             # for key in right_upper_lobe.keys():
#             #     temp = right_upper_lobe[key]
#             #     temp_t = []
#             #
#             #     for i in range(len(temp)):
#             #         temp_t.append([temp[i][0] - x, temp[i][1] - y])
#             #
#             #     right_upper_lobe[key] = temp_t
#             #
#             # for key in right_lower_lobe.keys():
#             #     temp = right_lower_lobe[key]
#             #     temp_t = []
#             #
#             #     for i in range(len(temp)):
#             #         temp_t.append([temp[i][0] - x, temp[i][1] - y])
#             #
#             #     right_lower_lobe[key] = temp_t
#
#             # rect_left = cv.boundingRect(left_eye_main)
#             # x, y, w, h = rect_left
#             # cropped_le = img[y:y + h, x:x + w].copy()
#             # left_eye_main = left_eye_main - left_eye_main.min(axis=0)
#             # mask_le = np.zeros(cropped_le.shape[:2], np.uint8)
#             # cv.drawContours(mask_le, [left_eye_main], -1, (255, 255, 255), -1, cv.LINE_AA)
#             # dst_le = cv.bitwise_and(cropped_le, cropped_le, mask=mask_le)
#             # bg = np.ones_like(cropped_le, np.uint8) * 255
#             # cv.bitwise_not(bg, bg, mask=mask_le)
#             # dst_le = bg + dst_le
#
#             # for key in regions_le.keys():
#             #     temp = regions_le[key]
#             #     temp_t = []
#             #
#             #     for i in range(len(temp)):
#             #         temp_t.append([temp[i][0] - x, temp[i][1] - y])
#             #
#             #     regions_le[key] = temp_t
#
#             # for key in left_upper_lobe.keys():
#             #     temp = left_upper_lobe[key]
#             #     temp_t = []
#             #
#             #     for i in range(len(temp)):
#             #         temp_t.append([temp[i][0] - x, temp[i][1] - y])
#             #
#             #     left_upper_lobe[key] = temp_t
#             #
#             # for key in left_lower_lobe.keys():
#             #     temp = left_lower_lobe[key]
#             #     temp_t = []
#             #
#             #     for i in range(len(temp)):
#             #         temp_t.append([temp[i][0] - x, temp[i][1] - y])
#             #
#             #     left_lower_lobe[key] = temp_t
#
#             kernel = np.array([[0, -1, 0], [-1, 6.5, -1], [0, -1, 0]])
#
#             thresh = 0
#
#             # dst_re = cv.cvtColor(dst_re, cv.COLOR_BGR2GRAY)
#             # dst_re = cv.threshold(dst_re, thresh, 255, cv.THRESH_BINARY)[1]
#             # dst_re = cv.filter2D(dst_re, -1, kernel)
#             # dst_re = cv.medianBlur(dst_re, 5)
#             # dst_re = cv.threshold(dst_re, thresh, 255, cv.THRESH_BINARY)[1]
#             # dst_re = cv.Canny(image=dst_re, threshold1=50, threshold2=200)
#
#             # dst_le = cv.cvtColor(dst_le, cv.COLOR_BGR2GRAY)
#             # dst_le = cv.filter2D(dst_le, -1, kernel)
#             # dst_le = cv.medianBlur(dst_le, 5)
#             # dst_le = cv.threshold(dst_le, thresh, 255, cv.THRESH_BINARY)[1]
#             # dst_le = cv.Sobel(src=dst_le, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
#             # dst_le = cv.Canny(image=dst_le, threshold1=50, threshold2=200)
#
#             ar_re = []
#             ar_le = []
#
#             # for key in regions_re.keys():
#             #     # mask_re = np.zeros_like(dst_re)
#             #     #
#             #     list_coords = np.array(regions_re[key])
#             #     # # print("list_coords_re:", list_coords)
#             #     # cv.fillPoly(mask_re, [list_coords], 255)
#             #     # mask_re = cv.threshold(mask_re, 200, 255, cv.THRESH_BINARY)[1]
#             #     # masked_image = cv.bitwise_and(dst_re, mask_re)
#             #     masked_image = dst_re
#             #
#             #     # black_pixel_count_re = 0
#             #     # white_pixel_count_re = 0
#             #
#             #     pixel_count = 0
#             #     values = 0
#             #
#             #     for y in range(masked_image.shape[0]):
#             #         for x in range(masked_image.shape[1]):
#             #             if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
#             #                 pixel_count += 1
#             #                 values += masked_image[y, x]
#             #
#             #                 # if masked_image[y, x] < 20:
#             #                 #     black_pixel_count_re += 1
#             #                 #
#             #                 # else:
#             #                 #     white_pixel_count_re += 1
#             #
#             #     # percents_re.append(round(black_pixel_count_re / (black_pixel_count_re + white_pixel_count_re), 2))
#             #     percents_re.append(round((values / pixel_count), 2))
#             #     text_temp = ":" + str(pixel_count) + ":"
#             #     ar_re.append(text_temp)
#             #
#             # for key in regions_le.keys():
#             #     # mask_le = np.zeros_like(dst_le)
#             #     #
#             #     list_coords = np.array(regions_le[key])
#             #     # # print("list_coords_le:", list_coords)
#             #     # cv.fillPoly(mask_le, [list_coords], 255)
#             #     # # cv.drawContours(mask_le, [list_coords], -1, 255, 1, thickness=cv.LINE_AA)
#             #     # # mask_le = cv.threshold(mask_le, 200, 255, cv.THRESH_BINARY)[1]
#             #     # masked_image = cv.bitwise_and(dst_le, mask_le)
#             #     masked_image = dst_le
#             #
#             #     # black_pixel_count_le = 0
#             #     # white_pixel_count_le = 0
#             #
#             #     pixel_count = 0
#             #     values = 0
#             #
#             #     for y in range(masked_image.shape[0]):
#             #         for x in range(masked_image.shape[1]):
#             #             if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
#             #                 pixel_count += 1
#             #                 values += masked_image[y, x]
#             #
#             #                 # if masked_image[y, x] < 20:
#             #                 #     black_pixel_count_le += 1
#             #                 #
#             #                 # else:
#             #                 #     white_pixel_count_le += 1
#             #
#             #     # percents_le.append(round(black_pixel_count_le / (black_pixel_count_le + white_pixel_count_le), 2))
#             #     percents_le.append(round((values / pixel_count), 2))
#             #     text_temp = ":" + str(pixel_count) + ":"
#             #     ar_le.append(text_temp)
#
#             index = 0
#             # for key in right_upper_lobe.keys():
#             #     list_coords = np.array(right_upper_lobe[key])
#             #     masked_image = dst_re
#             #     pixel_count = 0
#             #
#             #     for y in range(masked_image.shape[0]):
#             #         for x in range(masked_image.shape[1]):
#             #             if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
#             #                 pixel_count += 1
#             #
#             #     ar_re[index] = str(pixel_count) + ar_re[index]
#             #     index += 1
#             #
#             # index = 0
#             # for key in right_lower_lobe.keys():
#             #     list_coords = np.array(right_lower_lobe[key])
#             #     masked_image = dst_re
#             #     pixel_count = 0
#             #
#             #     for y in range(masked_image.shape[0]):
#             #         for x in range(masked_image.shape[1]):
#             #             if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
#             #                 pixel_count += 1
#             #
#             #     ar_re[index] = ar_re[index] + str(pixel_count)
#             #     index += 1
#             #
#             # index = 0
#             # for key in left_upper_lobe.keys():
#             #     list_coords = np.array(left_upper_lobe[key])
#             #     masked_image = dst_re
#             #     pixel_count = 0
#             #
#             #     for y in range(masked_image.shape[0]):
#             #         for x in range(masked_image.shape[1]):
#             #             if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
#             #                 pixel_count += 1
#             #
#             #     ar_le[index] = str(pixel_count) + ar_le[index]
#             #     index += 1
#             #
#             # index = 0
#             # for key in left_lower_lobe.keys():
#             #     list_coords = np.array(left_lower_lobe[key])
#             #     masked_image = dst_re
#             #     pixel_count = 0
#             #
#             #     for y in range(masked_image.shape[0]):
#             #         for x in range(masked_image.shape[1]):
#             #             if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
#             #                 pixel_count += 1
#             #
#             #     ar_le[index] = ar_le[index] + str(pixel_count)
#             #     index += 1
#
#             # print("percents_re:", percents_re)
#             # print("percents_le:", percents_le)
#
#             # contours, _ = cv.findContours(dst_le, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
#             # contour_image = np.zeros_like(dst_le)
#             # cv.drawContours(contour_image, contours, -1, (255, 255, 255), 2)  # Draw all contours with thickness 2
#             # # print("_:", _)
#
#
#             # cv.imshow("right_eye cropped", dst_re)
#             # cv.imshow("left_eye cropped", dst_le)
#             # cv.imshow("left_eye contoured", contour_image)
#
#             # Display cropped image
#
#             if (right_eye is not None) and (flag_re is True):
#                 right_eye = cv.cvtColor(right_eye, cv.COLOR_BGR2GRAY)
#
#                 # right_eye = cv.GaussianBlur(right_eye, (7, 7), 0)
#                 # kernel = np.array([[0, -1, 0],
#                 #                    [-1, 5, -1],
#                 #                    [0, -1, 0]])
#                 # right_eye = cv.filter2D(right_eye, -1, kernel)
#
#                 ratio_h_w = right_eye.shape[0] / right_eye.shape[1]
#                 size_1 = (200, int(ratio_h_w * 200))
#                 size_2 = (int(100 / ratio_h_w), 100)
#
#                 # if size_1[1] <= 100:
#                 #     for i in range(100 - size_1[1]):
#                 #         pass
#                 #
#                 # else:
#                 #     pass
#
#                 temp = min(100, max(int(200 * (right_eye.shape[0] / right_eye.shape[1])), 100))
#                 right_eye = cv.resize(right_eye, (200, temp),
#                                       interpolation=cv.INTER_LINEAR)
#
#                 # print(right_eye.shape)
#                 right_eye = np.array(right_eye)
#                 newrow = np.zeros((1, 200), dtype=int)
#
#                 # print("right_eye shapes:", newrow.shape, right_eye.shape)
#                 # for i in range(100 - temp):
#                 #     right_eye = np.vstack([right_eye, newrow])
#
#                 # _, threshold = cv.threshold(right_eye, 3, 255, cv.THRESH_BINARY_INV)
#
#                 # (thresh, right_eye) = cv.threshold(right_eye, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#                 # contours, hierarchy = cv.findContours(threshold,
#                 #                                       cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#                 # for cnt in contours:
#                 #     (x, y, w, h) = cv.boundingRect(cnt)
#                 #     # cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
#                 #     cv.rectangle(right_eye, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                 #     cv.line(right_eye, (x + int(w / 2), 0), (x + int(w / 2), right_eye.shape[0]), (0, 255, 0), 2)
#                 #     cv.line(right_eye, (0, y + int(h / 2)), (right_eye.shape[1], y + int(h / 2)), (0, 255, 0), 2)
#
#                 # M = cv.getRotationMatrix2D((int((bounds_right[2][0] + bounds_right[3][0]) / 2),
#                 #                                     int((bounds_right[2][1] + bounds_right[3][1]) / 2)),
#                 #                                    angle_of_inclination(bounds_right[2], bounds_right[3]), 1.0)
#                 # right_eye = cv.warpAffine(right_eye, M, (320, 320))
#
#                 # cv.putText(right_eye, str((right_eye.shape[0], right_eye.shape[1])), (50, 50),
#                 #            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
#
#                 # right_eye = cv.resize(right_eye, (200, 100),
#                 #                           interpolation=cv.INTER_LINEAR)
#
#                 # se = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
#                 # bg = cv.morphologyEx(right_eye, cv.MORPH_DILATE, se)
#                 # out_gray = cv.divide(right_eye, bg, scale=255)
#                 # out_binary = cv.threshold(out_gray, 0, 255, cv.THRESH_OTSU)[1]
#
#                 # thresh2 = cv.adaptiveThreshold(right_eye, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                 #                                 cv.THRESH_BINARY, 199, 5)
#
#                 # cv.imshow("right_eye", right_eye)
#                 # cv.imshow("right_eye", out_binary)
#                 # cv.imshow("right_eye", thresh2)
#                 # cv.imwrite("eye-right/re_O_" + str(count + 499) + ".jpg", right_eye)
#
#             if (left_eye is not None) and (flag_le is True):
#                 left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)
#                 # (thresh, left_eye) = cv.threshold(left_eye, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#
#                 # left_eye = cv.GaussianBlur(left_eye, (7, 7), 0)
#                 # kernel = np.array([[0, -1, 0],
#                 #                    [-1, 5, -1],
#                 #                    [0, -1, 0]])
#                 # left_eye = cv.filter2D(left_eye, -1, kernel)
#
#                 # _, threshold = cv.threshold(left_eye, 3, 255, cv.THRESH_BINARY_INV)
#
#                 ratio_h_w = right_eye.shape[0] / right_eye.shape[1]
#
#                 # (thresh, im_bw) = cv.threshold(left_eye, 200, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#
#                 size_1 = (200, int(ratio_h_w * 200))
#                 size_2 = (int(100 / ratio_h_w), 100)
#
#                 temp = min(100, max(int(200 * (left_eye.shape[0] / left_eye.shape[1])), 100))
#                 left_eye = cv.resize(left_eye, (200, temp),
#                                      interpolation=cv.INTER_LINEAR)
#
#                 newrow = np.zeros((1, 200), dtype=int)
#                 # for i in range(100 - temp):
#                 #     left_eye = np.vstack([left_eye, newrow])
#
#                 # contours, hierarchy = cv.findContours(threshold,
#                 #                                       cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#                 # for cnt in contours:
#                 #     (x, y, w, h) = cv.boundingRect(cnt)
#                 #     # cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
#                 #     cv.rectangle(left_eye, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                 #     cv.line(left_eye, (x + int(w / 2), 0), (x + int(w / 2), left_eye.shape[0]), (0, 255, 0), 2)
#                 #     cv.line(left_eye, (0, y + int(h / 2)), (left_eye.shape[1], y + int(h / 2)), (0, 255, 0), 2)
#
#                 # M = cv.getRotationMatrix2D((int((bounds_left[2][0] + bounds_left[3][0]) / 2),
#                 #                                    int((bounds_left[2][1] + bounds_left[3][1]) / 2)),
#                 #                                   angle_of_inclination(bounds_left[2], bounds_left[3]), 1.0)
#                 # left_eye = cv.warpAffine(left_eye, M, (320, 320))
#
#                 # cv.putText(left_eye, str((left_eye.shape[0], left_eye.shape[1])), (50, 50),
#                 #            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
#
#                 # left_eye = cv.resize(left_eye, (200, 100),
#                 #                       interpolation=cv.INTER_LINEAR)
#
#                 # se = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
#                 # bg = cv.morphologyEx(left_eye, cv.MORPH_DILATE, se)
#                 # out_gray = cv.divide(left_eye, bg, scale=255)
#                 # out_binary = cv.threshold(out_gray, 0, 255, cv.THRESH_OTSU)[1]
#
#                 # thresh2 = cv.adaptiveThreshold(left_eye, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                 #                                cv.THRESH_BINARY, 199, 5)
#
#                 # print(left_eye)
#                 # print(len(left_eye), len(left_eye[0]))
#                 left_eye = left_eye / 255
#
#                 # left_eye_temp = np.expand_dims(left_eye, axis=-1)
#                 left_eye_temp = np.array(left_eye)
#                 # left_eye_temp = (left_eye_temp * 255).astype(np.uint8)
#                 left_eye_temp = left_eye_temp.flatten()
#                 left_eye_temp = left_eye_temp.reshape((-1, 200, 100, 1))
#
#                 # print(left_eye_temp.shape)
#                 # predictions = model_left_eye.predict(left_eye_temp)
#                 # predictions = None
#
#                 # class_labels = ['0', '1', '2', '3']  # Replace with your class labels
#                 # class_labels = ['R', 'G', 'B', 'O']  # Replace with your class labels
#                 # predicted_class_index = np.argmax(predictions)
#                 # predicted_class = class_labels[predicted_class_index]
#                 #
#                 # print("Predicted class:", predicted_class)
#                 # print("Predictions:", predictions)
#
#                 # cv.imshow("left_eye", left_eye)
#                 # cv.imshow("left_eye", out_binary)
#                 # cv.imshow("left_eye", thresh2)
#                 # cv.imwrite("eye-left/le_O_" + str(count + 499) + ".jpg", left_eye)
#
#             angle_right = None
#             angle_left = None
#
#             if (focus_points[4] is not None) and (focus_points[0] is not None):
#                 angle_right = calculate_angle(focus_points[0], focus_points[4], [focus_points[4][0], focus_points[4][1] - 0.001])
#
#             if(focus_points[4] is not None) and (focus_points[2] is not None):
#                 angle_left = calculate_angle([focus_points[4][0], focus_points[4][1] - 0.001], focus_points[4], focus_points[2])
#
#             # print(angle_right, angle_left)
#
#             cTime = time.time()
#             fps = 1 / (cTime - pTime)
#             pTime = cTime
#
#             # cv.putText(img, str(np.round_([motion[0], motion[2], motion[4]], decimals=3)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
#             # cv.putText(img, str(count), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
#             # cv.imshow("Image", img)
#             # cv.waitKey(1)
#
#             temp_predict = []
#
#             # rows.append([count, 'O'])
#             temp_final = [(100 * (iter_count - 1)) + count]
#             for i in percents_re:
#                 temp_final.append(round((i / 256), 5))
#                 temp_predict.append(round((i / 256), 5))
#             for i in ar_re:
#                 temp_final.append(i)
#             temp_final.append(angle_right)
#
#             for i in percents_le:
#                 temp_final.append(round((i / 256), 5))
#                 temp_predict.append(round((i / 256), 5))
#             for i in ar_le:
#                 temp_final.append(i)
#             temp_final.append(angle_left)
#
#             temp_predict.append(angle_right)
#             temp_predict.append(angle_left)
#
#             # temp_final.append(x_screen)
#             # temp_final.append(y_screen)
#
#             rows.append(temp_final)
#
#             temp_predict = np.array(temp_predict)
#
#             # predicted_vals = pickled_model.predict([temp_predict])
#             # predicted_vals = pickled_model_combined.predict([temp_predict])
#             # predicted_vals = loaded_model.predict(temp_predict)
#
#             # predicted_x = predicted_vals[0][0]
#             # predicted_y = predicted_vals[0][1]
#             # # predicted_x = predicted_vals
#             # # predicted_y = 5
#             #
#             # predicted_x_upper = math.ceil(predicted_x)
#             # predicted_y_upper = math.ceil(predicted_y)
#             #
#             # if predicted_x_upper > 10:
#             #     predicted_x_upper = 10
#             # if predicted_x_upper < 1:
#             #     predicted_x_upper = 1
#             #
#             # if predicted_y_upper > 8:
#             #     predicted_y_upper = 8
#             # if predicted_y_upper < 1:
#             #     predicted_y_upper = 1
#             #
#             # predicted_x_lower = math.floor(predicted_x)
#             # predicted_y_lower = math.floor(predicted_y)
#             #
#             # if predicted_x_lower < 1:
#             #     predicted_x_lower = 1
#             # if predicted_x_lower > 10:
#             #     predicted_x_lower = 10
#             #
#             # if predicted_y_lower < 1:
#             #     predicted_y_lower = 1
#             # if predicted_y_lower > 8:
#             #     predicted_y_lower = 8
#             #
#             # total += 1
#             # if (random_x <= regions_screen[(predicted_x_lower, predicted_y_lower)][1][0]) and (random_x >= regions_screen[(predicted_x_lower, predicted_y_lower)][0][0]):
#             #     correct_x += 1
#             #
#             # if (random_y <= regions_screen[(predicted_x_lower, predicted_y_lower)][1][1]) and (random_y >= regions_screen[(predicted_x_lower, predicted_y_lower)][0][1]):
#             #     correct_y += 1
#             #
#             # print("metrics:", total, correct_x, correct_y)
#             #
#             # pygame.draw.rect(screen, color, pygame.Rect(regions_screen[(predicted_x_lower, predicted_y_lower)][0][0],
#             #                                             regions_screen[(predicted_x_lower, predicted_y_lower)][0][1],
#             #                                             regions_screen[(predicted_x_lower, predicted_y_lower)][1][0],
#             #                                             regions_screen[(predicted_x_lower, predicted_y_lower)][1][1]))
#             # # pygame.draw.rect(screen, color, pygame.Rect(regions_screen[(predicted_x_upper, predicted_y_upper)][0][0],
#             # #                                             regions_screen[(predicted_x_upper, predicted_y_upper)][0][1],
#             # #                                             regions_screen[(predicted_x_upper, predicted_y_upper)][1][0],
#             # #                                             regions_screen[(predicted_x_upper, predicted_y_upper)][1][1]))
#
#             count += 1
#             radius = 50
#
#             if len(re_points) == 2:
#                 re = np.linalg.norm(np.array(re_points[0]) - np.array(re_points[1]))
#
#             if len(le_points) == 2:
#                 le = np.linalg.norm(np.array(le_points[0]) - np.array(le_points[1]))
#
#             # if p_195 is not None:
#             #     num = np.linalg.norm(np.array(p_195) - np.array(nose))
#             #
#             # if p_94 is not None:
#             #     denom = np.linalg.norm(np.array(nose) - np.array(p_94))
#             #
#             # re_part = re / (re + le)
#             # num_part = num / (num + denom)
#             # a = int(min(radius * re_part, radius * (1 - re_part)))
#             # b = int(min(radius * num_part, radius * (1 - num_part)))
#             #
#             # print(a, b)
#             # cv.ellipse(img, (nose[0], nose[1]), (a, b), 0, 0, 360, (0, 255, 0), -1)
#
#             cv.imshow("img", img)
#
#             if cv.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#             # for event in pygame.event.get():
#             #     if event.type == pygame.QUIT:
#             #         run = False
#             #
#             # pygame.display.update()
#
#     iter_count += 1
#
# # pygame.quit()
# cap.release()
# cv.destroyAllWindows()
#
# print("time taken:", time.time() - time_overall)
#
# # print("metrics final:", total, correct_x, correct_y)
#
# # with open("eye-coords.csv", 'w') as csvfile:
# #     csvwriter = csv.writer(csvfile)
# #     csvwriter.writerow(['sno', 'r'])
# #     csvwriter.writerows(rows)
#
# # with open(filename_eye_data, 'a') as csvfile:
# #     csvwriter = csv.writer(csvfile)
# #     # csvwriter.writerow(['sno', 'r'])
# #     csvwriter.writerows(rows)

import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import math
import matplotlib.pyplot as plt

from CriticNetwork import EyeVidPre_and_ViViT

header = ["sno",
          "re_1", "re_2", "re_3", "re_4", "re_5", "re_6", "re_7", "re_8",
          "ar_re_1", "ar_re_2", "ar_re_3", "ar_re_4", "ar_re_5", "ar_re_6", "ar_re_7", "ar_re_8",
          "rea",
          "le_1", "le_2", "le_3", "le_4", "le_5", "le_6", "le_7", "le_8",
          "ar_le_1", "ar_le_2", "ar_le_3", "ar_le_4", "ar_le_5", "ar_le_6", "ar_le_7", "ar_le_8",
          "lea",
          "x", "y"]

# model_left_eye = load_model('cnn_model_1_left_eye.h5')
filename_eye_data = "eye_data_percents_2.csv"

# Task1: get the eye point
# Task2: see if the eye points have enough resolution..., if not, zoom in, do resolution
# Task3: calculate position based on this data

# OR

# Task2: just tell if the person is looking at the instructor or not // first this: ...

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh

flag_up = False
time_up_start = None
pTime = 0

cap = cv.VideoCapture(0)

def calculate_angle(p1, p2, p3):
    # ang1 = np.arctan2(*p1[::-1])
    # ang2 = np.arctan2(*p2[::-1])
    # return np.rad2deg((ang1 - ang2) % (2 * np.pi))

    # Calculate the vectors between the points
    vector1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    vector2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # Calculate the dot product
    dot_product = np.dot(vector1, vector2)

    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the angle in radians
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))

    # Convert radians to degrees
    angle_deg = np.degrees(angle_rad)

    if p3[0] < p1[0]:
        return -angle_deg

    return angle_deg

motion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# count = 1

right_eye_video = []
left_eye_video = []

right_eye_main = []
right_eye_main_dict = {33: None, 246: None, 161: None, 160: None, 159: None, 158: None, 157: None,
                       173: None, 133: None, 155: None, 154: None, 153: None, 145: None, 144: None,
                       163: None, 7: None}
r_lobe_coords = {130: None, 247: None, 30: None, 29: None, 27: None, 28: None, 56: None, 190: None, 243: None, 25: None,
                 110: None, 24: None, 23: None, 22: None, 26: None, 112: None}
regions_re = {1: [33, 246, 7], 2: [246, 7, 163, 161], 3: [161, 163, 144, 160], 4: [160, 144, 145, 159],
              5: [159, 145, 153, 158], 6: [158, 153, 154, 157], 7: [157, 154, 155, 173], 8: [173, 155, 133]}
right_upper_lobe = {1: [247, 246, 33, 130], 2: [247, 30, 161, 246], 3: [30, 29, 160, 161], 4: [29, 27, 159, 160],
                    5: [27, 28, 158, 159], 6: [28, 56, 157, 158], 7: [56, 190, 173, 157], 8: [190, 243, 133, 173]}
right_lower_lobe = {1: [130, 33, 7, 25], 2: [7, 163, 110, 25], 3: [163, 144, 24, 110], 4: [144, 145, 23, 24],
                    5: [145, 153, 22, 23], 6: [153, 154, 26, 22], 7: [154, 155, 112, 26], 8: [155, 133, 243, 112]}

left_eye_main = []
left_eye_main_dict = {263: None, 466: None, 388: None, 387: None, 386: None, 385: None, 384: None,
                      398: None, 362: None, 382: None, 381: None, 380: None, 374: None, 373: None,
                      390: None, 249: None}
l_lobe_coords = {463: None, 414: None, 286: None, 258: None, 257: None, 259: None, 260: None, 467: None, 359: None,
                 255: None, 339: None, 254: None, 253: None, 252: None, 256: None, 341: None}
regions_le = {1: [263, 466, 249], 2: [466, 249, 390, 388], 3: [388, 390, 373, 387], 4: [387, 373, 374, 386],
              5: [386, 374, 380, 385], 6: [385, 380, 381, 384], 7: [384, 381, 382, 398], 8: [398, 382, 362]}
left_upper_lobe = {1: [359, 263, 466, 467], 2: [467, 466, 388, 260], 3: [260, 388, 387, 259], 4: [259, 387, 386, 257],
                   5: [257, 386, 385, 258], 6: [258, 385, 384, 286], 7: [286, 384, 398, 414], 8: [414, 398, 362, 463]}
left_lower_lobe = {1: [359, 263, 249, 255], 2: [255, 249, 390, 339], 3: [339, 390, 373, 254], 4: [254, 373, 374, 253],
                   5: [253, 374, 380, 252], 6: [252, 380, 381, 256], 7: [256, 381, 382, 341], 8: [341, 382, 362, 463]}


resized_re_t_before = None
resized_le_t_before = None

resized_re_OTSU_before = None
resized_le_OTSU_before = None

resized_re_KMeans_before = None
resized_le_KMeans_before = None
def KMeans_Image(image, K=3):
    print("inside KMeans_Image", image.shape)

    Z = image.reshape((-1, 2))
    Z = np.float32(Z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))

    return res2

def angle_of_inclination(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2

    delta_x = x2 - x1
    delta_y = y2 - y1
    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# Task1:

drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

count_temp = 1
last = [None, None]

time_overall = time.time()

regions_R = {'ul': [200, 480], 'lr': [600, 640]}
regions_G = {'ul': [200, 320], 'lr': [600, 480]}
regions_B = {'ul': [200, 160], 'lr': [600, 320]}

iter_count = 1

run = True
# while run:

eye_crop_window = 32

eye_right_cropped_model_input = []
eye_left_cropped_model_input = []

while iter_count <= 4:
    re = None
    le = None
    p_195 = None
    p_94 = None
    nose = None
    glabella = None

    num = None
    denom = None

    re_points = []
    le_points = []

    count = 1

    while count <= 500:
        regions_re = {1: [33, 246, 7], 2: [246, 7, 163, 161], 3: [161, 163, 144, 160], 4: [160, 144, 145, 159],
                      5: [159, 145, 153, 158], 6: [158, 153, 154, 157], 7: [157, 154, 155, 173], 8: [173, 155, 133]}
        regions_le = {1: [263, 466, 249], 2: [466, 249, 390, 388], 3: [388, 390, 373, 387], 4: [387, 373, 374, 386],
                      5: [386, 374, 380, 385], 6: [385, 380, 381, 384], 7: [384, 381, 382, 398], 8: [398, 382, 362]}

        r_lobe_coords = {130: None, 247: None, 30: None, 29: None, 27: None, 28: None, 56: None, 190: None, 243: None,
                         25: None,
                         110: None, 24: None, 23: None, 22: None, 26: None, 112: None}
        right_upper_lobe = {1: [247, 246, 33, 130], 2: [247, 30, 161, 246], 3: [30, 29, 160, 161], 4: [29, 27, 159, 160],
                            5: [27, 28, 158, 159], 6: [28, 56, 157, 158], 7: [56, 190, 173, 157], 8: [190, 243, 133, 173]}
        right_lower_lobe = {1: [130, 33, 7, 25], 2: [7, 163, 110, 25], 3: [163, 144, 24, 110], 4: [144, 145, 23, 24],
                            5: [145, 153, 22, 23], 6: [153, 154, 26, 22], 7: [154, 155, 112, 26], 8: [155, 133, 243, 112]}

        l_lobe_coords = {463: None, 414: None, 286: None, 258: None, 257: None, 259: None, 260: None, 467: None, 359: None,
                         255: None, 339: None, 254: None, 253: None, 252: None, 256: None, 341: None}
        left_upper_lobe = {1: [359, 263, 466, 467], 2: [467, 466, 388, 260], 3: [260, 388, 387, 259],
                           4: [259, 387, 386, 257],
                           5: [257, 386, 385, 258], 6: [258, 385, 384, 286], 7: [286, 384, 398, 414],
                           8: [414, 398, 362, 463]}
        left_lower_lobe = {1: [359, 263, 249, 255], 2: [255, 249, 390, 339], 3: [339, 390, 373, 254],
                           4: [254, 373, 374, 253],
                           5: [253, 374, 380, 252], 6: [252, 380, 381, 256], 7: [256, 381, 382, 341],
                           8: [341, 382, 362, 463]}

        percents_re = []
        percents_le = []

        success, img = cap.read()
        img_for_eye = img.copy()
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        flag_re = False
        flag_le = False
        flag_n = False

        right_eye = None
        left_eye = None

        # focus_points = [None, None, None, None, None]
        # #                ri, ro, li, lo, n
        #
        # face_points = [None, None, None]
        # #              lm,  c, rm
        #
        # bounds_right = [None, None, None, None]
        # #                 ur,   lr,   rc,   lc
        # bounds_left = [None, None, None, None]
        # #                 ur,   lr,   rc,   lc

        if success:
            right_eye_main = []
            left_eye_main = []

            with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    min_detection_confidence=0.5) as face_mesh:
                if True:
                    # Convert the BGR image to RGB before processing.
                    results_2 = face_mesh.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

                    # Print and draw face mesh landmarks on the image.
                    if not results_2.multi_face_landmarks:
                        continue

                    # print(len(results_2.multi_face_landmarks))

                    # annotated_image = img.copy()
                    for face_landmarks in results_2.multi_face_landmarks:
                        # mpDraw.draw_landmarks(
                        #     image=img,
                        #     landmark_list=face_landmarks,
                        #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                        #     landmark_drawing_spec=drawing_spec,
                        #     connection_drawing_spec=drawing_spec)

                        for _, data_point in enumerate(face_landmarks.landmark):
                            # print("data inner:", data_point.x, data_point.y)
                            h, w, c = img.shape
                            cx, cy = int(data_point.x * w), int(data_point.y * h)

                            if int(_) == 168:
                                re_points.append([cx, cy])
                                le_points.append([cx, cy])
                                glabella = [cx, cy]
                            if int(_) == 133:
                                re_points.append([cx, cy])
                            if int(_) == 362:
                                le_points.append([cx, cy])
                            # if int(_) == 195:
                            #     p_195 = [cx, cy]
                            # if int(_) == 94:
                            #     p_94 = [cx, cy]
                            # if int(_) == 4:
                            #     nose = [cx, cy]
                            #
                            # if int(_) == 223:
                            #     bounds_right[0] = [cx, cy]
                            # if int(_) == 230:
                            #     bounds_right[1] = [cx, cy]
                            # if int(_) == 35:
                            #     bounds_right[2] = [cx, cy]
                            #     focus_points[1] = [cx, cy]
                            # if int(_) == 245:
                            #     bounds_right[3] = [cx, cy]
                            #     focus_points[0] = [cx, cy]
                            #
                            # if int(_) == 443:
                            #     bounds_left[0] = [cx, cy]
                            # if int(_) == 450:
                            #     bounds_left[1] = [cx, cy]
                            # if int(_) == 465:
                            #     bounds_left[2] = [cx, cy]
                            #     focus_points[2] = [cx, cy]
                            # if int(_) == 265:
                            #     bounds_left[3] = [cx, cy]
                            #     focus_points[3] = [cx, cy]

                            # if int(_) == 0:
                            #     focus_points[4] = [cx, cy]

                            if int(_) in right_eye_main_dict.keys():
                                right_eye_main_dict[int(_)] = [cx, cy]
                            if int(_) in left_eye_main_dict.keys():
                                left_eye_main_dict[int(_)] = [cx, cy]

                            if int(_) in [130, 247, 30, 29, 27, 28, 56, 190, 243, 25, 110, 24, 23, 22, 26, 112]:
                                r_lobe_coords[int(_)] = [cx, cy]

                            if int(_) in [463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341]:
                                l_lobe_coords[int(_)] = [cx, cy]

                    # for cx, cy in bounds_right:
                    #     cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED)
                    #
                    # for cx, cy in bounds_left:
                    #     cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)


            # if results.pose_landmarks:
            #     # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            #     joints_dict = {}
            #
            #     for id, lm in enumerate(results.pose_landmarks.landmark):
            #         h, w, c = img.shape
            #         cx, cy = int(lm.x * w), int(lm.y * h)
            #         cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
            #
            #         id_temp = int(id)
            #
            #         joints_dict[id_temp] = (cx, cy)
            #
            #         if 5 in joints_dict:
            #             face_points[2] = np.array(joints_dict[5])
            #
            #         if 2 in joints_dict:
            #             face_points[0] = np.array(joints_dict[2])
            #
            #         if (4 in joints_dict) and (6 in joints_dict):
            #             focus_points[0] = np.array(joints_dict[4])
            #             focus_points[1] = np.array(joints_dict[6])
            #
            #             flag_re = True
            #
            #         else:
            #             flag_re = False
            #
            #         if (1 in joints_dict) and (3 in joints_dict):
            #             focus_points[2] = np.array(joints_dict[1])
            #             focus_points[3] = np.array(joints_dict[3])
            #
            #             flag_le = True
            #
            #         else:
            #             flag_le = False
            #
            #         if 0 in joints_dict:
            #             focus_points[4] = np.array(joints_dict[0])
            #             face_points[1] = np.array(joints_dict[0])
            #
            #             flag_n = True
            #
            #         else:
            #             flag_n = False
            #
            #     # Crop the right and left eyes position
            #
            #     if flag_re is True:
            #         top_left = None
            #         bot_right = None
            #         center = (focus_points[0] + focus_points[1]) / 2
            #         center[0] = int(center[0])
            #         center[1] = int(center[1])
            #
            #         height = int(abs(np.linalg.norm(focus_points[0] - focus_points[1])) / 2)
            #         right_eye = img_for_eye[min(focus_points[1][1] - height, focus_points[0][1] - height) - 5:max(focus_points[1][1] + height, focus_points[0][1] + height) + 5,
            #                                 focus_points[1][0] - 5:focus_points[0][0] + 5]
            #
            #     if flag_le is True:
            #         top_left = None
            #         bot_right = None
            #         center = (focus_points[2] + focus_points[3]) / 2
            #         center[0] = int(center[0])
            #         center[1] = int(center[1])
            #
            #         height = int(abs(np.linalg.norm(focus_points[2] - focus_points[3])) / 2)
            #         left_eye = img_for_eye[min(focus_points[2][1] - height, focus_points[3][1] - height) - 5:max(focus_points[2][1] + height, focus_points[3][1] + height) + 5,
            #                                focus_points[2][0] - 5:focus_points[3][0] + 5]
            #
            #     # Display cropped image
            #
            #     if (right_eye is not None) and (flag_re is True):
            #         right_eye = cv.cvtColor(right_eye, cv.COLOR_BGR2GRAY)
            #         # (thresh, right_eye) = cv.threshold(right_eye, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            #         contours, hierarchy = cv.findContours(right_eye,
            #                                               cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            #         cv.drawContours(right_eye, contours, -1, (0, 255, 0), 3)
            #         cv.imshow("right_eye", right_eye)
            #         # cv.imwrite("re/re_" + str(count) + ".jpg", right_eye)
            #
            #     if (left_eye is not None) and (flag_le is True):
            #         left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)
            #         # (thresh, left_eye) = cv.threshold(left_eye, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            #         contours, hierarchy = cv.findContours(left_eye,
            #                                               cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            #         cv.drawContours(left_eye, contours, -1, (0, 255, 0), 3)
            #         cv.imshow("left_eye", left_eye)
            #         # cv.imwrite("le/le_" + str(count) + ".jpg", left_eye)

            for key in right_eye_main_dict:
                right_eye_main.append(right_eye_main_dict[key])

            for key in left_eye_main_dict:
                left_eye_main.append(left_eye_main_dict[key])

            right_eye_main = np.array(right_eye_main, np.int32)
            right_eye_main = right_eye_main.reshape((-1, 1, 2))

            left_eye_main = np.array(left_eye_main, np.int32)
            left_eye_main = left_eye_main.reshape((-1, 1, 2))

            isClosed = True
            color = (255, 0, 0)
            thickness = 2

            img = cv.polylines(img, [right_eye_main], isClosed, color, thickness)
            img = cv.polylines(img, [left_eye_main], isClosed, color, thickness)

            # if bounds_right[0] is not None:
            #     flag_re = True
            #
            # if bounds_left[0] is not None:
            #     flag_le = True
            #
            # if flag_re is True:
            #     right_eye = img_for_eye[bounds_right[0][1]:bounds_right[1][1], bounds_right[2][0]:bounds_right[3][0]]
            #
            # if flag_le is True:
            #     left_eye = img_for_eye[bounds_left[0][1]:bounds_left[1][1], bounds_left[2][0]:bounds_left[3][0]]

            for key in regions_re.keys():
                temp = regions_re[key]
                temp_coords = []

                for index in temp:
                    temp_coords.append(right_eye_main_dict[index])

                regions_re[key] = temp_coords

            for key in regions_le.keys():
                temp = regions_le[key]
                temp_coords = []

                for index in temp:
                    temp_coords.append(left_eye_main_dict[index])

                regions_le[key] = temp_coords

            for key in right_upper_lobe.keys():
                temp = right_upper_lobe[key]
                temp_coords = []

                for index in temp:
                    if index in r_lobe_coords.keys():
                        temp_coords.append(r_lobe_coords[index])

                    else:
                        temp_coords.append(right_eye_main_dict[index])

                right_upper_lobe[key] = temp_coords

            for key in right_lower_lobe.keys():
                temp = right_lower_lobe[key]
                temp_coords = []

                for index in temp:
                    if index in r_lobe_coords.keys():
                        temp_coords.append(r_lobe_coords[index])

                    else:
                        temp_coords.append(right_eye_main_dict[index])

                right_lower_lobe[key] = temp_coords

            for key in left_upper_lobe.keys():
                temp = left_upper_lobe[key]
                temp_coords = []

                for index in temp:
                    if index in l_lobe_coords.keys():
                        temp_coords.append(l_lobe_coords[index])

                    else:
                        temp_coords.append(left_eye_main_dict[index])

                left_upper_lobe[key] = temp_coords

            for key in left_lower_lobe.keys():
                temp = left_lower_lobe[key]
                temp_coords = []

                for index in temp:
                    if index in l_lobe_coords.keys():
                        temp_coords.append(l_lobe_coords[index])

                    else:
                        temp_coords.append(left_eye_main_dict[index])

                left_lower_lobe[key] = temp_coords

            print("regions_re:", regions_re)
            print("regions_le:", regions_le)
            print("right_upper_lobe:", right_upper_lobe)
            print("right_lower_lobe:", right_lower_lobe)
            print("left_upper_lobe:", left_upper_lobe)
            print("left_lower_lobe:", left_lower_lobe)

            rect_right = cv.boundingRect(right_eye_main)
            x, y, w, h = rect_right
            cropped_re = img[y:y + h, x:x + w].copy()
            right_eye_main = right_eye_main - right_eye_main.min(axis=0)
            mask_re = np.zeros(cropped_re.shape[:2], np.uint8)
            cv.drawContours(mask_re, [right_eye_main], -1, (255, 255, 255), -1, cv.LINE_AA)
            dst_re = cv.bitwise_and(cropped_re, cropped_re, mask=mask_re)
            bg = np.ones_like(cropped_re, np.uint8) * 255
            cv.bitwise_not(bg, bg, mask=mask_re)
            dst_re = bg + dst_re

            for key in regions_re.keys():
                temp = regions_re[key]
                temp_t = []

                for i in range(len(temp)):
                    temp_t.append([temp[i][0] - x, temp[i][1] - y])

                regions_re[key] = temp_t

            for key in right_upper_lobe.keys():
                temp = right_upper_lobe[key]
                temp_t = []

                for i in range(len(temp)):
                    temp_t.append([temp[i][0] - x, temp[i][1] - y])

                right_upper_lobe[key] = temp_t

            for key in right_lower_lobe.keys():
                temp = right_lower_lobe[key]
                temp_t = []

                for i in range(len(temp)):
                    temp_t.append([temp[i][0] - x, temp[i][1] - y])

                right_lower_lobe[key] = temp_t

            rect_left = cv.boundingRect(left_eye_main)
            x, y, w, h = rect_left
            cropped_le = img[y:y + h, x:x + w].copy()
            left_eye_main = left_eye_main - left_eye_main.min(axis=0)
            mask_le = np.zeros(cropped_le.shape[:2], np.uint8)
            cv.drawContours(mask_le, [left_eye_main], -1, (255, 255, 255), -1, cv.LINE_AA)
            dst_le = cv.bitwise_and(cropped_le, cropped_le, mask=mask_le)
            bg = np.ones_like(cropped_le, np.uint8) * 255
            cv.bitwise_not(bg, bg, mask=mask_le)
            dst_le = bg + dst_le

            for key in regions_le.keys():
                temp = regions_le[key]
                temp_t = []

                for i in range(len(temp)):
                    temp_t.append([temp[i][0] - x, temp[i][1] - y])

                regions_le[key] = temp_t

            for key in left_upper_lobe.keys():
                temp = left_upper_lobe[key]
                temp_t = []

                for i in range(len(temp)):
                    temp_t.append([temp[i][0] - x, temp[i][1] - y])

                left_upper_lobe[key] = temp_t

            for key in left_lower_lobe.keys():
                temp = left_lower_lobe[key]
                temp_t = []

                for i in range(len(temp)):
                    temp_t.append([temp[i][0] - x, temp[i][1] - y])

                left_lower_lobe[key] = temp_t

            kernel = np.array([[0, -1, 0], [-1, 6.5, -1], [0, -1, 0]])

            thresh = 0

            dst_re = cv.cvtColor(dst_re, cv.COLOR_BGR2GRAY)
            # dst_re = cv.threshold(dst_re, thresh, 255, cv.THRESH_BINARY)[1]
            dst_re = cv.filter2D(dst_re, -1, kernel)
            dst_re = cv.medianBlur(dst_re, 5)
            # dst_re = cv.threshold(dst_re, thresh, 255, cv.THRESH_BINARY)[1]
            # dst_re = cv.Canny(image=dst_re, threshold1=50, threshold2=200)

            dst_le = cv.cvtColor(dst_le, cv.COLOR_BGR2GRAY)
            dst_le = cv.filter2D(dst_le, -1, kernel)
            dst_le = cv.medianBlur(dst_le, 5)
            # dst_le = cv.threshold(dst_le, thresh, 255, cv.THRESH_BINARY)[1]
            # dst_le = cv.Sobel(src=dst_le, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
            # dst_le = cv.Canny(image=dst_le, threshold1=50, threshold2=200)

            ar_re = []
            ar_le = []

            for key in regions_re.keys():
                # mask_re = np.zeros_like(dst_re)
                #
                list_coords = np.array(regions_re[key])
                # # print("list_coords_re:", list_coords)
                # cv.fillPoly(mask_re, [list_coords], 255)
                # mask_re = cv.threshold(mask_re, 200, 255, cv.THRESH_BINARY)[1]
                # masked_image = cv.bitwise_and(dst_re, mask_re)
                masked_image = dst_re

                # black_pixel_count_re = 0
                # white_pixel_count_re = 0

                pixel_count = 0
                values = 0

                for y in range(masked_image.shape[0]):
                    for x in range(masked_image.shape[1]):
                        if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
                            pixel_count += 1
                            values += masked_image[y, x]

                            # if masked_image[y, x] < 20:
                            #     black_pixel_count_re += 1
                            #
                            # else:
                            #     white_pixel_count_re += 1

                # percents_re.append(round(black_pixel_count_re / (black_pixel_count_re + white_pixel_count_re), 2))
                percents_re.append(round((values / pixel_count), 2))
                text_temp = ":" + str(pixel_count) + ":"
                ar_re.append(text_temp)

            for key in regions_le.keys():
                # mask_le = np.zeros_like(dst_le)
                #
                list_coords = np.array(regions_le[key])
                # # print("list_coords_le:", list_coords)
                # cv.fillPoly(mask_le, [list_coords], 255)
                # # cv.drawContours(mask_le, [list_coords], -1, 255, 1, thickness=cv.LINE_AA)
                # # mask_le = cv.threshold(mask_le, 200, 255, cv.THRESH_BINARY)[1]
                # masked_image = cv.bitwise_and(dst_le, mask_le)
                masked_image = dst_le

                # black_pixel_count_le = 0
                # white_pixel_count_le = 0

                pixel_count = 0
                values = 0

                for y in range(masked_image.shape[0]):
                    for x in range(masked_image.shape[1]):
                        if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
                            pixel_count += 1
                            values += masked_image[y, x]

                            # if masked_image[y, x] < 20:
                            #     black_pixel_count_le += 1
                            #
                            # else:
                            #     white_pixel_count_le += 1

                # percents_le.append(round(black_pixel_count_le / (black_pixel_count_le + white_pixel_count_le), 2))
                percents_le.append(round((values / pixel_count), 2))
                text_temp = ":" + str(pixel_count) + ":"
                ar_le.append(text_temp)

            index = 0
            for key in right_upper_lobe.keys():
                list_coords = np.array(right_upper_lobe[key])
                masked_image = dst_re
                pixel_count = 0

                for y in range(masked_image.shape[0]):
                    for x in range(masked_image.shape[1]):
                        if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
                            pixel_count += 1

                ar_re[index] = str(pixel_count) + ar_re[index]
                index += 1

            index = 0
            for key in right_lower_lobe.keys():
                list_coords = np.array(right_lower_lobe[key])
                masked_image = dst_re
                pixel_count = 0

                for y in range(masked_image.shape[0]):
                    for x in range(masked_image.shape[1]):
                        if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
                            pixel_count += 1

                ar_re[index] = ar_re[index] + str(pixel_count)
                index += 1

            index = 0
            for key in left_upper_lobe.keys():
                list_coords = np.array(left_upper_lobe[key])
                masked_image = dst_re
                pixel_count = 0

                for y in range(masked_image.shape[0]):
                    for x in range(masked_image.shape[1]):
                        if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
                            pixel_count += 1

                ar_le[index] = str(pixel_count) + ar_le[index]
                index += 1

            index = 0
            for key in left_lower_lobe.keys():
                list_coords = np.array(left_lower_lobe[key])
                masked_image = dst_re
                pixel_count = 0

                for y in range(masked_image.shape[0]):
                    for x in range(masked_image.shape[1]):
                        if cv.pointPolygonTest(list_coords, (x, y), False) >= 0:
                            pixel_count += 1

                ar_le[index] = ar_le[index] + str(pixel_count)
                index += 1

            print("percents_re:", percents_re)
            print("percents_le:", percents_le)

            # contours, _ = cv.findContours(dst_le, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
            # contour_image = np.zeros_like(dst_le)
            # cv.drawContours(contour_image, contours, -1, (255, 255, 255), 2)  # Draw all contours with thickness 2
            # # print("_:", _)

            print("cropped shapes:", dst_re.shape, dst_le.shape)

            resized_re = cv.resize(dst_re, (128, 32), interpolation=cv.INTER_LINEAR)
            resized_le = cv.resize(dst_le, (128, 32), interpolation=cv.INTER_LINEAR)

            # cv.imshow("dst_re", dst_re)
            # cv.imshow("dst_le", dst_le)

            otsu_threshold, resized_re_OTSU = cv.threshold(
                resized_re, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
            )
            otsu_threshold_2, resized_le_OTSU = cv.threshold(
                resized_le, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU,
            )

            resized_re_t = np.expand_dims(resized_re, axis=-1)
            resized_le_t = np.expand_dims(resized_le, axis=-1)

            resized_re_OTSU = np.expand_dims(resized_re_OTSU, axis=-1)
            resized_le_OTSU = np.expand_dims(resized_le_OTSU, axis=-1)

            resized_re_t = np.concatenate((resized_re_t, resized_re_OTSU), axis=-1)
            resized_le_t = np.concatenate((resized_le_t, resized_le_OTSU), axis=-1)

            # Repetition needs to be changed
            resized_re_KMeans = KMeans_Image(resized_re)
            resized_le_KMeans = KMeans_Image(resized_le)

            resized_re_KMeans = np.expand_dims(resized_re_KMeans, axis=-1)
            resized_le_KMeans = np.expand_dims(resized_le_KMeans, axis=-1)

            resized_re_t = np.concatenate((resized_re_t, resized_re_KMeans), axis=-1)
            resized_le_t = np.concatenate((resized_le_t, resized_le_KMeans), axis=-1)

            print("cropped shapes:", resized_re.shape, resized_le.shape)
            print("cropped shapes in _t:", resized_re_t.shape, resized_le_t.shape)

            # TODO: Do image subtraction
            cv.imshow("right_eye cropped", resized_re_t)
            cv.imshow("left_eye cropped", resized_le_t)

            # TODO: Uncomment this for subtraction
            # if resized_re_t_before is not None:
            #     resized_re_t_sub = cv.subtract(resized_re_t, resized_re_t_before)
            #     cv.imshow("right_eye subtracted", resized_re_t_sub)
            # else:
            #     resized_re_t_before = resized_re_t
            #
            # if resized_le_t_before is not None:
            #     resized_le_t_sub = cv.subtract(resized_le_t, resized_le_t_before)
            #     cv.imshow("left_eye subtracted", resized_le_t_sub)
            # else:
            #     resized_le_t_before = resized_le_t

            reward = 0

            # Subtraction for OTSU binarization
            if resized_re_OTSU_before is not None:
                resized_re_OTSU_sub = cv.subtract(resized_re_OTSU, resized_re_OTSU_before)
                resized_re_OTSU_sub = np.where(resized_re_OTSU_sub > 0, 255, 0).astype(np.uint8)

                wp_c = np.sum(resized_re_OTSU_sub == 255)
                r_w = wp_c / (32 * 128)
                r_b = 1 - r_w
                reward += r_b

                # rows, cols = np.indices(resized_re_OTSU_sub.shape)
                #
                # # Flatten the arrays for easier computation
                # rows = rows.flatten()
                # cols = cols.flatten()
                # weights = resized_re_OTSU_sub.flatten()
                #
                # # Calculate the weighted mean of the coordinates
                # mean_y = np.average(rows, weights=weights)
                # mean_x = np.average(cols, weights=weights)
                #
                # mean_point = (int(mean_x), int(mean_y))
                # print("mean_point from OTSU re:", mean_point)

                cv.imshow("right_eye OTSU subtracted", resized_re_OTSU_sub)
                resized_re_OTSU_before = resized_re_OTSU
            else:
                resized_re_OTSU_before = resized_re_OTSU

            if resized_le_OTSU_before is not None:
                resized_le_OTSU_sub = cv.subtract(resized_le_OTSU, resized_le_OTSU_before)
                resized_le_OTSU_sub = np.where(resized_le_OTSU_sub > 0, 255, 0).astype(np.uint8)

                wp_c = np.sum(resized_le_OTSU_sub == 255)
                r_w = wp_c / (32 * 128)
                r_b = 1 - r_w
                reward += r_b

                cv.imshow("left_eye OTSU subtracted", resized_le_OTSU_sub)
                resized_le_OTSU_before = resized_le_OTSU
            else:
                resized_le_OTSU_before = resized_le_OTSU

            # # Subtraction for KMeans
            # if resized_re_KMeans_before is not None:
            #     resized_re_KMeans_sub = cv.subtract(resized_re_KMeans, resized_re_KMeans_before)
            #     resized_re_KMeans_sub = np.where(resized_re_KMeans_sub > 0, 255, 0).astype(np.uint8)
            #
            #     cv.imshow("right_eye KMeans subtracted", resized_re_KMeans_sub)
            #     resized_re_KMeans_before = resized_re_KMeans
            # else:
            #     resized_re_KMeans_before = resized_re_KMeans
            #
            # if resized_le_KMeans_before is not None:
            #     resized_le_KMeans_sub = cv.subtract(resized_le_KMeans, resized_le_KMeans_before)
            #     resized_le_KMeans_sub = np.where(resized_le_KMeans_sub > 0, 255, 0).astype(np.uint8)
            #
            #     cv.imshow("left_eye KMeans subtracted", resized_le_KMeans_sub)
            #     resized_le_KMeans_before = resized_le_KMeans
            # else:
            #     resized_le_KMeans_before = resized_le_KMeans

            # cv.imshow("left_eye contoured", contour_image)

            # right_eye_video.append(resized_re)
            # left_eye_video.append(resized_le)
            #
            # # Display cropped image
            #
            # if (right_eye is not None) and (flag_re is True):
            #     right_eye = cv.cvtColor(right_eye, cv.COLOR_BGR2GRAY)
            #
            #     # right_eye = cv.GaussianBlur(right_eye, (7, 7), 0)
            #     # kernel = np.array([[0, -1, 0],
            #     #                    [-1, 5, -1],
            #     #                    [0, -1, 0]])
            #     # right_eye = cv.filter2D(right_eye, -1, kernel)
            #
            #     ratio_h_w = right_eye.shape[0] / right_eye.shape[1]
            #     size_1 = (200, int(ratio_h_w * 200))
            #     size_2 = (int(100 / ratio_h_w), 100)
            #
            #     # if size_1[1] <= 100:
            #     #     for i in range(100 - size_1[1]):
            #     #         pass
            #     #
            #     # else:
            #     #     pass
            #
            #     temp = min(100, max(int(200 * (right_eye.shape[0] / right_eye.shape[1])), 100))
            #     right_eye = cv.resize(right_eye, (200, temp),
            #                           interpolation=cv.INTER_LINEAR)
            #
            #     # print(right_eye.shape)
            #     right_eye = np.array(right_eye)
            #     newrow = np.zeros((1, 200), dtype=int)
            #
            # if (left_eye is not None) and (flag_le is True):
            #     left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)
            #
            #     ratio_h_w = right_eye.shape[0] / right_eye.shape[1]
            #
            #     # (thresh, im_bw) = cv.threshold(left_eye, 200, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            #
            #     size_1 = (200, int(ratio_h_w * 200))
            #     size_2 = (int(100 / ratio_h_w), 100)
            #
            #     temp = min(100, max(int(200 * (left_eye.shape[0] / left_eye.shape[1])), 100))
            #     left_eye = cv.resize(left_eye, (200, temp),
            #                          interpolation=cv.INTER_LINEAR)
            #
            #     newrow = np.zeros((1, 200), dtype=int)
            #     left_eye = left_eye / 255
            #
            #     # left_eye_temp = np.expand_dims(left_eye, axis=-1)
            #     left_eye_temp = np.array(left_eye)
            #     # left_eye_temp = (left_eye_temp * 255).astype(np.uint8)
            #     left_eye_temp = left_eye_temp.flatten()
            #     left_eye_temp = left_eye_temp.reshape((-1, 200, 100, 1))
            #
            # angle_right = None
            # angle_left = None
            #
            # # if (focus_points[4] is not None) and (focus_points[0] is not None):
            # #     angle_right = calculate_angle(focus_points[0], focus_points[4], [focus_points[4][0], focus_points[4][1] - 0.001])
            # #
            # # if(focus_points[4] is not None) and (focus_points[2] is not None):
            # #     angle_left = calculate_angle([focus_points[4][0], focus_points[4][1] - 0.001], focus_points[4], focus_points[2])
            #
            # # print(angle_right, angle_left)
            #
            # cTime = time.time()
            # fps = 1 / (cTime - pTime)
            # pTime = cTime
            #
            # # cv.putText(img, str(np.round_([motion[0], motion[2], motion[4]], decimals=3)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            # # cv.putText(img, str(count), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            # # cv.imshow("Image", img)
            # # cv.waitKey(1)
            #
            # temp_predict = []
            #
            # # rows.append([count, 'O'])
            # temp_final = [(100 * (iter_count - 1)) + count]
            # for i in percents_re:
            #     temp_final.append(round((i / 256), 5))
            #     temp_predict.append(round((i / 256), 5))
            # for i in ar_re:
            #     temp_final.append(i)
            # temp_final.append(angle_right)
            #
            # for i in percents_le:
            #     temp_final.append(round((i / 256), 5))
            #     temp_predict.append(round((i / 256), 5))
            # for i in ar_le:
            #     temp_final.append(i)
            # temp_final.append(angle_left)
            #
            # temp_predict.append(angle_right)
            # temp_predict.append(angle_left)
            #
            # # count += 1
            # # radius = 50
            # #
            # # if len(re_points) == 2:
            # #     re = np.linalg.norm(np.array(re_points[0]) - np.array(re_points[1]))
            # #
            # # if len(le_points) == 2:
            # #     le = np.linalg.norm(np.array(le_points[0]) - np.array(le_points[1]))
            # #
            # # if p_195 is not None:
            # #     num = np.linalg.norm(np.array(p_195) - np.array(nose))
            # #
            # # if p_94 is not None:
            # #     denom = np.linalg.norm(np.array(nose) - np.array(p_94))
            # #
            # # re_part = re / (re + le)
            # # num_part = num / (num + denom)
            # # a = int(min(radius * re_part, radius * (1 - re_part)))
            # # b = int(min(radius * num_part, radius * (1 - num_part)))
            # #
            # # print(a, b)
            # # cv.ellipse(img, (nose[0], nose[1]), (a, b), 0, 0, 360, (0, 255, 0), -1)

            reward /= 2
            print("reward from the eyes:", reward)

            # TODO: Uncomment this for inference
            if eye_crop_window != 0:
                eye_right_cropped_model_input.append(resized_re)
                eye_left_cropped_model_input.append(resized_le)

                print("eye_crop_window:", eye_crop_window)

                eye_crop_window -= 1

            else:
                eye_crop_window = 32
                eye_right_cropped_model_input = np.array(eye_right_cropped_model_input)
                eye_left_cropped_model_input = np.array(eye_left_cropped_model_input)

                print("eye_right_cropped_model_input:", eye_right_cropped_model_input.shape)
                print("eye_left_cropped_model_input:", eye_left_cropped_model_input.shape)

                last_layer_from_upper_branch, softmaxed_layer = EyeVidPre_and_ViViT.PreProcessandFreezedOutput(
                    eye_right_cropped_model_input, eye_left_cropped_model_input)
                print("last_layer_from_upper_branch:", last_layer_from_upper_branch)

                eye_right_cropped_model_input = []
                eye_left_cropped_model_input = []

            cv.imshow("img", img)
            print("shape of image:", img.shape)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    iter_count += 1

cap.release()
cv.destroyAllWindows()

# print("right_eye_video:", right_eye_video)
# print("left_eye_video:", left_eye_video)

print("time taken:", time.time() - time_overall)
