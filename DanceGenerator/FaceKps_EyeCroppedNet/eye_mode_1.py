# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize MediaPipe FaceMesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # Enables iris tracking

# # Indices for eye & iris keypoints from FaceMesh
# LEFT_IRIS = [474, 475, 476, 477]  # Center of left iris
# LEFT_EYE = [33, 133]  # Outer and inner corners of left eye
# LEFT_FULL_EYE = [249, 263, 263, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]

# RIGHT_IRIS = [469, 470, 471, 472]  # Center of right iris
# RIGHT_EYE = [362, 263]  # Outer and inner corners of right eye
# RIGHT_FULL_EYE = [7, 33, 33, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]

# # Capture video
# cap = cv2.VideoCapture(0)

# def draw_landmarks(image, landmarks, indices, color):
#     """Draw small circles at given landmark indices."""
#     for i in indices:
#         x, y = int(landmarks[i][0] * image.shape[1]), int(landmarks[i][1] * image.shape[0])
#         cv2.circle(image, (x, y), 2, color, -1)

# def get_iris_position(landmarks, eye_points, iris_points):
#     """Calculate relative iris position between inner and outer eye corners."""
#     eye_center_x = (landmarks[eye_points[0]][0] + landmarks[eye_points[1]][0]) / 2
#     iris_center_x = sum([landmarks[i][0] for i in iris_points]) / len(iris_points)
#     return iris_center_x - eye_center_x  # Positive → Right, Negative → Left

# def determine_gaze(left_offset, right_offset, threshold=0.01):
#     """Determine gaze direction based on iris position."""
#     avg_offset = (left_offset + right_offset) / 2
#     if avg_offset < -threshold:
#         return "Looking Left"
#     elif avg_offset > threshold:
#         return "Looking Right"
#     else:
#         return "Looking Center"

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to RGB for MediaPipe
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(frame_rgb)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Convert landmarks into a dictionary
#             landmarks = {i: (lm.x, lm.y) for i, lm in enumerate(face_landmarks.landmark)}

#             # Draw eye landmarks (yellow) and iris landmarks (blue)
#             # draw_landmarks(frame, landmarks, LEFT_EYE + RIGHT_EYE, (0, 255, 255))  # Yellow
#             # draw_landmarks(frame, landmarks, LEFT_IRIS + RIGHT_IRIS, (255, 0, 0))  # Blue

#             draw_landmarks(frame, landmarks, RIGHT_FULL_EYE, (0, 0, 255))  # Red
#             draw_landmarks(frame, landmarks, LEFT_FULL_EYE, (0, 255, 0))  # Red

#             # Compute iris offsets
#             left_offset = get_iris_position(landmarks, LEFT_EYE, LEFT_IRIS)
#             right_offset = get_iris_position(landmarks, RIGHT_EYE, RIGHT_IRIS)

#             # Determine gaze
#             gaze_direction = determine_gaze(left_offset, right_offset)

#             # Display text on frame
#             cv2.putText(frame, f"Gaze: {gaze_direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     # Show frame
#     cv2.imshow('Iris Tracking with Landmarks', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np

import matplotlib.pyplot as plt
import time

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # Enables iris tracking

# Indices for eye keypoints from FaceMesh
LEFT_FULL_EYE = [249, 263, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
RIGHT_FULL_EYE = [7, 33, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]

RIGHT_EYE_REGIONS = [[7, 33, 246], [7, 246, 161, 163], [163, 161, 160, 144], [144, 160, 159, 145], [145, 159, 158, 153], [153, 158, 157, 154], [154, 157, 173, 155]]
#     246, 161, 160, 159, 158, 157, 173
# 33 I   II   III  IV    V   VI   VII
#      7,  163, 144, 145, 153, 154, 155

LEFT_EYE_REGIONS = [[149, 263, 466], [149, 466, 388, 390], [390, 388, 387, 373], [373, 387, 386, 374], [374, 386, 385, 380], [380, 385, 384, 381], [381, 384, 398, 382]]
# 398, 384, 385, 386, 387, 388, 466
#    VII   VI    V   IV  III   II   I 263
# 382, 381, 380, 374, 373, 390, 149

# Capture video
cap = cv2.VideoCapture(0)

def crop_eye_region(image, landmarks, eye_indices):
    points = np.array([(int(landmarks[i][0] * image.shape[1]), int(landmarks[i][1] * image.shape[0])) for i in eye_indices])

    x, y, w, h = cv2.boundingRect(points)

    return y, x, image[y:y+h, x:x+w] if w > 0 and h > 0 else None

def eye_coars_loc(cropped_eye, rgn_bdrs, time_step):
    region_avgs = []
    cropped_eye = cv2.cvtColor(cropped_eye, cv2.COLOR_BGR2GRAY)

    # print("time_step, rgn_bdrs:", time_step, rgn_bdrs)
    # print("time_step, cropped_eye shape:", time_step, cropped_eye.shape)

    # creating the masks
    for rgn_idx, region in enumerate(rgn_bdrs):
        region = np.array(region, dtype=np.int32)
        region = region.reshape((-1, 1, 2))
        region_number = rgn_idx + 1

        # print("rgn_idx, time_step, region shape:", region_number, time_step, region.shape)

        mask = np.zeros_like(cropped_eye, dtype=np.uint8)
        cv2.fillPoly(mask, np.array(region), 255)

        y_coords, x_coords = np.where(mask == 255)

        pxl_vals = [cropped_eye[y, x] for x, y in zip(x_coords, y_coords)]
        pxl_avg = np.mean(pxl_vals, axis=0)
        # print("time_step, pxl_avg:", time_step, pxl_avg)

        region_avgs.append(pxl_avg)

        # masked = cv2.bitwise_and(cropped_eye, mask)
    
    # print("time_step, region_avgs:", time_step, region_avgs)

    region_avgs = np.array(region_avgs)
    mean = np.mean(region_avgs)
    std = np.std(region_avgs)
    region_avgs_norm = (region_avgs - mean) / std

    return region_avgs, region_avgs_norm

# per frame processing for both eyes; frames taken as input
def eye_mode_1(frame, landmarks, frame_step, coarse_loc_type='std_norm', cropped=False, lec=None, rec=None):
    eye_mode_1_start_time = time.time()

    '''
    frame: takes the complete frame as input
    landmarks: takes the whole dictionary of landmarks as input
    frame_step: takes the index of the frame step during the iteration (1 to 60)
    coarse_loc_type: mode of giving the output of the coarse location of eye, either averaged or corrected to standard deviation
    cropped: boolean to decide the input, whether to take the complete frame as input or just the cropped eye (will require croppping during the main process)
    lec: left eye cropped as input if cropped to be True, otherwise None
    rec: right eye cropped as input if cropped to be True, otherwise None
    '''
    print("eneterred the function to encode the eye-cropped frames...")
    print("enterred with:")
    print("frame_step:", frame_step)
    print("coarse_loc_type:", coarse_loc_type)
    print("----------")

    xr, yr, xl, yl = 0, 0, 0, 0
    left_eye_crop = lec
    right_eye_crop = rec

    if not cropped:
        print("inside the condition where the input frames are cropped")

        # Crop left and right eye
        yl, xl, left_eye_crop = crop_eye_region(frame, landmarks, LEFT_FULL_EYE)
        yr, xr, right_eye_crop = crop_eye_region(frame, landmarks, RIGHT_FULL_EYE)

    right_eye_region_coords = [[[int(landmarks[point][0] * frame.shape[1]) - xr, int(landmarks[point][1] * frame.shape[0]) - yr] for point in rgn] for rgn in RIGHT_EYE_REGIONS]
    left_eye_region_coords = [[[int(landmarks[point][0] * frame.shape[1]) - xl, int(landmarks[point][1] * frame.shape[0]) - yl] for point in rgn] for rgn in LEFT_EYE_REGIONS]

    # right_eye_region_coords = [[[int(landmarks[point][0] * frame.shape[1]), int(landmarks[point][1] * frame.shape[0])] for point in rgn] for rgn in RIGHT_EYE_REGIONS]
    # left_eye_region_coords = [[[int(landmarks[point][0] * frame.shape[1]), int(landmarks[point][1] * frame.shape[0])] for point in rgn] for rgn in LEFT_EYE_REGIONS]

    left_regions_avg, left_regions_avg_norm = eye_coars_loc(left_eye_crop, left_eye_region_coords, frame_step)
    right_regions_avg, right_regions_avg_norm = eye_coars_loc(right_eye_crop, right_eye_region_coords, frame_step)

    # left_regions_avg = eye_coars_loc(frame, left_eye_region_coords, time_steps)
    # right_regions_avg = eye_coars_loc(frame, right_eye_region_coords, time_steps)

    eye_mode_1_end_time = time.time()
    print("time taken for the eye_mode_1 function to complete:", eye_mode_1_end_time - eye_mode_1_start_time)

    if coarse_loc_type == 'avg':
        return right_regions_avg, left_regions_avg
    elif coarse_loc_type == 'std_norm':
        return right_regions_avg_norm, left_regions_avg_norm
    else:
        return -1

def run_isolated():
    time_steps = 180
    left_rgn_avgs = []
    right_rgn_avgs = []

    left_rgn_avgs_norm = []
    right_rgn_avgs_norm = []

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]  # Process only the first detected face
            landmarks = {i: (lm.x, lm.y) for i, lm in enumerate(face_landmarks.landmark)}

            # Crop left and right eye
            yl, xl, left_eye_crop = crop_eye_region(frame, landmarks, LEFT_FULL_EYE)
            yr, xr, right_eye_crop = crop_eye_region(frame, landmarks, RIGHT_FULL_EYE)

            right_eye_region_coords = [[[int(landmarks[point][0] * frame.shape[1]) - xr, int(landmarks[point][1] * frame.shape[0]) - yr] for point in rgn] for rgn in RIGHT_EYE_REGIONS]
            left_eye_region_coords = [[[int(landmarks[point][0] * frame.shape[1]) - xl, int(landmarks[point][1] * frame.shape[0]) - yl] for point in rgn] for rgn in LEFT_EYE_REGIONS]

            # right_eye_region_coords = [[[int(landmarks[point][0] * frame.shape[1]), int(landmarks[point][1] * frame.shape[0])] for point in rgn] for rgn in RIGHT_EYE_REGIONS]
            # left_eye_region_coords = [[[int(landmarks[point][0] * frame.shape[1]), int(landmarks[point][1] * frame.shape[0])] for point in rgn] for rgn in LEFT_EYE_REGIONS]

            left_regions_avg, left_regions_avg_norm = eye_coars_loc(left_eye_crop, left_eye_region_coords, time_steps)
            right_regions_avg, right_regions_avg_norm = eye_coars_loc(right_eye_crop, right_eye_region_coords, time_steps)

            # left_regions_avg = eye_coars_loc(frame, left_eye_region_coords, time_steps)
            # right_regions_avg = eye_coars_loc(frame, right_eye_region_coords, time_steps)

            left_rgn_avgs.append(left_regions_avg)
            right_rgn_avgs.append(right_regions_avg)

            left_rgn_avgs_norm.append(left_regions_avg_norm)
            right_rgn_avgs_norm.append(right_regions_avg_norm)

            # Display cropped eyes in separate windows
            if left_eye_crop is not None:
                cv2.imshow("Left Eye", left_eye_crop)

            if right_eye_crop is not None:
                cv2.imshow("Right Eye", right_eye_crop)
            
            time_steps -= 1

        # Show the original frame
        cv2.imshow('FaceMesh Eye Tracking', frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or (time_steps == 0):
            break

    end_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

    print("time taken:", end_time - start_time)

    # Create a figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # First plot
    axes[0].imshow(left_rgn_avgs, cmap='gray')
    axes[0].set_title("left_rgn_avgs")
    axes[0].axis("off")  # Hide axes

    # Second plot
    axes[1].imshow(left_rgn_avgs_norm, cmap='gray')
    axes[1].set_title("left_rgn_avgs_norm")
    axes[1].axis("off")  # Hide axes

    plt.show()

# run_isolated()