import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from enum import IntEnum


# class Joint(IntEnum):
#     NOSE = 0
#     NECK = 1
#     RSHOULDER = 2
#     RELBOW = 3
#     RWRIST = 4
#     LSHOULDER = 5
#     LELBOW = 6
#     LWRIST = 7
#     MIDHIP = 8
#     RHIP = 9
#     RKNEE = 10
#     RANKLE = 11
#     LHIP = 12
#     LKNEE = 13
#     LANKLE = 14
#     REYE = 15
#     LEYE = 16
#     REAR = 17
#     LEAR = 18
#     LBIGTOE = 19
#     LSMALLTOE = 20
#     LHEEL = 21
#     RBIGTOE = 22
#     RSMALLTOE = 23
#     RHEEL = 24


class Joint(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Main scoring code
class DanceScorer:
    # Range values for the min-max joint angles
    RANGE = {
                "lshoulder" : 3.099920992,
                "rShoulder" : 3.115298634,
                "lelbow" : 3.139355155,
                "relbow" : 3.140255528,
                "lhip" : 2.306497931,
                "rhip" : 2.352498353,
                "lknee" : 2.422342539,
                "rknee" : 2.163526058,
                "lankle" : 2.097560167,
                "rankle" : 2.271983564
            }
    SIGMA_SCALE = 12

    def __init__(self):

        # Instantiate two lists to store the teacher and student poses
        self.poses = {
            "student" : [],
            "teacher" : []
        }


        # Each element in this dictionary is a list of length n storing the tracked metrics
        self.position_metrics = {
            "student" : {
                "lshoulder" : [],
                "rShoulder" : [],
                "lelbow" : [],
                "relbow" : [],
                "lhip" : [],
                "rhip" : [],
                "lknee" : [],
                "rknee" : [],
                "lankle" : [],
                "rankle" : []
            },
            "teacher" : {
                "lshoulder" : [],
                "rShoulder" : [],
                "lelbow" : [],
                "relbow" : [],
                "lhip" : [],
                "rhip" : [],
                "lknee" : [],
                "rknee" : [],
                "lankle" : [],
                "rankle" : []
            }
        }

        # Each element in this dictionary is a list of length n-1
        # These are all "first derviative" metrics like velocity
        self.velocity_metrics = {
            "student" : {
                "lshoulder" : [],
                "rShoulder" : [],
                "lelbow" : [],
                "relbow" : [],
                "lhip" : [],
                "rhip" : [],
                "lknee" : [],
                "rknee" : [],
                "lankle" : [],
                "rankle" : []
            },
            "teacher" : {
                "lshoulder" : [],
                "rShoulder" : [],
                "lelbow" : [],
                "relbow" : [],
                "lhip" : [],
                "rhip" : [],
                "lknee" : [],
                "rknee" : [],
                "lankle" : [],
                "rankle" : []
            }
        }

    def _calc_angle(self, joint, start_joint, end_joint):

        if joint[2]< 0.1 or start_joint[2]<0.1 or end_joint[2]<0.1:
            return -1

        # Calculate two vectors that form joint
        v1 = start_joint[0:2] - joint[0:2]
        v2 = end_joint[0:2] - joint[0:2]

        # Calc dot product
        dot_prod = np.dot(v1,v2)

        # Calculate magnitudes
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)

        # Calculate angle
        if dot_prod/v1_mag/v2_mag > 1.:
            return np.arccos(1.)
        elif dot_prod/v1_mag/v2_mag < -1.:
            return np.arccos(-1.)
        return np.arccos(dot_prod/v1_mag/v2_mag)

    def _calc_velocity(self, prev_joint, cur_joint):

        if prev_joint[2]<0.1 or cur_joint[2]<0.1:
            return -1

        v1 = prev_joint[0:2]
        v2 = cur_joint[0:2]

        return np.linalg.norm(v2-v1)



    # Joint we are considering
    # For each of these, we calculate the angle and velocity
    # Left Shoulder
    # Left Elbow
    # Left Hip
    # Left Knee
    # Left Ankle
    # Right Shoulder
    # Right Elbow
    # Right Hip
    # Right Knee
    # Right Ankle

    def _calc_dance_metrics(self, dancer):
        # select data
        if(dancer != "student" and dancer != "teacher"):
            raise Exception("Selected dancer must be a student or teacher")

        # Create numpy arrays of the right length
        for joint in self.position_metrics[dancer]:
            self.position_metrics[dancer][joint] = np.zeros(shape = (len(self.poses[dancer]), ), dtype = np.float32)

            if len(self.poses[dancer]) == 0:
                self.velocity_metrics[dancer][joint] = np.zeros(shape = (len(self.poses[dancer]), ), dtype = np.float32)
            else:
                self.velocity_metrics[dancer][joint] = np.zeros(shape = (len(self.poses[dancer]), ), dtype = np.float32)


        for i, pose in enumerate(self.poses[dancer]):
            # joint_angle_args = {
            #     "lshoulder" : [pose[0,Joint.LSHOULDER,:], pose[0,Joint.NECK,:], pose[0,Joint.LELBOW,:]],
            #     "rShoulder" : [pose[0,Joint.RSHOULDER,:], pose[0,Joint.NECK,:], pose[0,Joint.RELBOW,:]],
            #     "lelbow" : [pose[0,Joint.LELBOW,:], pose[0,Joint.LSHOULDER,:], pose[0,Joint.LWRIST,:]],
            #     "relbow" : [pose[0,Joint.RELBOW,:], pose[0,Joint.RSHOULDER,:], pose[0,Joint.RWRIST,:]],
            #     "lhip" : [pose[0,Joint.LHIP,:], pose[0,Joint.NECK,:], pose[0,Joint.LKNEE,:]],
            #     "rhip" : [pose[0,Joint.RHIP,:], pose[0,Joint.NECK,:], pose[0,Joint.RKNEE,:]],
            #     "lknee" : [pose[0,Joint.LKNEE,:], pose[0,Joint.LHIP,:], pose[0,Joint.LANKLE,:]],
            #     "rknee" : [pose[0,Joint.RKNEE,:], pose[0,Joint.RHIP,:], pose[0,Joint.RANKLE,:]],
            #     "lankle" : [pose[0,Joint.LANKLE,:], pose[0,Joint.LKNEE,:], pose[0,Joint.LBIGTOE,:]],
            #     "rankle" : [pose[0,Joint.RANKLE,:], pose[0,Joint.RKNEE,:], pose[0,Joint.RBIGTOE,:]]
            # }

            joint_angle_args = {
                "lshoulder": [pose[0, Joint.LEFT_SHOULDER, :], pose[0, Joint.NOSE, :], pose[0, Joint.LEFT_ELBOW, :]],
                "rShoulder": [pose[0, Joint.RIGHT_SHOULDER, :], pose[0, Joint.NOSE, :], pose[0, Joint.RIGHT_ELBOW, :]],
                "lelbow": [pose[0, Joint.LEFT_ELBOW, :], pose[0, Joint.LEFT_SHOULDER, :], pose[0, Joint.LEFT_WRIST, :]],
                "relbow": [pose[0, Joint.RIGHT_ELBOW, :], pose[0, Joint.RIGHT_SHOULDER, :], pose[0, Joint.RIGHT_WRIST, :]],
                "lhip": [pose[0, Joint.LEFT_HIP, :], pose[0, Joint.NOSE, :], pose[0, Joint.LEFT_KNEE, :]],
                "rhip": [pose[0, Joint.RIGHT_HIP, :], pose[0, Joint.NOSE, :], pose[0, Joint.RIGHT_KNEE, :]],
                "lknee": [pose[0, Joint.LEFT_KNEE, :], pose[0, Joint.LEFT_HIP, :], pose[0, Joint.LEFT_ANKLE, :]],
                "rknee": [pose[0, Joint.RIGHT_KNEE, :], pose[0, Joint.RIGHT_HIP, :], pose[0, Joint.RIGHT_ANKLE, :]],
                "lankle": [pose[0, Joint.LEFT_ANKLE, :], pose[0, Joint.LEFT_KNEE, :], pose[0, Joint.LEFT_FOOT_INDEX, :]],
                "rankle": [pose[0, Joint.RIGHT_ANKLE, :], pose[0, Joint.RIGHT_KNEE, :], pose[0, Joint.RIGHT_FOOT_INDEX, :]]
            }


            # Calculate all of the joint angles and write them to the position metrics dictionary
            for joint, args in joint_angle_args.items():
                self.position_metrics[dancer][joint][i] = self._calc_angle(*args)


            if(i > 0):
                posePrev = self.poses[dancer][i-1]
                # joint_vel_args = {
                #     "lshoulder" : [posePrev[0,Joint.LSHOULDER,:], pose[0,Joint.LSHOULDER,:]],
                #     "rShoulder" : [posePrev[0,Joint.RSHOULDER,:], pose[0,Joint.RSHOULDER,:]],
                #     "lelbow" : [posePrev[0,Joint.LELBOW,:], pose[0,Joint.LELBOW,:]],
                #     "relbow" : [posePrev[0,Joint.RELBOW,:], pose[0,Joint.RELBOW,:]],
                #     "lhip" : [posePrev[0,Joint.LHIP,:], pose[0,Joint.LHIP,:]],
                #     "rhip" : [posePrev[0,Joint.RHIP,:], pose[0,Joint.RHIP,:]],
                #     "lknee" : [posePrev[0,Joint.LKNEE,:], pose[0,Joint.LKNEE,:]],
                #     "rknee" : [posePrev[0,Joint.RKNEE,:], pose[0,Joint.RKNEE,:]],
                #     "lankle" : [posePrev[0,Joint.LANKLE,:], pose[0,Joint.LANKLE,:]],
                #     "rankle" : [posePrev[0,Joint.RANKLE,:], pose[0,Joint.RANKLE,:]]
                # }

                joint_vel_args = {
                    "lshoulder": [posePrev[0, Joint.LEFT_SHOULDER, :], pose[0, Joint.LEFT_SHOULDER, :]],
                    "rShoulder": [posePrev[0, Joint.RIGHT_SHOULDER, :], pose[0, Joint.RIGHT_SHOULDER, :]],
                    "lelbow": [posePrev[0, Joint.LEFT_ELBOW, :], pose[0, Joint.LEFT_ELBOW, :]],
                    "relbow": [posePrev[0, Joint.RIGHT_ELBOW, :], pose[0, Joint.RIGHT_ELBOW, :]],
                    "lhip": [posePrev[0, Joint.LEFT_HIP, :], pose[0, Joint.LEFT_HIP, :]],
                    "rhip": [posePrev[0, Joint.RIGHT_HIP, :], pose[0, Joint.RIGHT_HIP, :]],
                    "lknee": [posePrev[0, Joint.LEFT_KNEE, :], pose[0, Joint.LEFT_KNEE, :]],
                    "rknee": [posePrev[0, Joint.RIGHT_KNEE, :], pose[0, Joint.RIGHT_KNEE, :]],
                    "lankle": [posePrev[0, Joint.LEFT_ANKLE, :], pose[0, Joint.LEFT_ANKLE, :]],
                    "rankle": [posePrev[0, Joint.RIGHT_ANKLE, :], pose[0, Joint.RIGHT_ANKLE, :]]
                }


                for joint, args in joint_vel_args.items():
                    self.velocity_metrics[dancer][joint][i-1] = self._calc_velocity(*args)

    def add_frame_pose(self, student_pose, teacher_pose):
        """Add pose from a pair of frames from the student and teacher.

        Args:
            student_pose: A dict-type object that contains the the (x,y) coords of all of keypoints of the student
            teacher_pose: A dict-type object that contains the the (x,y) coords of all of keypoints of the teacher
        """

        self.poses["student"].append(student_pose)
        self.poses["teacher"].append(teacher_pose)


    # def generate_wireframe_video(self, fname):
    #     api = cv2.CAP_FFMPEG
    #     code = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    #     output = cv2.VideoWriter(fname, api, code, 30, (1920,1080*2))

    #     # Resolution of the video frames
    #     resolution = (1920,1080)


    #     joint_connections = [
    #         [Joint.NECK, Joint.NOSE],
    #         [Joint.NECK, Joint.LSHOULDER],
    #         [Joint.LSHOULDER, Joint.LELBOW],
    #         [Joint.LELBOW, Joint.LWRIST],
    #         [Joint.NECK, Joint.RSHOULDER],
    #         [Joint.RSHOULDER, Joint.RELBOW],
    #         [Joint.RELBOW, Joint.RWRIST],
    #         [Joint.NECK, Joint.MIDHIP],
    #         [Joint.MIDHIP, Joint.LHIP],
    #         [Joint.LHIP, Joint.LKNEE],
    #         [Joint.LKNEE, Joint.LANKLE],
    #         [Joint.LANKLE, Joint.LBIGTOE],
    #         [Joint.MIDHIP, Joint.RHIP],
    #         [Joint.RHIP, Joint.RKNEE],
    #         [Joint.RKNEE, Joint.RANKLE],
    #         [Joint.RANKLE, Joint.RBIGTOE]
    #     ]

    #     print(len(self.poses["student"]))
    #     print(len(self.poses["teacher"]))
    #     with tqdm(total=len(self.poses["student"]), desc='Writing') as pbar:
    #         for pose_student, pose_teacher in zip(self.poses["student"], self.poses["teacher"]):
    #             image_student = np.zeros(shape = (resolution[1], resolution[0], 3), dtype = np.uint8)
    #             image_teacher = np.zeros(shape = (resolution[1], resolution[0], 3), dtype = np.uint8)

    #             for connection in joint_connections:
    #                 if(pose_student[0, connection[0], 2] > 0.1 and pose_student[0, connection[1], 2] > 0.1):
    #                     start_point = tuple(pose_student[0, connection[0], 0:2])

    #                     # End coordinate, here (250, 250)
    #                     # represents the bottom right corner of image
    #                     end_point = tuple(pose_student[0, connection[1], 0:2])

    #                     # Green color in BGR
    #                     color = (255, 0, 0)

    #                     # Line thickness of 9 px
    #                     thickness = 9

    #                     # Using cv2.line() method
    #                     # Draw a diagonal green line with thickness of 9 px
    #                     image_student = cv2.line(image_student, start_point, end_point, color, thickness)

    #                 if(pose_teacher[0, connection[0], 2] > 0.1 and pose_teacher[0, connection[1], 2] > 0.1):
    #                     start_point = tuple(pose_teacher[0, connection[0], 0:2])

    #                     # End coordinate, here (250, 250)
    #                     # represents the bottom right corner of image
    #                     end_point = tuple(pose_teacher[0, connection[1], 0:2])

    #                     # Green color in BGR
    #                     color = (0, 0, 255)

    #                     # Line thickness of 9 px
    #                     thickness = 9

    #                     # Using cv2.line() method
    #                     # Draw a diagonal green line with thickness of 9 px
    #                     image_teacher = cv2.line(image_teacher, start_point, end_point, color, thickness)

    #             image = np.concatenate((image_teacher, image_student), axis=0)
    #             output.write(image)
    #             pbar.update(1)
    #     output.release()

    def score_dancer(self):
        """Generates a score rating the quality of the dancer.

        Returns:
            A dictionary containing scores for individual limbs as well as an overall score
        """

        self._calc_dance_metrics("student")
        self._calc_dance_metrics("teacher")



        position_errors = {
                "lshoulder" : None,
                "rShoulder" : None,
                "lelbow" : None,
                "relbow" : None,
                "lhip" : None,
                "rhip" : None,
                "lknee" : None,
                "rknee" : None,
                "lankle" : None,
                "rankle" : None
            }

        velocity_errors = {
            "lshoulder" : None,
            "rShoulder" : None,
            "lelbow" : None,
            "relbow" : None,
            "lhip" : None,
            "rhip" : None,
            "lknee" : None,
            "rknee" : None,
            "lankle" : None,
            "rankle" : None
        }

        avg_position_errors = {
            "lshoulder" : None,
            "rShoulder" : None,
            "lelbow" : None,
            "relbow" : None,
            "lhip" : None,
            "rhip" : None,
            "lknee" : None,
            "rknee" : None,
            "lankle" : None,
            "rankle" : None
        }

        avg_velocity_errors = {
            "lshoulder" : None,
            "rShoulder" : None,
            "lelbow" : None,
            "relbow" : None,
            "lhip" : None,
            "rhip" : None,
            "lknee" : None,
            "rknee" : None,
            "lankle" : None,
            "rankle" : None
        }

        scores = {
            "lshoulder" : None,
            "rShoulder" : None,
            "lelbow" : None,
            "relbow" : None,
            "lhip" : None,
            "rhip" : None,
            "lknee" : None,
            "rknee" : None,
            "lankle" : None,
            "rankle" : None
        }

        for joint in position_errors:
            for i in range(self.position_metrics['student'][joint].shape[0]):
                if self.position_metrics['student'][joint][i]==-1 or self.position_metrics['teacher'][joint][i]==-1:
                    self.position_metrics['student'][joint][i] = 0
                    self.position_metrics['teacher'][joint][i] = 0
            position_errors[joint] = np.linalg.norm(np.expand_dims(self.position_metrics["student"][joint] - self.position_metrics["teacher"][joint], axis = 1), axis = 1)
            velocity_errors[joint] = np.linalg.norm(np.expand_dims(self.velocity_metrics["student"][joint] - self.velocity_metrics["teacher"][joint], axis = 1), axis = 1)

            avg_position_errors[joint] = np.average(position_errors[joint])
            avg_velocity_errors[joint] = np.average(velocity_errors[joint])

            sigma = DanceScorer.RANGE[joint]/DanceScorer.SIGMA_SCALE

            z = avg_position_errors[joint]/sigma
            scores[joint] = (-1*(norm.cdf(abs(z))*2-1))+1

        total = 0
        avg = 0

        for joint, score in scores.items():
            # Scale score by 2.5 to make it less disheartening
            # With the current scheme, the scores are very low, scale them up so they saturate the 0-100 spectrum better
            if(score != 1):
                avg += 2.2*score
                total += 1

        scores["average"] = avg/total

        return scores

# # aligns the input videos
# def align(fname1, fname2, write=False, outpath1='', outpath2='', offset=0):
#     '''
#     captures videos and aligns the
#     :param fname1: filepath
#     :param fname2: filepath
#     :param write: save output videos or not
#     :param outpath1: path for video1 if saving
#     :param outpath2: path for video2 if saving
#     :return: frames1, frames2, fps, shape1, shape2
#     '''
#     # delay = alignment_by_row(fname1, fname2, '..')
#     delay = (0,0)
#     cap1 = cv2.VideoCapture(fname1)
#     cap2 = cv2.VideoCapture(fname2)
#     fps = cap1.get(cv2.CAP_PROP_FPS)
#     frame_width1 = cap1.get(3)
#     frame_height1 = cap1.get(4)
#     frame_width2 = cap2.get(3)
#     frame_height2 = cap2.get(4)
#     cap1.set(cv2.CAP_PROP_POS_MSEC, offset + delay[1] * 1000)
#     cap2.set(cv2.CAP_PROP_POS_MSEC, offset + delay[0] * 1000)

#     frames1 = []
#     frames2 = []
#     with tqdm(total=cap1.get(cv2.CAP_PROP_FRAME_COUNT), desc='Processing') as pbar:
#         while 1:
#             success1, frame1 = cap1.read()
#             success2, frame2 = cap2.read()
#             if success1 and success2:
#                 frames1.append(frame1)
#                 frames2.append(frame2)
#                 pbar.update(1)
#             else:
#                 break
#     if write:
#         write_video(outpath1, frames1, fps, (int(frame_width1), int(frame_height1)))
#         write_video(outpath2, frames2, fps, (int(frame_width2), int(frame_height2)))
#     return frames1, frames2, fps, (int(frame_width1), int(frame_height1)), (int(frame_width2), int(frame_height2))

# # write the frames as a video
# def write_video(fname, frames, fps, shape):
#     api = cv2.CAP_FFMPEG
#     code = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
#     output = cv2.VideoWriter(fname, api, code, fps, shape)
#     with tqdm(total=len(frames), desc='Writing') as pbar:
#         for frame in frames:
#             output.write(frame)
#             pbar.update(1)
#     output.release()

# # checks the alignment of the input videos
# def check_alignment(frames1, frames2, fps, shape1, shape2, outpath):
#     '''
#     Pairs videos side-by-side and saves combined files to outpath
#     :param frames1: list of aligned frames
#     :param frames2: list of aligned frames
#     :param fps: int number of fps for output video
#     :param shape1: shape of frames1
#     :param shape2: shape of frames2
#     :param outpath: file path to output combined video
#     :return: None
#     '''
#     shape = (int((shape1[0] + shape2[0])/2), int((shape1[1] + shape2[1]) / 4))
#     frames = []
#     for i in tqdm(range(len(frames1))):
#         vis = np.concatenate((frames1[i], frames2[i]), axis=1)
#         new_vis = cv2.resize(vis, shape)
#         frames.append(new_vis)

#     write_video(outpath, frames, fps, shape)

# def check_alignment_from_files(fname1, fname2, outpath):  # 
#     cap1 = cv2.VideoCapture(fname1)
#     cap2 = cv2.VideoCapture(fname2)
#     fps = cap2.get(cv2.CAP_PROP_FPS)
#     frame_width1 = cap1.get(3)
#     frame_height1 = cap1.get(4)
#     frame_width2 = cap2.get(3)
#     frame_height2 = cap2.get(4)

#     frames1 = []
#     frames2 = []

#     while 1:
#         success1, frame1 = cap1.read()
#         success2, frame2 = cap2.read()
#         if success1 and success2:
#             frames1.append(frame1)
#             frames2.append(frame2)
#         else:
#             break
#     check_alignment(frames1, frames2, fps, (int(frame_width1), int(frame_height1)), (int(frame_width2), int(frame_height2)), outpath)

class PoseEstimator:
    def __init__(self):
        # Mediapipe Pose initialization
        # self.mp_pose = mp.solutions.pose
        # self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False)
        self.dance_scorer = DanceScorer()

    # def process_image(self, image):
    #     """
    #     Processes a single image to extract pose landmarks using Mediapipe.
    #     """
    #     # Convert the image to RGB as Mediapipe requires RGB input
    #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     result = self.pose.process(image_rgb)
    #     return result

    # def process_image_path(self, path):
    #     """
    #     Processes an image from a file path.
    #     """
    #     image_to_process = cv2.imread(path)
    #     return self.process_image(image_to_process)

    # def display_pose(self, result, image):
    #     """
    #     Displays the pose landmarks on the image.
    #     """
    #     mp_drawing = mp.solutions.drawing_utils
    #     annotated_image = image.copy()
    #     mp_drawing.draw_landmarks(annotated_image, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
    #     cv2.imshow("Mediapipe Pose", annotated_image)
    #     cv2.waitKey(0)
    
    def process_keypoint_pair(self, kp1, kp2):
        print("reached process_keypoint_pair")
        print("details from process_keypoint_pair:")
        print("kp1:", type(kp1), kp1)
        print("kp1:", type(kp2), kp2)

        try:
            self.dance_scorer.add_frame_pose(kp1, kp2)
            print("done with process in process_keypoint_pair")
        except Exception as e_process_keypoint_pair:
            print("error with process_keypoint_pair:", e_process_keypoint_pair)

    def process_image_pair(self, image1, image2):
        """
        Generates pose estimation results and evaluates them using DanceScorer.
        """
        # Process and get landmarks for both images
        result1 = self.process_image(image1)
        result2 = self.process_image(image2)

        # Extract pose landmarks
        pose_keypoints1 = self._extract_pose_landmarks(result1)
        pose_keypoints2 = self._extract_pose_landmarks(result2)

        # assert pose_keypoints1.shape == (1, 33, 3)
        # assert pose_keypoints2.shape == (1, 33, 3)

        self.dance_scorer.add_frame_pose(pose_keypoints1, pose_keypoints2)
        return result1, result2

    # def _extract_pose_landmarks(self, result):
    #     """
    #     Extracts pose landmarks from the Mediapipe result.
    #     """
    #     if result.pose_landmarks is not None:
    #         landmarks = result.pose_landmarks.landmark
    #         pose_keypoints = np.array([[landmark.x, landmark.y, landmark.visibility] for landmark in landmarks])
    #         pose_keypoints = pose_keypoints[np.newaxis, :]  # Add batch dimension
    #     else:
    #         # Return an empty array if no landmarks are detected
    #         pose_keypoints = np.zeros((1, 33, 3))
    #     return pose_keypoints

    def dance_end(self):
        """
        Computes the dance score at the end.
        """
        return self.dance_scorer.score_dancer()

    # def iterate_over_video(self, path):
    #     """
    #     Processes a video to extract pose landmarks and write the results to a new video.
    #     """
    #     video = cv2.VideoCapture(path)
    #     fps = video.get(cv2.CAP_PROP_FPS)
    #     frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     frames = []

    #     with tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)), desc='Processing') as pbar:
    #         while True:
    #             success, frame = video.read()
    #             if success:
    #                 result = self.process_image(frame)
    #                 frames.append(self._annotate_frame(result, frame))
    #                 pbar.update(1)
    #             else:
    #                 break

    #     print('1/1')
    #     self.write_video('result.mp4', frames, fps, (frame_width, frame_height))

    # def _annotate_frame(self, result, frame):
    #     """
    #     Annotates a frame with pose landmarks.
    #     """
    #     mp_drawing = mp.solutions.drawing_utils
    #     annotated_frame = frame.copy()
    #     mp_drawing.draw_landmarks(annotated_frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
    #     return annotated_frame

    # def write_video(self, fname, frames, fps, shape):
    #     """
    #     Writes frames to a video file.
    #     """
    #     api = cv2.CAP_FFMPEG
    #     code = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    #     output = cv2.VideoWriter(fname, api, code, fps, shape)
    #     with tqdm(total=len(frames), desc='Writing') as pbar:
    #         for frame in frames:
    #             output.write(frame)
    #             pbar.update(1)
    #     output.release()

    # def compare_videos(self, path1, path2, write_skeleton=False, skeleton_out1='', skeleton_out2='',
    #                    write_aligned=False, aligned_out1='', aligned_out2='',
    #                    write_combined=False, combined_out=''):
    def compare_kps(self, seq1, seq2):
        """
        Compares two videos by processing frames and evaluating pose similarity.
        """
        # frames1, frames2, fps, shape1, shape2 = align(path1, path2, outpath1=aligned_out1,
        #                                               outpath2=aligned_out2,
        #                                               write=write_aligned)

        cvOut1 = []
        cvOut2 = []
        for i in tqdm(range(len(seq1))):
            # result1, result2 = self.process_image_pair(frames1[i], frames2[i])
            # cvOut1.append(self._annotate_frame(result1, frames1[i]))
            # cvOut2.append(self._annotate_frame(result2, frames2[i]))

            self.process_keypoint_pair(seq1[i], seq2[i])

        # if write_skeleton:
        #     print('1/2')
        #     self.write_video(skeleton_out1, cvOut1, fps, shape1)
        #     print('2/2')
        #     self.write_video(skeleton_out2, cvOut2, fps, shape2)
        # if write_combined:
        #     check_alignment(frames1, frames2, fps, shape1, shape2, combined_out)

        try:
            final_result = self.dance_end()
        except Exception as e_dance_end:
            print("error in dance_end operation:", e_dance_end)
        return final_result

# pose_estimator = PoseEstimator()

# fname1 = 'sowmya_sawadeeka_shorter.mp4'
# fname2 = 'teacher_sawadeeka_shorter.mp4'

# print(pose_estimator.compare_videos(fname1, fname2, write_skeleton=False, skeleton_out1='', skeleton_out2=''))