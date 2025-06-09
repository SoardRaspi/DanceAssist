import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from enum import IntEnum


class Joint(IntEnum):
    RIGHT_ELBOW = 0
    RIGHT_SHOULDER = 1
    LEFT_SHOULDER = 2
    LEFT_ELBOW = 3


# Main scoring code
class DanceScorer_Angles:
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
        self.angle_metrics = {
            "student" : {
                "relbow" : [],
                "rShoulder" : [],
                "lshoulder" : [],
                "lelbow" : []
            },
            "teacher" : {
                "relbow" : [],
                "rShoulder" : [],
                "lshoulder" : [],
                "lelbow" : []
            }
        }

        # Each element in this dictionary is a list of length n-1
        # These are all "first derviative" metrics like velocity
        self.velocity_metrics = {
            "student" : {
                "relbow" : [],
                "rShoulder" : [],
                "lshoulder" : [],
                "lelbow" : []
            },
            "teacher" : {
                "relbow" : [],
                "rShoulder" : [],
                "lshoulder" : [],
                "lelbow" : []
            }
        }
    
    def calc_angular_velocity(self, prev_angle, curr_angle):
        print("enterred calc_angular_velocity function with values:", prev_angle, curr_angle)
        if prev_angle < 0.1 or curr_angle < 0.1:
            return -1

        v1 = prev_angle
        v2 = curr_angle

        return np.linalg.norm(v2 - v1)


    def _calc_dance_metrics(self, dancer):
        print("enterred _calc_dance_metrics function:", dancer, file=open(log_file, 'a'))
        # select data
        if(dancer != "student" and dancer != "teacher"):
            raise Exception("Selected dancer must be a student or teacher")

        # Create numpy arrays of the right length
        for joint in self.angle_metrics[dancer]:
            print("inside the loop of _calc_dance_metrics function:", joint, file=open(log_file, 'a'))
            self.angle_metrics[dancer][joint] = np.zeros(shape = (len(self.poses[dancer]), ), dtype = np.float32)

            if len(self.poses[dancer]) == 0:
                self.velocity_metrics[dancer][joint] = np.zeros(shape = (len(self.poses[dancer]), ), dtype = np.float32)
            else:
                self.velocity_metrics[dancer][joint] = np.zeros(shape = (len(self.poses[dancer]), ), dtype = np.float32)


        for i, pose in enumerate(self.poses[dancer]):
            print("inside the loop in DanceScorer_Angles _calc_dance_metrics:", i, type(pose), pose, file=open(log_file, 'a'))

            # joint_angle_args = {
            #     "lshoulder": [pose[0, Joint.LEFT_SHOULDER, :]],
            #     "rShoulder": [pose[0, Joint.RIGHT_SHOULDER, :]],
            #     "lelbow": [pose[0, Joint.LEFT_ELBOW, :]],
            #     "relbow": [pose[0, Joint.RIGHT_ELBOW, :]]
            # }

            joint_angle_args = {
                "lshoulder": [pose[Joint.LEFT_SHOULDER]],
                "rShoulder": [pose[Joint.RIGHT_SHOULDER]],
                "lelbow": [pose[Joint.LEFT_ELBOW]],
                "relbow": [pose[Joint.RIGHT_ELBOW]]
            }

            joint_angles = {
                "relbow": [],
                "rShoulder": [],
                "lshoulder": [],
                "lelbow": []
            }


            # Calculate all of the joint angles and write them to the position metrics dictionary
            for joint, args in joint_angle_args.items():
                print("inside the args loop:", joint, args, file=open(log_file, 'a'))
                # self.position_metrics[dancer][joint][i] = self._calc_angle(*args)
                self.angle_metrics[dancer][joint][i] = args[0]


            if(i > 0):
                print("line 136 in dance comparison code:", i, type(pose), pose, file=open(log_file, 'a'))
                print("data for posePrev:", dancer, self.poses[dancer], file=open(log_file, 'a'))
                posePrev = self.poses[dancer][i-1]

                print("posePrev:", posePrev, file=open(log_file, 'a'))

                # joint_vel_args = {
                #     "lshoulder": [posePrev[0, Joint.LEFT_SHOULDER, :], pose[0, Joint.LEFT_SHOULDER, :]],
                #     "rShoulder": [posePrev[0, Joint.RIGHT_SHOULDER, :], pose[0, Joint.RIGHT_SHOULDER, :]],
                #     "lelbow": [posePrev[0, Joint.LEFT_ELBOW, :], pose[0, Joint.LEFT_ELBOW, :]],
                #     "relbow": [posePrev[0, Joint.RIGHT_ELBOW, :], pose[0, Joint.RIGHT_ELBOW, :]]
                # }

                joint_vel_args = {
                    "lshoulder": [posePrev[Joint.LEFT_SHOULDER], pose[Joint.LEFT_SHOULDER]],
                    "rShoulder": [posePrev[Joint.RIGHT_SHOULDER], pose[Joint.RIGHT_SHOULDER]],
                    "lelbow": [posePrev[Joint.LEFT_ELBOW], pose[Joint.LEFT_ELBOW]],
                    "relbow": [posePrev[Joint.RIGHT_ELBOW], pose[Joint.RIGHT_ELBOW]]
                }

                print("joint_vel_args:", joint_vel_args, file=open(log_file, 'a'))


                for joint, args in joint_vel_args.items():
                    print("inside the loop after line 136:", joint, args, file=open(log_file, 'a'))
                    print("inside the loop after line 136 2:", type(args), args, file=open(log_file, 'a'))

                    try:

                        temp_angular_velocity = self.calc_angular_velocity(args[0], args[1])
                    except Exception as e_line_136_1:
                        print("exception in loop after line 136 1:", e_line_136_1, file=open(log_file, 'a'))
                    
                    try:
                        self.velocity_metrics[dancer][joint][i-1] = temp_angular_velocity
                    except Exception as e_line_136_2:
                        print("exception in loop after line 136 2:", e_line_136_2, file=open(log_file, 'a'))

                    print("temp_angular_velocity after line 136:", temp_angular_velocity, file=open(log_file, 'a'))


    def add_frame_pose(self, student_pose, teacher_pose):
        """Add pose from a pair of frames from the student and teacher.

        Args:
            student_pose: A dict-type object that contains the the (x,y) coords of all of keypoints of the student
            teacher_pose: A dict-type object that contains the the (x,y) coords of all of keypoints of the teacher
        """

        self.poses["student"].append(student_pose)
        self.poses["teacher"].append(teacher_pose)

    def score_dancer(self):
        """Generates a score rating the quality of the dancer.

        Returns:
            A dictionary containing scores for individual limbs as well as an overall score
        """

        print("enterred in score_dance function", file=open(log_file, 'a'))
        self._calc_dance_metrics("student")
        print("done with calc_dance_metrics for student", file=open(log_file, 'a'))
        self._calc_dance_metrics("teacher")
        print("done with calc_dance_metrics for teacher", file=open(log_file, 'a'))



        angle_errors = {
            "relbow" : None,
            "rShoulder" : None,
            "lshoulder" : None,
            "lelbow" : None
        }

        velocity_errors = {
            "relbow" : None,
            "rShoulder" : None,
            "lshoulder" : None,
            "lelbow" : None
        }

        avg_angle_errors = {
            "relbow" : None,
            "rShoulder" : None,
            "lshoulder" : None,
            "lelbow" : None
        }

        avg_velocity_errors = {
            "relbow" : None,
            "rShoulder" : None,
            "lshoulder" : None,
            "lelbow" : None
        }

        scores = {
            "relbow" : None,
            "rShoulder" : None,
            "lshoulder" : None,
            "lelbow" : None
        }

        print("data inside DanceScorer's score_dancer function:", angle_errors, self.angle_metrics, file=open(log_file, 'a'))
        # print("angle_metrics in score_dance before error calculation:", self.angle_metrics, file=open(log_file, 'a'))
        # print("velocity_metrics in score_dance before error calculation:", self.velocity_metrics, file=open(log_file, 'a'))

        for joint in angle_errors:
            for i in range(self.angle_metrics['student'][joint].shape[0]):
                if self.angle_metrics['student'][joint][i]==-1 or self.angle_metrics['teacher'][joint][i]==-1:
                    self.angle_metrics['student'][joint][i] = 0
                    self.angle_metrics['teacher'][joint][i] = 0
            angle_errors[joint] = np.linalg.norm(np.expand_dims(self.angle_metrics["student"][joint] - self.angle_metrics["teacher"][joint], axis = 1), axis = 1)
            velocity_errors[joint] = np.linalg.norm(np.expand_dims(self.velocity_metrics["student"][joint] - self.velocity_metrics["teacher"][joint], axis = 1), axis = 1)

            print("angle_errors[joint] in loop of score_dancer after line 136:", angle_errors[joint], file=open(log_file, 'a'))
            print("velocity_errors[joint] in loop of score_dancer after line 136:", velocity_errors[joint], file=open(log_file, 'a'))

            avg_angle_errors[joint] = np.average(angle_errors[joint])
            avg_velocity_errors[joint] = np.average(velocity_errors[joint])

            print("avg_angle_errors[joint] in loop of score_dancer after line 136:", avg_angle_errors[joint], file=open(log_file, 'a'))
            print("avg_velocity_errors[joint] in loop of score_dancer after line 136:", avg_velocity_errors[joint], file=open(log_file, 'a'))

            sigma = DanceScorer_Angles.RANGE[joint]/DanceScorer_Angles.SIGMA_SCALE

            print("sigma in loop after line 136:", sigma, file=open(log_file, 'a'))

            z = avg_angle_errors[joint]/sigma

            print("z in loop after line 136:", z, file=open(log_file, 'a'))

            score_join_temp = (-1 * (norm.cdf(abs(z / 360.0)) * 2 - 1)) + 1
            scores[joint] = score_join_temp

            print("score_join_temp in loop after line 136:", score_join_temp, file=open(log_file, 'a'))

        total = 0
        avg = 0

        print("scores in score_dancer after line 136:", scores, file=open(log_file, 'a'))

        for joint, score in scores.items():
            # Scale score by 2.5 to make it less disheartening
            # With the current scheme, the scores are very low, scale them up so they saturate the 0-100 spectrum better
            if(score != 1):
                avg += 2.2*score
                total += 1

        scores["average"] = avg/total

        return scores


log_file = "DDPG v2 log.txt"


class PoseEstimator:
    def __init__(self):
        self.dance_scorer = DanceScorer_Angles()
    
    def process_keypoint_pair(self, kp1, kp2):
        print("reached process_keypoint_pair", file=open(log_file, 'a'))
        print("details from process_keypoint_pair:", file=open(log_file, 'a'))
        print("kp1:", type(kp1), kp1, file=open(log_file, 'a'))
        print("kp1:", type(kp2), kp2, file=open(log_file, 'a'))

        try:
            self.dance_scorer.add_frame_pose(kp1, kp2)
            print("done with process in process_keypoint_pair", file=open(log_file, 'a'))
        except Exception as e_process_keypoint_pair:
            print("error with process_keypoint_pair:", e_process_keypoint_pair, file=open(log_file, 'a'))

    def dance_end(self):
        """
        Computes the dance score at the end.
        """
        print("inside the dance_end function, before", file=open(log_file, 'a'))
        return self.dance_scorer.score_dancer()
    
    def compare_kps(self, seq1, seq2):
        """
        Compares two videos by processing frames and evaluating pose similarity.
        """
        # frames1, frames2, fps, shape1, shape2 = align(path1, path2, outpath1=aligned_out1,
        #                                               outpath2=aligned_out2,
        #                                               write=write_aligned)

        seq1 = np.array(seq1)

        seq1 = seq1.reshape(60, 4)
        seq2 = seq2.reshape(60, 4)

        print("seq1 input:", type(seq1), seq1.shape, seq1, file=open(log_file, 'a'))
        print("seq2 input:", type(seq2), seq2.shape, seq2, file=open(log_file, 'a'))

        cvOut1 = []
        cvOut2 = []
        for i in tqdm(range(len(seq1))):
            # result1, result2 = self.process_image_pair(frames1[i], frames2[i])
            # cvOut1.append(self._annotate_frame(result1, frames1[i]))
            # cvOut2.append(self._annotate_frame(result2, frames2[i]))

            print("inside the compare_kps loop:", i, file=open(log_file, 'a'))

            self.process_keypoint_pair(seq1[i], seq2[i])

        try:
            final_result = self.dance_end()
            return final_result
        except Exception as e_dance_end:
            print("error in dance_end operation:", e_dance_end, file=open(log_file, 'a'))

# pose_estimator = PoseEstimator()

# fname1 = 'sowmya_sawadeeka_shorter.mp4'
# fname2 = 'teacher_sawadeeka_shorter.mp4'

# print(pose_estimator.compare_videos(fname1, fname2, write_skeleton=False, skeleton_out1='', skeleton_out2=''))