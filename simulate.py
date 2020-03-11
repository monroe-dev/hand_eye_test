import cv2
import numpy as np
from numpy import linalg
from kinematics import homogeneous_matrix, homogeneous_inverse


def newPose(min_theta, max_theta, min_tvec, max_tvec):
    """ Generate new random poses within the specific range (rvec and tvec)
    Did it need to specify the range of tvec ?
    """
    axis = np.random.uniform(-1, 1, (3, 1))
    theta = np.random.uniform(min_theta, max_theta)
    rvec = axis * theta
    tvec = np.random.uniform(min_tvec, max_tvec, (3, 1))
    R = np.empty((3, 3))
    cv2.Rodrigues(rvec, R)
    return R, tvec


def generateNewPose(nPose, noise):
    """ T_hand2eye, T_base2world are constant(fixed) matrix.
        T_base2hand, T_eye2world are different matrix at the each pose."""

    eye_noise = [-3, 3, -3, 3]      # [min_rvec, max_rvec, min_t, max_t]
    robot_noise = [-3, 3, -3, 3]      # [min_rvec, max_rvec, min_t, max_t]
    # Generate
    R_hand2eye, t_hand2eye = newPose(np.deg2rad(10), np.deg2rad(50), 0.05, 0.5)
    T_hand2eye = homogeneous_matrix(R_hand2eye, t_hand2eye)
    R_base2world, t_base2world = newPose(np.deg2rad(5), np.deg2rad(85), 0.5, 3.5)
    T_base2world = homogeneous_matrix(R_base2world, t_base2world)

    R_eye2world = np.zeros((1, 3, 3))
    t_eye2world = np.zeros((1, 3, 1))
    R_base2hand = np.zeros((1, 3, 3))
    t_base2hand = np.zeros((1, 3, 1))

    for i in range(nPose):
        R_base2hand_, t_base2hand_ = newPose(np.deg2rad(5), np.deg2rad(40), 0.5, 1.5)
        T_base2hand = homogeneous_matrix(R_base2hand_, t_base2hand_)

        T_eye2base = np.dot(homogeneous_inverse(T_hand2eye), homogeneous_inverse(T_base2hand))
        T_eye2world = np.dot(T_eye2base, T_base2world)
        R_eye2world_ = T_eye2world[:3, :3]
        t_eye2world_ = T_eye2world[:3, 3]
        t_eye2world_ = np.expand_dims(t_eye2world_, axis=1)

        if noise:
            """ Estimation of world coordinate using camera(eye) has some error,
                and the robot also usually exist positioning error."""
            # World coordinate
            # Add to arbitrary noise.
            R_eye2world_noise = T_eye2world[:3, :3]
            rvec_eye2world_noise = np.empty((3, 1))
            # Orientation
            cv2.Rodrigues(R_eye2world_noise, rvec_eye2world_noise)
            rvec_eye2world_noise = np.add(rvec_eye2world_noise, np.random.uniform(eye_noise[0], eye_noise[1], (3, 1)))
            cv2.Rodrigues(rvec_eye2world_noise, R_eye2world_noise)
            R_eye2world_ = R_eye2world_noise
            # Position
            t_eye2world_noise = T_eye2world[:3, 3]
            t_eye2world_noise = np.add(np.expand_dims(t_eye2world_noise, axis=1), np.random.uniform(eye_noise[2], [3], (3, 1)))
            t_eye2world_ = t_eye2world_noise

            # Robot hand
            # Orientation
            R_base2hand_noise = T_base2hand[:3, :3]
            rvec_base2hand_noise = np.empty((3, 1))
            cv2.Rodrigues(R_base2hand_noise, rvec_base2hand_noise)
            rvec_base2hand_noise = np.add(rvec_base2hand_noise, np.random.uniform(robot_noise[0], robot_noise[1], (3, 1)))
            cv2.Rodrigues(rvec_base2hand_noise, R_base2hand_noise)
            R_base2hand_ = R_base2hand_noise
            # Position
            t_base2hand_noise = T_base2hand[:3, 3]
            t_base2hand_noise = np.add(np.expand_dims(t_base2hand_noise, axis=1), np.random.uniform(robot_noise[2], robot_noise[3], (3, 1)))
            t_base2hand_ = t_base2hand_noise

        # Rotation matrix and translation vector at the every pose
        R_eye2world = np.concatenate((R_eye2world, np.expand_dims(R_eye2world_, axis=0)), axis=0)
        t_eye2world = np.concatenate((t_eye2world, np.expand_dims(t_eye2world_, axis=0)), axis=0)
        R_base2hand = np.concatenate((R_base2hand, np.expand_dims(R_base2hand_, axis=0)), axis=0)
        t_base2hand = np.concatenate((t_base2hand, np.expand_dims(t_base2hand_, axis=0)), axis=0)

    # Because first array was zeros..
    return R_eye2world[1:], t_eye2world[1:], R_base2hand[1:], t_base2hand[1:], R_hand2eye, t_hand2eye


def validate(numTest, numPose, numMethod):
    noise = True
    # Allocation
    rvec_hand2eye_true = np.empty((3, 1))
    R_hand2eye_est = np.empty((3, 3))
    t_hand2eye_est = np.empty((3, 1))
    rvec_hand2eye_est = np.empty((3, 1))
    # Five methods
    rvec_diff = np.empty((1, numMethod))
    t_diff = np.empty((1, numMethod))

    for i in range(numTest):
        # Generate a number of poses
        R_eye2world, t_eye2world, R_base2hand, t_base2hand, R_hand2eye_true, t_hand2eye_true = generateNewPose(numPose, noise)
        rvec_diff_ = np.empty(1)
        t_diff_ = np.empty(1)

        for j in range(numMethod):
            # Estimation of transformation from robot hand to camera
            cv2.calibrateHandEye(R_base2hand, t_base2hand, R_eye2world, t_eye2world, R_hand2eye_est, t_hand2eye_est, method=j)
            # Validate
            # Orientation
            cv2.Rodrigues(R_hand2eye_true, rvec_hand2eye_true)
            cv2.Rodrigues(R_hand2eye_est, rvec_hand2eye_est)
            rvec_diff__ = np.linalg.norm(np.subtract(rvec_hand2eye_true, rvec_hand2eye_est))
            # Translation
            t_diff__ = np.linalg.norm(np.subtract(t_hand2eye_true, t_hand2eye_est))
            # The result vectors at the each method
            rvec_diff_ = np.append(rvec_diff_, np.expand_dims(rvec_diff__, axis=0), axis=0)
            t_diff_ = np.append(t_diff_, np.expand_dims(t_diff__, axis=0), axis=0)

        rvec_diff_ = rvec_diff_.reshape(1, numMethod + 1)
        t_diff_ = t_diff_.reshape(1, numMethod + 1)
        # The result vectors at the each test
        rvec_diff = np.concatenate((rvec_diff, rvec_diff_[:, 1:numMethod + 1]), axis=0)
        t_diff = np.concatenate((t_diff, t_diff_[:, 1:numMethod + 1]), axis=0)

    mean_rvec_diff = np.mean(rvec_diff[1:, :], axis=0)
    mean_t_diff = np.mean(t_diff[1:, :], axis=0)

    std_rvec_diff = np.std(rvec_diff[1:, :], axis=0)
    std_t_diff = np.std(t_diff[1:, :], axis=0)

    max_rvec_diff = np.max(rvec_diff[1:, :], axis=0)
    max_t_diff = np.max(t_diff[1:, :], axis=0)

    return rvec_diff[1:], t_diff[1:], mean_rvec_diff, mean_t_diff, std_rvec_diff, std_t_diff, max_rvec_diff, max_t_diff

