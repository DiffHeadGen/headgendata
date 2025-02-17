from functools import cached_property
import os
import shutil
import face_alignment
import numpy as np
from PIL import Image
# from expdataloader.P4DLoader import *
from tqdm import tqdm

from expdataloader.P4DLoader import P4DLoader, P4DRowData
from expdataloader.utils import change_extension, get_image_paths, merge_video


def clac_quad(face_landmarks):
    lm = np.array(face_landmarks)
    lm_chin = lm[0:17, :2]  # left-right
    lm_eyebrow_left = lm[17:22, :2]  # left-right
    lm_eyebrow_right = lm[22:27, :2]  # left-right
    lm_nose = lm[27:31, :2]  # top-down
    lm_nostrils = lm[31:36, :2]  # top-down
    lm_eye_left = lm[36:42, :2]  # left-clockwise
    lm_eye_right = lm[42:48, :2]  # left-clockwise
    lm_mouth_outer = lm[48:60, :2]  # left-clockwise
    lm_mouth_inner = lm[60:68, :2]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    # quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    return c, qsize


def solve_transform(source_ldmks, target_ldmks):
    """
    求解从 source_ldmks 到 target_ldmks 的变换：target = source * resize + trans
    :param source_ldmks: (n, 2) 的数组，表示 source 点的坐标
    :param target_ldmks: (n, 2) 的数组，表示 target 点的坐标
    :return: resize (标量), trans (1, 2) 的数组
    """
    n = source_ldmks.shape[0]

    # 构造矩阵 A 和向量 b
    A = np.zeros((2 * n, 3))
    b = np.zeros((2 * n, 1))

    # 填充 A 和 b
    A[:n, 0] = source_ldmks[:, 0]  # source_x
    A[:n, 1] = 1  # 1 (for tx)
    A[n:, 0] = source_ldmks[:, 1]  # source_y
    A[n:, 2] = 1  # 1 (for ty)

    b[:n, 0] = target_ldmks[:, 0]  # target_x
    b[n:, 0] = target_ldmks[:, 1]  # target_y

    # 最小二乘求解
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # 提取结果
    resize = x[0, 0]
    trans = np.array([x[1, 0], x[2, 0]])

    return resize, trans


class Retargeter:
    def __init__(self, use_cache=False):
        self.use_cache = use_cache
        pass

    @cached_property
    def landmark_detector(self):
        return face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)

    def get_landmarks(self, img_path):
        if self.use_cache:
            save_path = os.path.splitext(img_path)[0] + ".npy"
            if os.path.exists(save_path):
                ldmks = np.load(save_path)
                return ldmks
        ldmkss = self.landmark_detector.get_landmarks(img_path)
        ldmks = ldmkss[0]
        if self.use_cache:
            np.save(save_path, ldmks)
        return ldmks

    def retarget(self, source_img_path, target_img_path, output_path):
        source_ldmks = self.get_landmarks(source_img_path)
        target_ldmks = self.get_landmarks(target_img_path)
        scale, trans = solve_transform(target_ldmks[:, :2], source_ldmks[:, :2])
        source_img = Image.open(source_img_path)
        target_img = Image.open(target_img_path)
        target_img = target_img.resize((int(target_img.width * scale), int(target_img.height * scale)))
        paste_pos = trans.astype(int)
        source_img.paste(target_img, tuple(paste_pos))
        source_img.save(output_path)


