import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import List
import PIL.ImageFile
import cv2
from PIL import Image
from natsort import natsorted
import numpy as np
import PIL
from PIL.ImageFile import ImageFile
import fcntl



def extract_few_frames(video_path, indices=[55, 100]):
    """
    从视频中提取指定索引的帧，并将每帧保存到临时文件中。

    参数:
        video_path (str): 视频文件路径。
        indices (list of int): 要提取的帧索引列表。

    返回:
        temp_files (list of str): 保存帧的临时文件路径列表。
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return []

    temp_files = []

    for idx in indices:
        # 设置视频指针到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        # 读取该帧
        ret, frame = cap.read()
        if not ret:
            print(f"无法读取第 {idx} 帧")
            continue

        # 创建临时文件保存帧图像
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_files.append(temp_file.name)

        # 保存帧到临时文件
        cv2.imwrite(temp_file.name, frame)

    # 释放视频资源
    cap.release()

    return temp_files


def get_video_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video file")
        return 0

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return num_frames


def resize_and_crop(img: ImageFile, target_width: int, target_height: int):
    """
    将图像缩放并裁剪到指定大小，同时保持原始比例。

    参数:
        img (PIL.Image.Image): 要处理的图像。
        target_width (int): 目标宽度。
        target_height (int): 目标高度。

    返回:
        PIL.Image.Image: 已裁剪并缩放到目标大小的图像。
    """
    # 获取原始图像尺寸
    img_width, img_height = img.size
    aspect_ratio = img_width / img_height
    target_ratio = target_width / target_height

    # 裁剪宽度或高度，使图像比例接近目标比例
    if aspect_ratio > target_ratio:
        # 原图太宽，需要裁剪宽度
        new_width = int(target_ratio * img_height)
        left = (img_width - new_width) // 2
        img = img.crop((left, 0, left + new_width, img_height))
    elif aspect_ratio < target_ratio:
        # 原图太高，需要裁剪高度
        new_height = int(img_width / target_ratio)
        top = (img_height - new_height) // 2
        img = img.crop((0, top, img_width, top + new_height))

    # 缩放到目标尺寸
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return img


def get_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        file: str
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    return natsorted(image_paths)


def count_images(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                count += 1
    return count


def count_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            count += 1
    return count

def get_file_name_without_ext(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def get_video_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        file: str
        for file in files:
            if file.endswith(".mp4"):
                image_paths.append(os.path.join(root, file))
    return natsorted(image_paths)


def img_grid(image_paths: np.ndarray, target_size=(512, 512), save_path="comparison_grid.png", vertical_margins=None, horizontal_margins=None):
    """
    使用PIL直接拼接图片并保存结果，避免通过matplotlib缩放引起的模糊。

    参数:
        image_paths (list of list of str): hxw 的矩阵，每个元素是图像文件的路径。
        target_size (tuple): 每张图片的目标大小 (width, height)。
        save_path (str): 拼接后的图像保存路径。
        vertical_margins (list of int): 每个列间的白边宽度列表，应有 (cols-1) 个元素。
        horizontal_margins (list of int): 每行之间的白边宽度列表，应有 (rows-1) 个元素。
    """
    rows, cols = image_paths.shape
    target_width, target_height = target_size

    # 若未指定白边宽度列表，则默认为全0
    def create_list(l, n):
        if not l:
            return [0] * n
        if len(l) < n:
            return l + [0] * (n - len(l))
        return l[:n]

    vertical_margins = create_list(vertical_margins, cols - 1)
    horizontal_margins = create_list(horizontal_margins, rows - 1)

    # 计算总宽度和总高度
    total_vertical_margin_width = sum(vertical_margins)
    total_horizontal_margin_height = sum(horizontal_margins)
    new_img_width = cols * target_width + total_vertical_margin_width
    new_img_height = rows * target_height + total_horizontal_margin_height
    new_img = Image.new("RGB", (new_img_width, new_img_height), (255, 255, 255))

    # 拼接每张图片
    for i in range(rows):
        x_offset = 0  # 每行的水平偏移量
        y_offset = i * target_height + sum(horizontal_margins[:i])  # 每行的垂直偏移量
        for j in range(cols):
            img_path:str = image_paths[i][j]
            img = Image.open(img_path)

            # 缩放图片到目标大小
            img = resize_and_crop(img, target_width, target_height)

            # 将图片粘贴到新图像上
            new_img.paste(img, (x_offset, y_offset))

            # 更新 x_offset，增加图片宽度和白边宽度
            x_offset += target_width
            if j < cols - 1:  # 添加白边
                x_offset += vertical_margins[j]

    # 保存拼接后的图像
    new_img.save(save_path, dpi=(300, 300))
    print(save_path)
    return save_path


def extract_all_frames(video_path, out_img_dir, fps=25, ext=".jpg"):
    cap_vid = cv2.VideoCapture(video_path)
    video_width = int(cap_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(out_img_dir, exist_ok=True)
    resize_option = f"-s {video_width // 2 * 2}x{video_height // 2 * 2}" if (video_width % 2 == 1 or video_height % 2 == 1) else ""
    ffmpeg_cmd = f"ffmpeg -loglevel error -y -i {video_path} {resize_option} -q:v 1 -vf 'fps={fps}' -start_number 0 {out_img_dir}/%06d{ext}"
    # os.system(ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd, shell=True)


def merge_video(input_files, out_path):
    dir_name = os.path.dirname(out_path)
    os.makedirs(dir_name, exist_ok=True)
    os.system(f"ffmpeg -framerate 25 -i {input_files} -c:v libx264 -pix_fmt yuv420p -loglevel error {out_path}")
    print(f"see {out_path}")


def crop_video_half(input_video, output_video):
    subprocess.run(["ffmpeg", "-i", input_video, "-vf", "crop=in_w/2:in_h:in_w/2:0", "-c:a", "copy", output_video])


def get_sub_dir(dir_path, *sub_names):
    sub_dir_path = os.path.join(dir_path, *sub_names)
    os.makedirs(sub_dir_path, exist_ok=True)
    return sub_dir_path


def change_extension(file_path, new_extension):
    base = os.path.splitext(file_path)[0]
    return base + new_extension


def get_first_mp4_file(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                return os.path.join(root, file)
    return None


class FileLock:
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.lock_fd = None

        # 如果锁文件不存在，创建空文件
        if not os.path.exists(lock_file):
            open(lock_file, "w").close()

    def acquire(self):
        """尝试获取文件锁"""
        try:
            self.lock_fd = open(self.lock_file, "w")
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except IOError:
            if self.lock_fd:
                self.lock_fd.close()
                self.lock_fd = None
            return False

    def release(self):
        """释放文件锁"""
        if self.lock_fd:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            self.lock_fd.close()
            self.lock_fd = None
            os.remove(self.lock_file)  # 删除文件锁

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动释放锁"""
        self.release()


if __name__ == "__main__":
    lock = FileLock("test.lock")
    print(lock.acquire())

    lock2 = FileLock("test.lock")
    print(lock2.acquire())


class LazyVideoWriter:
    def __init__(self, save_path, fps=25.0, save_images=False):
        self.writer = None
        assert save_path.endswith(".mp4"), "Only support mp4 format"
        self.save_path = save_path
        self.save_images = save_images
        self.fps = fps
        if save_images:
            self.image_dir = Path(save_path).with_suffix("")
            self.image_dir.mkdir(exist_ok=True)

    def write(self, image: np.ndarray):
        if self.writer is None:
            size = image.shape[:2][::-1]
            self.writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.save_images:
            image_path = self.image_dir / f"{self.writer.get(cv2.CAP_PROP_FRAME_COUNT):06d}.png"
            cv2.imwrite(str(image_path), image_bgr)
        self.writer.write(image_bgr)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        if exc_type is not None:
            print(f"An exception occurred: {exc_val}")
        return False


def video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


def reencode_audio(audio_file, output):
    cmd = f"ffmpeg -i {audio_file} -y -hide_banner -loglevel error {output}"
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def combine_video_and_audio(video_file, audio_file, output, quality=17, copy_audio=True):
    audio_codec = "-c:a copy" if copy_audio else ""
    cmd = (
        f"ffmpeg -i {video_file} -i {audio_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p "
        f"{audio_codec} -fflags +shortest -y -hide_banner -loglevel error {output}"
    )
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def convert_video(video_file, output, quality=17):
    cmd = f"ffmpeg -i {video_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p " f"-fflags +shortest -y -hide_banner -loglevel error {output}"
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def save_frames_to_video(frames, out_path, audio_path=None, fps=25, save_images=False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_video_file = tempfile.NamedTemporaryFile("w", suffix=".mp4", dir=out_path.parent)
    if save_images:
        out_image_dir = out_path.with_suffix("")
        out_image_dir.mkdir(exist_ok=True)
    with LazyVideoWriter(tmp_video_file.name, fps=fps, save_images=save_images) as writer:
        for frame in frames:
            writer.write(frame)
    if audio_path is not None:
        # needs to re-encode audio to AAC format first, or the audio will be ahead of the video!
        tmp_audio_file = tempfile.NamedTemporaryFile("w", suffix=".mp3", dir=out_path.parent)
        reencode_audio(audio_path, tmp_audio_file.name)
        combine_video_and_audio(tmp_video_file.name, tmp_audio_file.name, out_path)
        tmp_audio_file.close()
    else:
        convert_video(tmp_video_file.name, out_path)
    tmp_video_file.close()
