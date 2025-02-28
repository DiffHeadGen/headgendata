from functools import cached_property
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import cv2
import face_alignment
import insightface
import numpy as np
from tqdm import tqdm
from expdataloader.utils import get_sub_dir, get_video_paths, get_video_num_frames
from PIL import Image


def video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    assert ret, f"Cannot read video {video_path}"
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


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


def reencode_audio(audio_file, output):
    cmd = f"ffmpeg -i {audio_file} -y -hide_banner -loglevel error {output}"
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
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    return c, qsize


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

class FaceDetector:
    def __init__(self, ctx_id=0, det_thresh=0.5, det_size=(640, 640)) -> None:
        model = insightface.app.FaceAnalysis(
            allowed_modules=["detection", "recognition"],
            # name="buffalo_sc",
        )
        model.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)
        self.model = model

    def get(self, frame) -> list:
        faces = self.model.get(frame)
        largest_face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        return largest_face
    

class VideoCropper:
    def __init__(self):
        pass

    def crop_video(self, video_path, output_path, scale=1.8):
        save_frames_to_video(self.process_video(video_path, scale=scale), output_path)
        print(f"see {output_path }")

    @cached_property
    def ldmk_cache_dir(self):
        return get_sub_dir(".cache/ldmks")

    @cached_property
    def face_detector(self):
        return FaceDetector()

    @cached_property
    def landmark_detector(self):
        return face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)

    def get_landmarks(self, frame, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            return np.load(cache_path)
        landmarks = self.landmark_detector.get_landmarks(frame)[0]
        if cache_path:
            np.save(cache_path, landmarks)
        return landmarks

    def test(self, video_path):
        first_frame = get_first_frame(video_path)
        ldmks_path = os.path.join(self.ldmk_cache_dir, hashlib.md5(video_path.encode()).hexdigest() + ".npy")
        landmarks = self.get_landmarks(first_frame, ldmks_path)
        c, qsize = clac_quad(landmarks)
        img = Image.fromarray(first_frame)
        border = max(int(np.rint(qsize * 0.1)), 3)
        r = min(qsize // 2 + border, img.size[0] // 2, img.size[1] // 2)
        x = max(r, min(img.size[0] - r, c[0]))
        y = max(r, min(img.size[1] - r, c[1]))
        crop = (int(x - r), int(y - r), int(x + r), int(y + r))
        print(img.size, crop)

    def process_video(self, video_path, scale=1.8, fps=25, output_size=512):
        first_frame = get_first_frame(video_path)
        face = self.face_detector.get(first_frame)
        bbox = face.bbox
        l, t, r, b = bbox
        # extend to square
        img = Image.fromarray(first_frame)
        cx, cy = (l + r) // 2, (t + b) // 2
        ext = max((r - l) // 2, (b - t) // 2)
        r = min(int(ext * scale), img.size[0] // 2, img.size[1] // 2)
        x = max(r, min(img.size[0] - r, cx))
        y = max(r, min(img.size[1] - r, cy))
        crop = (int(x - r), int(y - r), int(x + r), int(y + r))
        print(img.size, crop)

        with tempfile.NamedTemporaryFile("w", suffix=".mp4") as tmp_file:
            cmd = f"ffmpeg -i {video_path} -r {fps} -y -hide_banner -loglevel error {tmp_file.name}"
            assert subprocess.run(shlex.split(cmd)).returncode == 0
            for i, frame in enumerate(tqdm(video_stream(tmp_file.name), total=get_video_num_frames(tmp_file.name))):
                img = Image.fromarray(frame)
                img = img.crop(crop).resize((output_size, output_size), Image.Resampling.LANCZOS)
                yield np.array(img)
                # if i > 10:
                #     break


def main():
    video_paths = get_video_paths(INPUT_DIR)
    cropper = VideoCropper()
    output_dir = f"./output"
    os.makedirs(output_dir, exist_ok=True)
    for video_path in video_paths:
        video_name: str = os.path.basename(video_path)
        if not video_name.startswith("cam"):
            continue
        print(video_path)
        output_path = os.path.join(output_dir, video_name)
        cropper.crop_video(video_path, output_path)


if __name__ == "__main__":
    INPUT_DIR = "/home/juyonggroup/shared3dv/dataset/orz/raw_dataset/089/EXP-8-jaw-1"
    # need to run: ml cudnn/8.9.1.23/cuda12
    main()
