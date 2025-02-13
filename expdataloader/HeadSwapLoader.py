import os
from pathlib import Path
from .utils import get_image_paths, get_sub_dir, extract_all_frames, merge_video
from itertools import islice
import traceback
from expdataloader.utils import FileLock

EXP_DATA_DIR = Path(__file__).parent.parent.__str__()


def get_pid(image_path):
    return os.path.basename(os.path.dirname(image_path))


def get_hsid(video_path):
    return os.path.basename(os.path.dirname(video_path))


class OriImageLoader:
    def __init__(self, hsid: str):
        assert hsid.startswith("HSID"), "HSID name should start with HSID"
        self.base_dir = get_sub_dir(EXP_DATA_DIR, hsid)
        self.ori_imgs_dir = get_sub_dir(self.base_dir, "ori_imgs")

    def get_image_paths(self):
        images_path = get_image_paths(self.ori_imgs_dir)
        if len(images_path) == 0:
            self.extract_imgs()
            images_path = get_image_paths(self.ori_imgs_dir)
        return images_path

    def extract_imgs(self):
        video_path = os.path.join(self.base_dir, "face_crop.mp4")
        assert os.path.exists(video_path), f"video not exists: {video_path}"
        extract_all_frames(video_path, self.ori_imgs_dir)


class ExpDataLoader:
    def __init__(self, name: str):
        self.base_dir = EXP_DATA_DIR
        self.name = name
        self.output_dir = get_sub_dir(self.base_dir, self.name)
        self.output_video_name_template = "{name}.mp4"
        self.lock_dir = get_sub_dir(self.output_dir, "lock")

    def get_output_video_name(self, name):
        return self.output_video_name_template.format(name=name)

    def get_sub_dir(self, name):
        return get_sub_dir(self.output_dir, name)

    def get_run_pairs(self):
        for pair in self.get_all_pairs():
            name = pair.replace(" ", "_")
            out_path = os.path.join(self.output_dir, self.get_output_video_name(name))
            if not os.path.exists(out_path):
                yield pair

    def print_run_pairs(self):
        pairs = self.get_all_pairs()
        print(list(pairs))

    def get_all_pairs(self):
        for line in open(os.path.join(self.base_dir, "run_pair.txt")).readlines():
            pair = line.strip()
            if pair:
                yield pair

    def run_all(self, max_num=0):
        pairs = self.get_run_pairs()
        if max_num > 0:
            pairs = islice(pairs, max_num)
        for pair in pairs:
            print(pair)
            self.exp_pair(pair)

    def get_images_loader(self, video_path: str):
        image_loader = OriImageLoader(get_hsid(video_path))
        image_loader.get_image_paths()
        return image_loader

    def merge_video(self, swap_dir, out_video_path):
        merge_video(f"{swap_dir}/%06d.jpg", out_video_path)
        return out_video_path

    def run_video(self, source_img_path, target_video_path, out_video_path):
        print(f"Source Image Path: {source_img_path}")
        print(f"Target Video Path: {target_video_path}")
        print(f"Output Video Path: {out_video_path}")
        # print()
        return out_video_path

    def exp_pair(self, pair: str):
        source_name = pair.split(" ")[0]  # PID
        target_name = pair.split(" ")[1]  # HSID
        name = pair.replace(" ", "_")
        lock = FileLock(os.path.join(self.lock_dir, f"{name}.lock"))
        if not lock.acquire():
            print(f"skip {name}")
            return

        source_img_path = get_image_paths(os.path.join(self.base_dir, "pids", source_name))[0]
        target_video_path = os.path.join(self.base_dir, target_name, "face_crop.mp4")
        assert os.path.exists(target_video_path), "target video is not exists"
        name = pair.replace(" ", "_")
        out_video_path = os.path.join(self.output_dir, self.get_output_video_name(name))
        os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
        if os.path.exists(out_video_path):
            print(f"already done, see {out_video_path}")
            return
        try:
            out_video_path = self.run_video(source_img_path, target_video_path, out_video_path)
            print(f"see {out_video_path}")
        except Exception as e:
            traceback.print_exc()
        finally:
            lock.release()
