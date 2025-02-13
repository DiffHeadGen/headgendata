from dataclasses import dataclass
from functools import cached_property
import os
from pathlib import Path

from natsort import natsorted
from expdataloader.utils import get_image_paths, get_sub_dir, merge_video, FileLock
import traceback

DATA_DIR = (Path(__file__).parent.parent / "data").__str__()
VFHQ_DIR = (Path(__file__).parent.parent / "data/VFHQ_testset").__str__()


class RowData:
    def __init__(self, data_name: str, base_output_dir:str=None):
        self.base_dir = os.path.join(VFHQ_DIR, data_name)
        self.data_name = data_name
        self.base_output_dir = base_output_dir

    @property
    def is_processed(self):
        return self.output_video_path and os.path.exists(self.output_video_path)
    
    @property
    def output_video_path(self):
        return os.path.join(self.base_output_dir, self.video_name)  
    
    @cached_property
    def output_dir(self):
        return get_sub_dir(self.base_output_dir, self.data_name)

    @cached_property
    def ori_imgs_dir(self):
        return os.path.join(self.base_dir, "ori_imgs")

    @cached_property
    def ori_img_paths(self):
        return get_image_paths(self.ori_imgs_dir)

    @property
    def target_img_paths(self):
        return self.ori_img_paths

    @property
    def target_video_path(self):
        path = os.path.join(self.base_dir, "video.mp4")
        if not os.path.exists(path):
            merge_video(f"{self.ori_imgs_dir}/%06d{self.img_ext}", path)
        return path

    @cached_property
    def source_img_path(self) -> str:
        return self.ori_img_paths[0]

    @property
    def name(self):
        return self.data_name

    @cached_property
    def img_ext(self):
        return os.path.splitext(self.source_img_path)[-1]

    @property
    def info(self):
        return f"RowData({self.name}), imgs: {len(self.ori_img_paths)}, ext:{self.img_ext}, processed: {self.is_processed}"

    @property
    def video_name(self):
        return f"{self.name}.mp4"

    def __str__(self):
        return f"RowData({self.name})"


class HeadGenLoader:
    def __init__(self, name: str):
        self.base_dir = DATA_DIR
        self.name = name
        self.output_dir = get_sub_dir(self.base_dir, self.name)
        self.lock_dir = get_sub_dir(self.output_dir, "lock")

    def get_all_data_rows(self):
        for data_name in natsorted(os.listdir(VFHQ_DIR)):
            row = RowData(data_name, self.output_dir)
            yield row

    @cached_property
    def all_data_rows(self):
        return list(self.get_all_data_rows())

    def get_run_data_rows(self):
        return [row for row in self.all_data_rows if not row.is_processed]

    def print_info(self):
        for i, row in enumerate(self.all_data_rows):
            print(i, row.info)

    def run_all(self):
        for row in self.get_run_data_rows():
            print(f"Processing: {row}")
            self.exp_data_row(row)

    def merge_video(self, image_dir, out_video_path):
        merge_video(f"{image_dir}/%06d.jpg", out_video_path)
        return out_video_path

    def run_video(self, row: RowData):
        raise NotImplementedError()

    def exp_data_row(self, row: RowData):
        lock = FileLock(os.path.join(self.lock_dir, f"{row.name}.lock"))
        if not lock.acquire():
            print(f"Already running, skip: {row.name}")
            return

        if row.is_processed:
            print(f"out_video_path exists: {row.output_video_path}")
            return

        try:
            print(f"Running: {row}")
            self.run_video(row)
        except Exception as e:
            error_file_path = os.path.join(self.lock_dir, f"{row.name}.error")
            with open(error_file_path, "w") as f:
                f.write(f"{e}\n")
                traceback.print_exc(file=f)
            print(f"Error occurred: {e}. Logged to {error_file_path}")
        finally:
            lock.release()


if __name__ == "__main__":
    print(DATA_DIR)
    loader = HeadGenLoader("test")
    for row in loader.all_data_rows:
        row.target_video_path
    # row = loader.all_data_rows[0]
    # loader.exp_data_row(row)
