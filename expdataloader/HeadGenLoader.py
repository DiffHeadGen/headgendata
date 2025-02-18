from dataclasses import dataclass
from functools import cached_property
import os
from pathlib import Path

from natsort import natsorted
from expdataloader.utils import count_images, extract_all_frames, get_image_paths, get_sub_dir, merge_video, FileLock
import traceback
import shutil

DATA_DIR = (Path(__file__).parent.parent / "data").__str__()
VFHQ_DIR = (Path(__file__).parent.parent / "data/VFHQ_testset").__str__()


class RowData:
    def __init__(
        self,
        data_name: str,
        base_output_dir: str = None,
        source_img_path: str = None,
        base_dir: str = VFHQ_DIR,
    ):
        self.data_name = data_name
        self.base_dir = os.path.join(base_dir, data_name)
        assert base_output_dir is not None, "base_output_dir is required"
        self.base_output_dir = base_output_dir
        self._source_img_path = source_img_path

    @cached_property
    def results_dir(self):
        return get_sub_dir(self.base_output_dir, "results")

    @cached_property
    def fast_review_dir(self):
        return get_sub_dir(self.base_output_dir, "fast_review")

    @property
    def is_processed(self):
        return self.output_video_path and os.path.exists(self.output_video_path)

    @property
    def output_video_path(self):
        return os.path.join(self.results_dir, self.data_name, "output.mp4")

    @property
    def frames_dir(self):
        return get_sub_dir(self.output_dir, "frames")

    def human_output(self):
        if os.path.exists(self.output_video_path):
            shutil.copyfile(self.output_video_path, self.fast_review_video_path)
            extract_all_frames(self.output_video_path, self.frames_dir)

    @property
    def fast_review_video_path(self):
        return os.path.join(self.fast_review_dir, self.video_name)

    def copy_output2fast_review(self):
        if os.path.exists(self.output_video_path):
            shutil.copyfile(self.output_video_path, self.fast_review_video_path)

    @cached_property
    def output_dir(self):
        return get_sub_dir(self.results_dir, self.data_name)

    @cached_property
    def ori_imgs_dir(self):
        return os.path.join(self.base_dir, "ori_imgs")

    @cached_property
    def ori_img_paths(self):
        return get_image_paths(self.ori_imgs_dir)

    @cached_property
    def num_frames(self):
        return count_images(self.ori_imgs_dir)

    @property
    def target_img_paths(self):
        return self.ori_img_paths

    @property
    def target_video_path(self):
        path = os.path.join(self.base_dir, "video.mp4")
        if not os.path.exists(path):
            merge_video(f"{self.ori_imgs_dir}/%06d{self.img_ext}", path)
        return path

    @property
    def source_img_path(self) -> str:
        if self._source_img_path is not None and os.path.exists(self._source_img_path):
            return self._source_img_path
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
        self.output_dir = get_sub_dir(self.base_dir, "test20250218", self.name)
        self.lock_dir = get_sub_dir(self.output_dir, "lock")

    def get_output_dir(self, out_name):
        return get_sub_dir(self.output_dir, out_name, self.name)

    def create_row(self, data_name):
        return RowData(data_name, self.output_dir)

    def get_all_data_rows(self):
        for data_name in natsorted(os.listdir(VFHQ_DIR)):
            yield self.create_row(data_name)

    @cached_property
    def all_data_rows(self):
        return list(self.get_all_data_rows())

    @cached_property
    def all_data_rows_dict(self):
        return {row.name: row for row in self.all_data_rows}

    def get_run_data_rows(self):
        return [row for row in self.all_data_rows if not row.is_processed]

    def print_info(self):
        for i, row in enumerate(self.all_data_rows):
            print(i, row.info)

    def run_all(self):
        for row in self.get_run_data_rows():
            print(f"Processing: {row}")
            self.exp_data_row(row)

    @cached_property
    def test_data_rows(self):
        ids = ["Clip+RUcLuQ17UV8+P0+C1+F29582-29745", "Clip+WDN72QkW5KQ+P3+C0+F95232-95342"]
        return [self.all_data_rows_dict[id] for id in ids]

    @cached_property
    def test_20250218_data_row(self):
        id = "Clip+moIOVVEIffQ+P0+C1+F25505-25726"
        row = self.all_data_rows_dict[id]
        row._source_img_path = os.path.join(self.base_dir, "test20250218", "000000.jpg")
        return row

    def test_20250218(self):
        row = self.test_20250218_data_row
        self.run_video(row)
        shutil.copy(row.target_video_path, row.output_dir)


    def run_test(self):
        for row_name in self.test_data_rows:
            row = self.all_data_rows_dict[row_name]
            self.exp_data_row(row)

    def merge_video(self, image_dir, out_video_path):
        merge_video(f"{image_dir}/%06d.jpg", out_video_path)
        return out_video_path

    def run_video(self, row: RowData):
        raise NotImplementedError()

    def exp_data_row(self, row: RowData):
        lock_file = os.path.join(self.lock_dir, f"{row.name}.lock")
        with FileLock(lock_file) as lock:
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
                    traceback.print_exc()
                print(f"Error occurred: {e}. Logged to {error_file_path}")

    def print_summary(self):
        print(f"Name: {self.name}")
        print(f"    Total: {len(self.all_data_rows)}")
        print(f"    Processed: {len([row for row in self.all_data_rows if row.is_processed])}")
        print(f"    Unprocessed: {len([row for row in self.all_data_rows if not row.is_processed])}")
        print(f"    Erorr: {len([file for file in os.listdir(self.lock_dir) if file.endswith('.error')])}")


if __name__ == "__main__":
    print(DATA_DIR)
    loader = HeadGenLoader("test")
    for row in loader.all_data_rows:
        row.target_video_path
    # row = loader.all_data_rows[0]
    # loader.exp_data_row(row)
