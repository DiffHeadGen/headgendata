from functools import cached_property
import os
from pathlib import Path

from natsort import natsorted
from tqdm import tqdm
from expdataloader.Retarget import Retargeter
from expdataloader.utils import change_extension, count_images, extract_all_frames, get_file_name_without_ext, get_image_paths, get_sub_dir, merge_video, FileLock
from expdataloader.dataset import InputData, VFHQ_TEST_DATASET, TEMP_TEST_DATASET, ORZ_TEST_DATASET, COMBINED_TEST_DATASET, InputDataSet
import traceback
import shutil
from typing import TypeVar, Generic, List

DATA_DIR = (Path(__file__).parent.parent / "data").__str__()
VFHQ_DIR = (Path(__file__).parent.parent / "data/VFHQ_testset").__str__()


class OutputData:
    def __init__(self, base_dir, data_name):
        self.base_dir = base_dir
        self.data_name = data_name

    @cached_property
    def results_dir(self):
        return get_sub_dir(self.base_dir, "results")

    @cached_property
    def fast_review_dir(self):
        return get_sub_dir(self.base_dir, "fast_review")

    @cached_property
    def output_dir(self):
        return get_sub_dir(self.results_dir, self.data_name)

    @cached_property
    def feature_dir(self):
        return get_sub_dir(self.output_dir, "feature")

    @cached_property
    def video_path(self):
        return os.path.join(self.results_dir, self.data_name, "output.mp4")

    @property
    def fast_review_video_path(self):
        return os.path.join(self.fast_review_dir, f"{self.data_name}.mp4")

    @cached_property
    def frames_dir(self):
        return get_sub_dir(self.output_dir, "frames")

    def human(self):
        if os.path.exists(self.video_path):
            shutil.copyfile(self.video_path, self.fast_review_video_path)
            extract_all_frames(self.video_path, self.frames_dir)

    def fast_review(self):
        if os.path.exists(self.video_path):
            shutil.copyfile(self.video_path, self.fast_review_video_path)

    @cached_property
    def ori_output_dir(self):
        return get_sub_dir(self.output_dir, "ori_output")
    
    @property
    def num_ori_output(self):
        return count_images(self.ori_output_dir)
    
    @cached_property
    def ori_output_comp_dir(self):
        return get_sub_dir(self.output_dir, "ori_output_comp")

    @property
    def ori_output_video_path(self):
        return os.path.join(self.output_dir, "ori_output.mp4")

    @property
    def ori_output_comp_video_path(self):
        return os.path.join(self.output_dir, "ori_output_comp.mp4")

    def merge_ori_output_video(self):
        merge_video(f"{self.ori_output_dir}/%6d.jpg", self.ori_output_video_path)

    def merge_ori_output_comp_video(self):
        merge_video(f"{self.ori_output_comp_dir}/%6d.jpg", self.ori_output_comp_video_path)

    def clear_ori_output(self):
        shutil.rmtree(self.ori_output_dir, ignore_errors=True)
        shutil.rmtree(self.ori_output_comp_dir, ignore_errors=True)
        try:
            os.remove(self.ori_output_video_path)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.ori_output_comp_video_path)
        except FileNotFoundError:
            pass

    @property
    def num_ori_output(self):
        return count_images(self.ori_output_dir)


class RowData:
    def __init__(self, source: InputData, target: InputData, output: OutputData):
        self.source = source
        self.target = target
        self.output = output
        self._source_img_path = None

    @property
    def data_name(self):
        return self.target.data_name

    @property
    def is_processed(self):
        return self.output_video_path and os.path.exists(self.output_video_path)

    @property
    def output_video_path(self):
        return self.output.video_path

    @property
    def output_dir(self):
        return self.output.output_dir

    @property
    def frames_dir(self):
        return self.output.frames_dir

    @cached_property
    def num_frames(self):
        return self.target.num_frames

    @property
    def source_img_path(self) -> str:
        return self.source.img_paths[0] if self._source_img_path is None else self._source_img_path

    @source_img_path.setter
    def source_img_path(self, img_path):
        self._source_img_path = img_path

    @property
    def source_name(self):
        return os.path.splitext(os.path.basename(self.source_img_path))[0]

    @cached_property
    def img_ext(self):
        return self.target.img_ext

    @property
    def info(self):
        return f"RowData({self.data_name}), imgs: {self.num_frames}, ext:{self.img_ext}, processed: {self.is_processed}"

    @property
    def video_name(self):
        return f"{self.data_name}.mp4"

    def __str__(self):
        return f"RowData({self.data_name})"

    @property
    def ori_output_dir(self):
        return self.output.ori_output_dir


TROW = TypeVar("TROW", bound=RowData)


class HeadGenLoader(Generic[TROW]):
    def __init__(self, name: str, row_type=RowData):
        self.base_dir = DATA_DIR
        self.name = name
        self.row_type = row_type
        self.exp_name = "combined_output"
        self.dataset:InputDataSet = COMBINED_TEST_DATASET

    @cached_property
    def output_dir(self):
        return get_sub_dir(self.base_dir, self.exp_name, self.name)

    @cached_property
    def lock_dir(self):
        return get_sub_dir(self.output_dir, "lock")

    def get_output_dir(self, out_name):
        return get_sub_dir(self.output_dir, out_name, self.name)

    def get_all_data_rows(self):
        for target in self.dataset.values:
            row = self.row_type(target, target, OutputData(self.output_dir, target.data_name))
            if os.path.exists(target.source_img_path):
                row.source_img_path = os.path.join(row.output_dir, os.path.basename(target.source_img_path))
                shutil.copyfile(target.source_img_path, row.source_img_path)
            yield row

    @cached_property
    def all_data_rows(self) -> List[TROW]:
        return list(self.get_all_data_rows())
    
    @cached_property
    def all_data_rows_dict(self):
        return {row.data_name: row for row in self.all_data_rows}
    
    def get_row(self, data_name):
        return self.all_data_rows_dict[data_name]

    def print_info(self):
        for i, row in enumerate(self.all_data_rows):
            print(i, row.info)

    def run_all(self):
        for row in self.all_data_rows:
            if row.is_processed:
                print(f"Skip: {row.data_name}")
                continue
            print(f"Processing: {row}")
            self.exp_data_row(row)

    @cached_property
    def test_20250218_row_data(self) -> TROW:
        self.exp_name = "test20250218"
        source = TEMP_TEST_DATASET.values[0]
        target = VFHQ_TEST_DATASET.get("Clip+moIOVVEIffQ+P0+C1+F25505-25726")
        output = OutputData(self.output_dir, target.data_name)
        row = self.row_type(source, target, output)
        return row

    def test_20250218(self):
        row = self.test_20250218_row_data
        self.run_video(row)
        shutil.copy(row.target.video_path, row.output_dir)

    def merge_video(self, image_dir, out_video_path):
        merge_video(f"{image_dir}/%06d.jpg", out_video_path)
        return out_video_path

    def run_video(self, row: TROW):
        raise NotImplementedError()

    def exp_data_row(self, row: TROW):
        lock_file = os.path.join(self.lock_dir, f"{row.data_name}.lock")
        with FileLock(lock_file) as lock:
            if not lock.acquire():
                print(f"Already running, skip: {row.data_name}")
                return

            if row.is_processed:
                print(f"Already processed, skip: {row.data_name}")
                return

            try:
                print(f"Running: {row}")
                self.run_video(row)
            except Exception as e:
                error_file_path = os.path.join(self.lock_dir, f"{row.data_name}.error")
                with open(error_file_path, "w") as f:
                    f.write(f"{e}\n")
                    traceback.print_exc(file=f)
                    traceback.print_exc()
                print(f"Error occurred: {e}. Logged to {error_file_path}")

    def retarget_video(self, row: TROW):
        self.retarget_row_imgs(row, row.ori_output_dir, row.frames_dir)
        self.merge_video(row.frames_dir, row.output_video_path)
        row.output.fast_review()

    @cached_property
    def retargeter(self):
        return Retargeter(black=True)

    def retarget_row_imgs(self, row: TROW, cropped_imgs_dir, output_dir):
        cropped_imgs_dict = {get_file_name_without_ext(img_path): img_path for img_path in get_image_paths(cropped_imgs_dir)}
        for img_path in tqdm(row.target.img_paths, total=row.num_frames, desc="Retargeting"):
            name = get_file_name_without_ext(img_path)
            if not name in cropped_imgs_dict:
                print(f"Missing cropped img: {name}")
                continue
            cropped_img_path = cropped_imgs_dict[name]
            output_path = os.path.join(output_dir, name+".jpg")
            self.retargeter.retarget(img_path, cropped_img_path, output_path)

    def clear_output(self, row: TROW):
        shutil.rmtree(row.frames_dir)
        if os.path.exists(row.output_video_path):
            os.remove(row.output_video_path)

    def clear_all_output(self):
        for row in self.all_data_rows:
            print("Clearing", row)
            self.clear_output(row)

    def print_summary(self):
        print(f"Name: {self.name}")
        print(f"    Total: {len(self.all_data_rows)}")
        print(f"    Processed: {len([row for row in self.all_data_rows if row.is_processed])}")
        print(f"    Unprocessed: {len([row for row in self.all_data_rows if not row.is_processed])}")
        print(f"    Erorr: {len([file for file in os.listdir(self.lock_dir) if file.endswith('.error')])}")


if __name__ == "__main__":
    print(DATA_DIR)
    loader = HeadGenLoader("test")
    # row = loader.all_data_rows[0]
    # loader.exp_data_row(row)
