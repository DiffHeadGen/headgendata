from functools import cached_property
import os
import shutil

from tqdm import tqdm
from expdataloader.HeadGenLoader import RowData, HeadGenLoader
from expdataloader.Retarget import Retargeter
from expdataloader.HeadGenLoader import OutputData
from expdataloader.dataset import InputData
from expdataloader.utils import change_extension, get_image_paths, get_sub_dir, merge_video, count_images, count_files


class P4DOutputData(OutputData):
    def __init__(self, base_dir, data_name):
        super().__init__(base_dir, data_name)

    @cached_property
    def crop_dir(self):
        return get_sub_dir(self.output_dir, "crop")

    @cached_property
    def align_images_dir(self):
        return get_sub_dir(self.crop_dir, "align_images")

    @cached_property
    def align_image_paths(self):
        return get_image_paths(self.align_images_dir)

    @cached_property
    def cropped_imgs_dir(self):
        return os.path.join(self.crop_dir, "align_images")

    @property
    def num_cropped_imgs(self):
        return count_images(self.cropped_imgs_dir)

    @property
    def cropped_img_paths(self):
        # This is dynamic, cannot use cached_property
        return get_image_paths(self.cropped_imgs_dir)

    def merge_cropped_frames(self):
        merge_video(f"{self.cropped_imgs_dir}/%6d.png", f"{self.crop_dir}/crop.mp4")

    @cached_property
    def bfm2flame_params_dir(self):
        return get_sub_dir(self.crop_dir, "bfm2flame_params_simplified")

    @property
    def num_bfm2flame_params(self):
        return count_files(self.bfm2flame_params_dir)

    @cached_property
    def retarget_imgs_dir(self):
        return get_sub_dir(self.output_dir, "retarget_imgs")

    @cached_property
    def crop_params_dir(self):
        return os.path.join(self.crop_dir, "crop_params")

class P4DRowData(RowData):
    def __init__(self, source: InputData, target: InputData, output: OutputData):
        super().__init__(source, target, output)
        self.output = P4DOutputData(output.base_dir, output.data_name)
        self.source_output = P4DOutputData(get_sub_dir(output.base_dir, "source"), source.data_name)

    @property
    def is_img_generated(self):
        return self.num_frames == self.output.num_ori_output

    @cached_property
    def source_name(self):
        return os.path.splitext(os.path.basename(self.source_img_path))[0]

    @property
    def is_img_aligned(self):
        return self.target.num_frames == self.output.num_cropped_imgs

    @property
    def is_bfm2flame_params_ready(self):
        return self.num_frames == self.output.num_bfm2flame_params


TEST_INPUT = InputData(dataset_dir="../test_data", data_name="id1")
TEST_OUTPUT = P4DOutputData(base_dir="../test_data", data_name="id1")
TEST_ROW_DATA_ID1 = P4DRowData(TEST_INPUT, TEST_INPUT, TEST_OUTPUT)


class P4DLoader(HeadGenLoader[P4DRowData]):
    def __init__(self, name="Protrait4Dv2"):
        super().__init__(name, P4DRowData)


class P4DRetargetLoader(P4DLoader):
    def __init__(self, name="Protrait4Dv2"):
        super().__init__(name)

    def run_video(self, row):
        self.retarget_video(row)
