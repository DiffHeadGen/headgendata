from functools import cached_property
import os
import shutil

from tqdm import tqdm
from expdataloader.HeadGenLoader import RowData, HeadGenLoader
from expdataloader.Retarget import Retargeter
from expdataloader.utils import change_extension, get_image_paths, get_sub_dir, merge_video, count_images, count_files


class P4DRowData(RowData):
    pass

    @cached_property
    def output_crop_dir(self):
        return get_sub_dir(self.output_dir, "crop")

    @cached_property
    def ori_output_dir(self):
        return get_sub_dir(self.output_dir, "ori_output")

    @cached_property
    def ori_output_comp_dir(self):
        return get_sub_dir(self.output_dir, "ori_output_comp")

    @property
    def ori_output_video_path(self):
        return os.path.join(self.output_dir, "ori_output.mp4")

    @property
    def ori_output_comp_video_path(self):
        return os.path.join(self.output_dir, "ori_output_comp.mp4")

    @property
    def is_img_generated(self):
        return len(count_images(self.ori_output_dir)) == self.num_frames

    def merge_ori_output_video(self):
        merge_video(f"{self.ori_output_dir}/%6d.jpg", self.ori_output_video_path)

    def merge_ori_output_comp_video(self):
        merge_video(f"{self.ori_output_comp_dir}/%6d.jpg", self.ori_output_comp_video_path)

    @cached_property
    def align_images_dir(self):
        return get_sub_dir(self.output_crop_dir, "align_images")

    @cached_property
    def align_image_paths(self):
        return get_image_paths(self.align_images_dir)

    @cached_property
    def source_img_path(self):
        return self.align_image_paths[0]

    @cached_property
    def source_name(self):
        return os.path.splitext(os.path.basename(self.source_img_path))[0]

    @property
    def is_img_aligned(self):
        return count_images(self.ori_imgs_dir) == count_images(self.cropped_imgs_dir)

    @property
    def cropped_img_paths(self):
        # This is dynamic, cannot use cached_property
        return get_image_paths(self.cropped_imgs_dir)

    @cached_property
    def cropped_imgs_dir(self):
        return os.path.join(self.output_crop_dir, "align_images")

    @cached_property
    def crop_params_dir(self):
        return os.path.join(self.output_crop_dir, "crop_params")

    @cached_property
    def retarget_imgs_dir(self):
        return get_sub_dir(self.output_dir, "retarget_imgs")

    def merge_cropped_frames(self):
        merge_video(f"{self.cropped_imgs_dir}/%6d.png", f"{self.output_crop_dir}/crop.mp4")

    @cached_property
    def bfm2flame_params_dir(self):
        return get_sub_dir(self.output_crop_dir, "bfm2flame_params_simplified")

    @property
    def is_bfm2flame_params_ready(self):
        return self.num_frames == len(count_files(self.bfm2flame_params_dir))


TEST_ROW_DATA_ID1 = P4DRowData(data_name="id1", base_output_dir="../test_data", base_dir="../test_data")


class P4DLoader(HeadGenLoader):
    def __init__(self, name="Protrait4Dv2"):
        super().__init__(name)

    def create_row(self, data_name) -> P4DRowData:
        return P4DRowData(data_name, self.output_dir)

    def run_video(self, row_data: P4DRowData):
        return super().run_video(row_data)


class RetargetLoader(P4DLoader):
    def __init__(self, name="Protrait4Dv2"):
        super().__init__(name)

    @cached_property
    def retargeter(self):
        return Retargeter()

    def run_video(self, row: P4DRowData):
        self.retarget_row_imgs(row, row.ori_output_dir, row.frames_dir)
        merge_video(f"{row.frames_dir}/%6d.jpg", row.output_video_path)
        row.copy_output2fast_review()

    def retarget_row_imgs(self, row: P4DRowData, cropped_imgs_dir, output_dir):
        for ori_img_path, cropped_img_path in tqdm(zip(row.ori_img_paths, get_image_paths(cropped_imgs_dir)), total=row.num_frames):
            name = os.path.basename(ori_img_path)
            output_path = os.path.join(output_dir, change_extension(name, ".jpg"))
            self.retargeter.retarget(ori_img_path, cropped_img_path, output_path)

    def clear_output(self, row: P4DRowData):
        shutil.rmtree(row.frames_dir)
        if os.path.exists(row.output_video_path):
            os.remove(row.output_video_path)

    def clear_all_output(self):
        for row in self.all_data_rows:
            print("Clearing", row)
            self.clear_output(row)
