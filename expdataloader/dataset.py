from enum import Enum
from functools import cached_property
import os
from pathlib import Path

from natsort import natsorted

from expdataloader.utils import count_images, get_image_paths, get_sub_dir, merge_video


BASE_DIR = Path(__file__).parent.parent / "data"
VFHQ_DIR = str(BASE_DIR / "VFHQ_testset")
TEMP_TEST_DIR = str(BASE_DIR / "temp_testset")
ORZ_DIR = str(BASE_DIR / "orz_testset")
COMBINED_DIR = str(BASE_DIR / "combined_testset")


class InputData:
    def __init__(self, dataset_dir, data_name):
        self.data_name = data_name
        self.dataset_dir = dataset_dir
        self.base_dir = os.path.join(dataset_dir, data_name)
        self.source_img_path = None

    @cached_property
    def imgs_dir(self):
        return os.path.join(self.base_dir, "ori_imgs")

    @cached_property
    def img_paths(self):
        return get_image_paths(self.imgs_dir)

    @property
    def first_img_path(self):
        return self.img_paths[0]

    @cached_property
    def num_frames(self):
        return count_images(self.imgs_dir)

    @cached_property
    def video_path(self):
        path = os.path.join(self.base_dir, "video.mp4")
        if not os.path.exists(path):
            merge_video(f"{self.imgs_dir}/%06d{self.img_ext}", path)
        return path

    @property
    def img_ext(self):
        return os.path.splitext(self.first_img_path)[1]


class InputDataSet:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def get(self, row_name):
        return self.dict[row_name]

    def get_all(self):
        for row_name in natsorted(os.listdir(self.dataset_dir)):
            yield InputData(self.dataset_dir, row_name)

    @cached_property
    def values(self):
        return list(self.get_all())

    @cached_property
    def dict(self):
        return {row.data_name: row for row in self.values}

    @cached_property
    def num_frames_all(self):
        frames = 0
        for data in self.values:
            frames += data.num_frames
        return frames


class VFHQTestDataSet(InputDataSet):
    def __init__(self):
        super().__init__(VFHQ_DIR)

    def get_all(self):
        for row in super().get_all():
            if row.data_name.startswith("Clip"):
                yield row

    @property
    def test_data_rows(self):
        ids = [
            "Clip+RUcLuQ17UV8+P0+C1+F29582-29745",
            "Clip+WDN72QkW5KQ+P3+C0+F95232-95342",
        ]
        return [self.get(id) for id in ids]


VFHQ_TEST_DATASET = VFHQTestDataSet()


class ORZTestDataSet(InputDataSet):
    def __init__(self):
        super().__init__(ORZ_DIR)

    def get_all(self):
        for row in super().get_all():
            if "EXP" in row.data_name:
                row.source_img_path = os.path.join(row.base_dir, "source.jpg")
                yield row


ORZ_TEST_DATASET = ORZTestDataSet()


class CombinedTestDataSet(InputDataSet):
    def __init__(self):
        super().__init__(COMBINED_DIR)

    def get_all(self):
        for row in super().get_all():
            if "EXP" in row.data_name or "Clip" in row.data_name:
                row.source_img_path = os.path.join(row.base_dir, "source_cross.jpg")
                yield row


COMBINED_TEST_DATASET = CombinedTestDataSet()


class TempTestDataSet(InputDataSet):
    def __init__(self):
        super().__init__(TEMP_TEST_DIR)


TEMP_TEST_DATASET = TempTestDataSet()

class DataSetNames(Enum):
    VFHQ = "VFHQ"
    TEMP = "TEMP"
    ORZ = "ORZ"
    COMBINED = "COMBINED"