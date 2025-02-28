from functools import cached_property
import os
from expdataloader import *


def main():
    # loader = P4DLoader()
    # loader.clear_all_output()
    loader = P4DRetargetLoader()
    loader.run_all()


def clear():
    loader = P4DLoader()
    loader.clear_all_output()


class OursRetargeter(RowDataLoader):
    def __init__(self, name="ours1"):
        super().__init__(name)

    @cached_property
    def retargeter(self):
        return Retargeter(black=False)

    def run_video(self, row):
        ori_output_dir = os.path.join(row.output.base_dir, "ori_output", row.output.data_name, "frames")
        self.retarget_row_imgs(row, ori_output_dir, row.frames_dir)
        self.merge_video(row.frames_dir, row.output_video_path)
        row.output.fast_review()

    def test(self):
        row = self.get_row("Clip+y5OFsRIRkwc+P0+C0+F9797-9938")
        self.run_video(row)

if __name__ == "__main__":
    loader = OursRetargeter()
    loader.run_all()
