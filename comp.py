from functools import cached_property
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import numpy as np
from tqdm import tqdm
from expdataloader import *
from expdataloader.utils import img_grid

HeadGenLoader


class CompLoader(RowDataLoader):
    def __init__(self, name="comp"):
        super().__init__(name)

    @cached_property
    def exp_name_list(self):
        return ["ours", "Protrait4Dv2", "VOODOO3D", "GAGAvatar", "FollowYourEmoji", "ROME", "XPortrait"]

    @cached_property
    def label_list(self):
        return ["source", "target"] + self.exp_name_list

    @cached_property
    def all_loaders(self):
        return [RowDataLoader(exp_name) for exp_name in self.exp_name_list]

    def merge_video(self, dir, out_path, fps=25):
        dir_name = os.path.dirname(out_path)
        os.makedirs(dir_name, exist_ok=True)
        # 使用 -pattern_type glob 处理非连续命名的文件
        ffmpeg_cmd = f"ffmpeg -framerate {fps} -pattern_type glob -i '{dir}/*.jpg' -c:v libx264 -pix_fmt yuv420p -loglevel error '{out_path}' -y"
        assert subprocess.run(shlex.split(ffmpeg_cmd)).returncode == 0
        print(f"see {out_path}")

    def run_video(self, row):
        idx = range(0, row.num_frames - 2, 4)
        for id in idx:
            img_path_list = []
            img_path_list.append(row.source_img_path)
            target_img_path = row.target.img_paths[id]
            img_path_list.append(target_img_path)
            for loader in self.all_loaders:
                input_row = loader.get_row(row.data_name)
                frames_dir = input_row.frames_dir
                ext = os.path.splitext(input_row.output.img_paths[0])[1]
                img_path = change_extension(os.path.join(frames_dir, os.path.basename(target_img_path)), ext)
                img_path_list.append(img_path)
            output_name = change_extension(os.path.basename(target_img_path), ".jpg")
            output_path = os.path.join(row.frames_dir, output_name)
            img_array = np.array(img_path_list).reshape(3, 3)
            img_grid(img_array, save_path=output_path)
        self.merge_video(row.frames_dir, row.output_video_path, fps=6)
        row.output.fast_review()


class CompLoaderImage(RowDataLoader):
    @cached_property
    def exp_name_list(self):
        return ["ROME", "Protrait4Dv2", "VOODOO3D", "GAGAvatar", "FollowYourEmoji", "XPortrait", "ours"]

    @cached_property
    def all_loaders(self):
        return [RowDataLoader(exp_name) for exp_name in self.exp_name_list]

    @cached_property
    def comp_cross_data(self):
        data = []
        with open("comp_cross.txt") as f:
            lines = f.readlines()
            for line in lines:
                path = Path(line.strip())
                data.append((path.parent.parent.name, path.name))
        return data
        # return [
        #     ("323_EXP-5-mouth_cam_222200047", "000236.jpg"),
        #     ("321_EXP-8-jaw-1_cam_222200042", "000124.jpg"),
        #     ("324_EXP-8-jaw-1_cam_222200039", "000044.jpg"),
        #     ("325_EXP-8-jaw-1_cam_222200045", "000112.jpg"),
        #     ("332_EXP-8-jaw-1_cam_222200038", "000080.jpg"),
        #     ("369_EXP-8-jaw-1_cam_222200048", "000092.jpg"),
        #     ("370_EXP-2-eyes_cam_221501007", "000096.jpg"),
        #     ("375_EXP-2-eyes_cam_222200036", "000100.jpg"),
        # ]

    def comp_cross_img(self):
        img_path_list = []
        for data_name, img_name in self.comp_cross_data:
            for loader in self.all_loaders:
                row = loader.get_row(data_name)
                img_path = os.path.join(row.frames_dir, img_name)
                img_path_list.append(img_path)
        img_array = np.array(img_path_list).reshape(-1, 7)
        img_grid(img_array, save_path=os.path.join(self.comp_cross_dir, "comp_cross.jpg"))

    def comp_cross_img_rs(self):
        img_path_list = []
        for data_name, img_name in self.comp_cross_data:
            row = self.get_row(data_name)
            img_path_list.append(row.source_img_path)
            img_path = os.path.join(row.target.imgs_dir, change_extension(img_name, row.target.img_ext))
            img_path_list.append(img_path)
        img_array = np.array(img_path_list).reshape(-1, 2)
        img_grid(img_array, save_path=os.path.join(self.comp_cross_dir, "comp_cross_rs.jpg"))

    @cached_property
    def comp_cross_dir(self):
        return get_sub_dir("data", "comp_cross_v2")

    def comp_cross_img_all(self):
        save_dir = self.comp_cross_dir
        for data_name, img_name in tqdm(self.comp_cross_data):
            row = self.get_row(data_name)
            paths = [(row.source_img_path, "source"), (os.path.join(row.target.imgs_dir, change_extension(img_name, row.target.img_ext)), "driving")]
            for loader in self.all_loaders:
                row = loader.get_row(data_name)
                paths.append((os.path.join(row.frames_dir, img_name), loader.name))
            for img_path, subdir in paths:
                os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
                save_path = os.path.join(save_dir, subdir, f"{data_name}+{img_name}")
                shutil.copyfile(img_path, save_path)

    def comp_cross(self):
        self.comp_cross_img()
        self.comp_cross_img_rs()
        self.comp_cross_img_all()


def comp():
    loader = CompLoader()
    # loader.all_data_rows[0].output.fast_review()
    # loader.run_video(loader.all_data_rows[0])
    loader.run_all()


def main():
    loader = CompLoaderImage("comp")
    loader.comp_cross()
    # for data in loader.comp_cross_data:
    #     print(data)


if __name__ == "__main__":
    main()
