from expdataloader import HeadGenLoader
from expdataloader.utils import get_sub_dir

if __name__ == '__main__':
    loader = HeadGenLoader("test")
    num_frames = 0
    for row in loader.all_data_rows:
        num_frames += len(row.ori_img_paths)
    print(num_frames)