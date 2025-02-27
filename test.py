import os
import shutil
from expdataloader import HeadGenLoader
from expdataloader.utils import get_sub_dir, get_first_mp4_file

def num_all_frames():
    loader = HeadGenLoader("test")
    num_frames = 0
    for row in loader.all_data_rows:
        num_frames += row.num_frames
    print(num_frames)
    
def xportrait():
    loader = HeadGenLoader("XPortrait")
    for row in loader.all_data_rows:
        if row.is_processed:
            continue
        output_video_path = get_first_mp4_file(row.output_dir)
        if not output_video_path:
            continue
        print(row)
        shutil.copy(output_video_path,row.output_video_path)
        row.output.human()

def mp_ldmks():
    loader = HeadGenLoader("test")
    no_ldmks = []
    for row in loader.all_data_rows:
        path = os.path.join(row.base_dir, "mp_ldmks/000000.mp")
        if not os.path.exists(path):
            # print(row, path)
            print(row.data_name)
            no_ldmks.append(row.data_name)
    print(len(no_ldmks))

def fye_error():
    data = [
        ("Clip+G0DGRma_p48+P0+C0+F11208-11383",[29]),
        ("Clip+moIOVVEIffQ+P0+C1+F25505-25726",[180,181,182,185,186])
    ]
    from expdataloader.dataset import VFHQ_TEST_DATASET
    dataset = VFHQ_TEST_DATASET
    for id, frame_ids, in data:
        input = dataset.get(id)
        for i in frame_ids:
            print(input.img_paths[i])
        
if __name__ == '__main__':
    # mp_ldmks()
    fye_error()