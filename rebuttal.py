import os
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import face_alignment
from functools import cached_property
from tqdm import tqdm  # 添加tqdm导入


def solve_transform(source_pts, target_pts):
    """
    计算从source到target的变换矩阵
    """
    # 计算质心
    source_centroid = np.mean(source_pts, axis=0)
    target_centroid = np.mean(target_pts, axis=0)
    
    # 中心化
    source_centered = source_pts - source_centroid
    target_centered = target_pts - target_centroid
    
    # 计算缩放因子
    source_norm = np.linalg.norm(source_centered)
    target_norm = np.linalg.norm(target_centered)
    scale = target_norm / source_norm
    
    # 计算平移
    trans = target_centroid - scale * source_centroid
    
    return scale, trans


class Retargeter:
    def __init__(self, use_cache=False, black=False):
        self.use_cache = use_cache
        self.black = black
        pass

    @cached_property
    def landmark_detector(self):
        return face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)

    def get_landmarks(self, img_path):
        if self.use_cache:
            save_path = os.path.splitext(img_path)[0] + ".npy"
            if os.path.exists(save_path):
                ldmks = np.load(save_path)
                return ldmks
        ldmkss = self.landmark_detector.get_landmarks(img_path)
        ldmks = ldmkss[0]
        if self.use_cache:
            np.save(save_path, ldmks)
        return ldmks

    def retarget(self, source_img_path, target_img_path, output_path):
        source_ldmks = self.get_landmarks(source_img_path)
        target_ldmks = self.get_landmarks(target_img_path)
        scale, trans = solve_transform(target_ldmks[:, :2], source_ldmks[:, :2])
        source_img = Image.open(source_img_path)
        if self.black:
            source_img = Image.new("RGB", source_img.size, (0, 0, 0))
        target_img = Image.open(target_img_path)
        target_img = target_img.resize((int(target_img.width * scale), int(target_img.height * scale)), Image.Resampling.LANCZOS)
        paste_pos = trans.astype(int)
        source_img.paste(target_img, tuple(paste_pos))
        source_img.save(output_path)


def merge_images_to_video(img_dir, output_video_path, fps=25, pattern="%06d.jpg"):
    """
    通用的图片序列合并为视频的函数
    
    Args:
        img_dir: 图片目录路径
        output_video_path: 输出视频路径
        fps: 视频帧率，默认30fps
        pattern: 图片命名格式，默认%06d.jpg
    """
    print(f"开始合并图片为视频...")
    
    # 检查图片目录是否存在
    if not Path(img_dir).exists():
        print(f"图片目录不存在: {img_dir}")
        return
    
    # 获取图片列表
    img_files = sorted(list(Path(img_dir).glob("*.jpg")))
    
    if not img_files:
        print(f"在 {img_dir} 中没有找到图片")
        return
    
    print(f"找到 {len(img_files)} 张图片，将合并为视频")
    
    # 构建ffmpeg命令
    cmd = [
        "ffmpeg",
        "-framerate", str(fps),  # 设置帧率
        "-i", str(Path(img_dir) / pattern),  # 输入图片序列
        "-c:v", "libx264",  # 使用H.264编码
        "-preset", "medium", # 编码预设
        "-crf", "23",        # 质量设置
        "-pix_fmt", "yuv420p", # 像素格式
        str(output_video_path),
        "-y"                 # 覆盖已存在的文件
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"成功合并视频到: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"合并视频失败: {e}")
        print(f"错误输出: {e.stderr.decode()}")


class VideoProcessor:
    def __init__(self, video_file, data_dir="data/rebuttal"):
        """
        初始化视频处理器
        
        Args:
            video_file: 视频文件名（在ori_video_dir中的文件名）
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.ori_video_dir = self.data_dir / "output"
        self.retarget_dir = self.data_dir / "retarget"
        
        # 获取视频文件名（去掉.mp4后缀）作为输出文件夹名
        self.video_name = Path(video_file).stem  # 去掉.mp4后缀
        self.video_output_dir = self.retarget_dir / self.video_name
        
        # 确保目录存在
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        (self.video_output_dir / "source_img").mkdir(exist_ok=True)
        (self.video_output_dir / "result_img").mkdir(exist_ok=True)
        
        # 视频文件路径
        self.video_path = self.ori_video_dir / video_file
        
        # 属性
        self.source_img_dir = self.video_output_dir / "source_img"
        self.result_img_dir = self.video_output_dir / "result_img"
        self.retarget_img_dir = self.video_output_dir / "retarget_img"
        
        # 确保retarget_img目录存在
        self.retarget_img_dir.mkdir(exist_ok=True)
    
    def extract_imgs(self, video_path, output_dir, pattern="%06d.jpg"):
        """
        使用ffmpeg提取视频的所有帧为图片
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出图片目录
            pattern: 图片命名格式，默认从0开始编号
        """
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-q:v", "2",     # 高质量
            "-start_number", "0",  # 从0开始编号
            str(output_dir / pattern),
            "-y"             # 覆盖已存在的文件
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"成功提取所有帧到: {output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"提取图片失败: {e}")
            print(f"错误输出: {e.stderr.decode()}")
    
    def extract_first_frame(self, video_path, output_path):
        """
        提取视频的第一帧
        
        Args:
            video_path: 输入视频路径
            output_path: 输出图片路径
        """
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vframes", "1",  # 只提取第一帧
            "-q:v", "2",      # 高质量
            str(output_path),
            "-y"              # 覆盖已存在的文件
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"成功提取第一帧到: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"提取第一帧失败: {e}")
            print(f"错误输出: {e.stderr.decode()}")
    
    def crop_video(self, input_path, output_path, x, y, width, height):
        """
        裁剪视频
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            x, y: 裁剪起始位置
            width, height: 裁剪尺寸
        """
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-vf", f"crop={width}:{height}:{x}:{y}",
            "-c:v", "libx264",  # 使用H.264编码
            "-preset", "medium", # 编码预设
            "-crf", "23",        # 质量设置
            str(output_path),
            "-y"                 # 覆盖已存在的文件
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"成功裁剪视频到: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"裁剪视频失败: {e}")
            print(f"错误输出: {e.stderr.decode()}")
    
    def align_images(self, use_cache=True, skip_if_exists=True):
        """
        将result_img_dir中的图片对齐到source_img_dir中对应编号的图片
        使用Retargeter进行人脸对齐，输出到retarget_img_dir
        
        Args:
            use_cache: 是否使用缓存
            skip_if_exists: 如果已经运行过是否跳过
        """
        print(f"开始对齐图片...")
        
        # 如果启用跳过选项，检查是否已经运行过
        if skip_if_exists:
            # 检查retarget_img目录是否存在
            if self.retarget_img_dir.exists():
                # 获取对齐后的图片数量
                retarget_imgs = list(self.retarget_img_dir.glob("*.jpg"))
                # 检查retarget.mp4是否存在
                retarget_video = self.video_output_dir / "retarget.mp4"
                
                # 获取输入的result图片数量
                result_imgs = list(self.result_img_dir.glob("*.jpg"))
                
                if (retarget_video.exists() and 
                    len(retarget_imgs) > 0 and 
                    len(retarget_imgs) == len(result_imgs)):
                    print(f"检测到已存在的对齐结果:")
                    print(f"  - 输入result图片数量: {len(result_imgs)}")
                    print(f"  - 对齐后图片数量: {len(retarget_imgs)}")
                    print(f"  - 输出视频: {retarget_video.name}")
                    print(f"跳过本次对齐操作")
                    return
                elif len(retarget_imgs) > 0 and len(retarget_imgs) != len(result_imgs):
                    print(f"检测到不完整的对齐结果:")
                    print(f"  - 输入result图片数量: {len(result_imgs)}")
                    print(f"  - 对齐后图片数量: {len(retarget_imgs)}")
                    print(f"将重新进行对齐操作")
        
        # 创建Retargeter实例，使用黑色背景
        retargeter = Retargeter(use_cache=use_cache, black=True)
        
        # 获取source和result图片列表
        source_imgs = sorted(list(self.source_img_dir.glob("*.jpg")))
        result_imgs = sorted(list(self.result_img_dir.glob("*.jpg")))
        
        if not source_imgs:
            print(f"在 {self.source_img_dir} 中没有找到source图片")
            return
        
        if not result_imgs:
            print(f"在 {self.result_img_dir} 中没有找到result图片")
            return
        
        print(f"找到 {len(source_imgs)} 张source图片，{len(result_imgs)} 张result图片")
        
        # 确保两个目录的图片数量一致
        min_count = min(len(source_imgs), len(result_imgs))
        print(f"将对齐 {min_count} 张图片")
        
        # 对齐对应编号的图片
        for i in tqdm(range(min_count), desc="对齐图片", unit="张"):
            try:
                source_img_path = source_imgs[i]
                result_img_path = result_imgs[i]
                
                # 生成输出文件名（保持相同的编号）
                output_filename = result_img_path.name
                output_path = self.retarget_img_dir / output_filename
                
                # 执行对齐：result图片对齐到对应的source图片
                retargeter.retarget(
                    source_img_path=str(source_img_path),  # source图片作为对齐目标
                    target_img_path=str(result_img_path),  # result图片需要被对齐
                    output_path=str(output_path)
                )
                
            except Exception as e:
                print(f"对齐图片 {result_img_path.name} 失败: {e}")
                continue
        
        print(f"图片对齐完成！输出到: {self.retarget_img_dir}")
        
        # 将对齐后的图片合并为视频
        output_video_path = self.video_output_dir / "retarget.mp4"
        merge_images_to_video(
            img_dir=self.retarget_img_dir,
            output_video_path=output_video_path
        )
    
    def process_video(self):
        """
        处理视频，裁剪为三个512×520的区域，然后resize到512×512
        视频实际结构：1552×528，每个部分512×520，带4像素边框
        """
        if not self.video_path.exists():
            print(f"视频文件不存在: {self.video_path}")
            return
        
        # 获取视频信息
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", str(self.video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            info = json.loads(result.stdout)
            
            # 获取视频尺寸
            video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
            if not video_stream:
                print("无法获取视频信息")
                return
            
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            print(f"视频尺寸: {width}x{height}")
            
            # 检查视频尺寸是否符合预期
            expected_width = 1552  # 4+512+4+512+4+512+4
            expected_height = 528  # 4+520+4
            
            if width != expected_width or height != expected_height:
                print(f"警告：视频尺寸不符合预期 {expected_width}x{expected_height}，实际为 {width}x{height}")
                print("将尝试按照实际尺寸处理...")
            
            # 定义裁剪参数
            crop_width = 512
            crop_height = 520
            border = 4
            
            # 计算三个区域的起始位置
            # 区域1: x = border, y = border
            # 区域2: x = border + crop_width + border, y = border  
            # 区域3: x = border + 2*(crop_width + border), y = border
            
            region1_x = border
            region2_x = border + crop_width + border
            region3_x = border + 2 * (crop_width + border)
            region_y = border
            
            print(f"裁剪参数:")
            print(f"  区域1: x={region1_x}, y={region_y}, w={crop_width}, h={crop_height}")
            print(f"  区域2: x={region2_x}, y={region_y}, w={crop_width}, h={crop_height}")
            print(f"  区域3: x={region3_x}, y={region_y}, w={crop_width}, h={crop_height}")
            
            # 区域1: 静态图片 (只提取第一帧)
            self.extract_first_frame_cropped(
                self.video_path,
                self.video_output_dir / "image.jpg",
                region1_x, region_y, crop_width, crop_height
            )
            
            # 区域2: source视频
            source_video_path = self.video_output_dir / "source.mp4"
            self.crop_and_resize_video(
                self.video_path,
                source_video_path,
                region2_x, region_y, crop_width, crop_height
            )
            
            # 提取source视频的图片
            self.extract_imgs(source_video_path, self.source_img_dir)
            
            # 区域3: result视频
            result_video_path = self.video_output_dir / "result.mp4"
            self.crop_and_resize_video(
                self.video_path,
                result_video_path,
                region3_x, region_y, crop_width, crop_height
            )
            
            # 提取result视频的图片
            self.extract_imgs(result_video_path, self.result_img_dir)
            
            print("视频处理完成！")
            
            # 对齐图片
            self.align_images()
            
        except subprocess.CalledProcessError as e:
            print(f"获取视频信息失败: {e}")
        except Exception as e:
            print(f"处理视频时出错: {e}")
    
    def extract_first_frame_cropped(self, video_path, output_path, x, y, width, height):
        """
        提取视频第一帧并裁剪到指定区域，然后resize到512×512
        
        Args:
            video_path: 输入视频路径
            output_path: 输出图片路径
            x, y: 裁剪起始位置
            width, height: 裁剪尺寸
        """
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vframes", "1",  # 只提取第一帧
            "-vf", f"crop={width}:{height}:{x}:{y},scale=512:512",  # 裁剪并resize
            "-q:v", "2",      # 高质量
            str(output_path),
            "-y"              # 覆盖已存在的文件
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"成功提取并处理第一帧到: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"提取第一帧失败: {e}")
            print(f"错误输出: {e.stderr.decode()}")
    
    def crop_and_resize_video(self, input_path, output_path, x, y, width, height):
        """
        裁剪视频并resize到512×512
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            x, y: 裁剪起始位置
            width, height: 裁剪尺寸
        """
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-vf", f"crop={width}:{height}:{x}:{y},scale=512:512",  # 裁剪并resize
            "-c:v", "libx264",  # 使用H.264编码
            "-preset", "medium", # 编码预设
            "-crf", "23",        # 质量设置
            str(output_path),
            "-y"                 # 覆盖已存在的文件
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"成功裁剪并resize视频到: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"裁剪视频失败: {e}")
            print(f"错误输出: {e.stderr.decode()}")


def process_all_videos(data_dir="data/rebuttal"):
    """
    处理data_dir/output目录下的所有视频文件
    
    Args:
        data_dir: 数据目录路径
    """
    ori_video_dir = Path(data_dir) / "output"
    
    if not ori_video_dir.exists():
        print(f"视频目录不存在: {ori_video_dir}")
        return
    
    # 获取所有mp4文件
    video_files = list(ori_video_dir.glob("*.mp4"))
    
    if not video_files:
        print(f"在 {ori_video_dir} 中没有找到mp4文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    print("=" * 50)
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] 处理视频: {video_file.name}")
        
        # 创建VideoProcessor实例
        processor = VideoProcessor(video_file.name, data_dir)
        print(f"输出目录: {processor.video_output_dir}")
        
        # 处理视频
        # processor.process_video()
        
        # 对齐图片
        processor.align_images()
        
        print(f"✓ 完成: {video_file.name}")
        print("-" * 30)
    
    print(f"\n所有视频处理完成！共处理 {len(video_files)} 个视频")
    print(f"输出目录: {Path(data_dir) / 'retarget'}")


def test_extract_first_frame():
    """
    测试方法：提取第一个视频的第一帧
    """
    from pathlib import Path
    
    # 获取第一个视频文件
    ori_video_dir = Path("data/rebuttal/output")
    video_files = list(ori_video_dir.glob("*.mp4"))
    
    if not video_files:
        print("没有找到视频文件")
        return
    
    # 选择第一个视频文件
    first_video = video_files[0]
    print(f"测试视频: {first_video.name}")
    
    # 创建VideoProcessor实例
    processor = VideoProcessor(first_video.name, "data/rebuttal")
    
    print(f"视频输出目录: {processor.video_output_dir}")
    print(f"Source图片目录: {processor.source_img_dir}")
    print(f"Result图片目录: {processor.result_img_dir}")
    print(f"Retarget图片目录: {processor.retarget_img_dir}")
    print(f"Retarget视频: {processor.video_output_dir / 'retarget.mp4'}")
    
    # 获取视频信息
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(processor.video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        info = json.loads(result.stdout)
        
        # 获取视频尺寸
        video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
        if video_stream:
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            print(f"视频尺寸: {width}x{height}")
            print(f"宽高比: {width/height:.2f}")
            
            # 检查是否符合预期
            expected_width = 1552
            expected_height = 528
            if width == expected_width and height == expected_height:
                print("✓ 视频尺寸符合预期")
            else:
                print(f"⚠ 视频尺寸不符合预期 ({expected_width}x{expected_height})")
            
            # 显示裁剪区域信息
            crop_width = 512
            crop_height = 520
            border = 4
            
            region1_x = border
            region2_x = border + crop_width + border
            region3_x = border + 2 * (crop_width + border)
            region_y = border
            
            print(f"\n裁剪区域信息:")
            print(f"  区域1 (静态图片): x={region1_x}, y={region_y}, w={crop_width}, h={crop_height}")
            print(f"  区域2 (source视频): x={region2_x}, y={region_y}, w={crop_width}, h={crop_height}")
            print(f"  区域3 (result视频): x={region3_x}, y={region_y}, w={crop_width}, h={crop_height}")
            
        else:
            print("无法获取视频信息")
            return
            
    except Exception as e:
        print(f"获取视频信息失败: {e}")
        return
    
    # 提取第一帧（区域1）
    output_path = processor.video_output_dir / "image.jpg"
    processor.extract_first_frame_cropped(
        processor.video_path, 
        output_path,
        region1_x, region_y, crop_width, crop_height
    )
    
    print(f"第一帧已保存到: {output_path}")
    print("请检查生成的图片是否正确裁剪和resize到512×512")


if __name__ == "__main__":
    # 运行测试
    # test_extract_first_frame()
    
    # 如果需要处理所有视频，取消下面的注释
    process_all_videos()
