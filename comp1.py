from dataclasses import dataclass
from functools import cached_property
import os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from expdataloader import *
from expdataloader.utils import LazyVideoWriter, img_grid, get_sub_dir
from expdataloader.dataset import DataSetNames


class CompLoaderImageV3(RowDataLoader):
    def __init__(
        self,
        name="comp_image_v3",
        p1_file="data/comp_cross_v3/p1.txt",
        p2_file="data/comp_cross_v3/p2.txt",
        output_dir="comp_cross_v3",
    ):
        super().__init__(name)
        self.p1_file = p1_file
        self.p2_file = p2_file
        self.output_dir = output_dir

    @cached_property
    def exp_name_list(self):
        return ["ROME", "Protrait4Dv2", "VOODOO3D", "GAGAvatar", "FollowYourEmoji", "XPortrait", "ours"]

    @cached_property
    def all_loaders(self):
        # Exclude 'ours' from the loaders since it's handled specially
        return [RowDataLoader(exp_name) for exp_name in self.exp_name_list[:-1]]

    def parse_txt_file(self, txt_file: str) -> Dict[str, List[str]]:
        """Parse txt file to extract dataset and image paths"""
        data = {}
        current_dataset = None
        
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.endswith(':'):
                    # This is a dataset name
                    current_dataset = line[:-1]  # Remove the colon
                    data[current_dataset] = []
                elif current_dataset and line.startswith('/'):
                    # This is an image path
                    data[current_dataset].append(line)
        
        return data

    def get_id_and_frame_from_path(self, path: str) -> Tuple[str, str]:
        """Extract data_id and frame number from ours method path"""
        # Path format: /nas_data/home/gaoxuan/diffavatar/combined_run/322_EXP-5-mouth_cam_222200041/frames/000140_result.jpg
        path_parts = Path(path).parts
        data_id = path_parts[-3]  # e.g., "322_EXP-5-mouth_cam_222200041"
        frame_name = Path(path).stem  # e.g., "000140_result"
        # Remove '_result' suffix to get clean frame number
        frame_number = frame_name.replace('_result', '')  # e.g., "000140"
        return data_id, frame_number

    def comp_cross_img(self, txt_file: str, output_name: str):
        """Generate comparison grid similar to comp.py but with ours paths from txt file"""
        # Parse the txt file
        parsed_data = self.parse_txt_file(txt_file)
        
        # Get output directory
        output_dir = get_sub_dir("data", self.output_dir)
        
        # Collect all images
        img_path_list = []
        
        # Process each dataset
        for dataset_name, ours_paths in parsed_data.items():
            print(f"Processing dataset: {dataset_name}")
            
            # Process each ours path
            for ours_path in ours_paths:
                data_id, frame_number = self.get_id_and_frame_from_path(ours_path)
                
                try:
                    # Add other methods (excluding 'ours')
                    for method_name in self.exp_name_list[:-1]:  # Exclude 'ours'
                        try:
                            # Use the appropriate dataset based on dataset_name
                            if dataset_name == "combined":
                                loader = RowDataLoader(method_name, dataset_name=DataSetNames.ORZ)
                            elif dataset_name == "orz_isolated":
                                loader = RowDataLoader(method_name, dataset_name=DataSetNames.ORZ_ISOLATED)
                            else:
                                loader = RowDataLoader(method_name)
                            
                            row = loader.get_row(data_id)
                            img_path = os.path.join(row.frames_dir, f"{frame_number}.jpg")
                            if os.path.exists(img_path):
                                img_path_list.append(img_path)
                            else:
                                print(f"Missing image: {img_path}")
                                img_path_list.append("")  # Placeholder for missing image
                        except Exception as e:
                            print(f"Error loading {method_name} for {data_id}: {e}")
                            img_path_list.append("")  # Placeholder for missing image
                    
                    # Add ours method (use the direct path from txt file)
                    img_path_list.append(ours_path)
                    
                except Exception as e:
                    print(f"Error processing {data_id}: {e}")
                    continue
        
        # Generate comparison grid
        if img_path_list:
            # Filter out empty paths and create placeholder for missing images
            filtered_paths = []
            placeholder_path = os.path.join(output_dir, "placeholder.jpg")
            
            # Create placeholder image if it doesn't exist
            if not os.path.exists(placeholder_path):
                from PIL import Image
                placeholder_img = Image.new("RGB", (512, 512), (255, 255, 255))
                placeholder_img.save(placeholder_path)
            
            for path in img_path_list:
                if path and os.path.exists(path):
                    filtered_paths.append(path)
                else:
                    filtered_paths.append(placeholder_path)
            
            # Reshape to (-1, 7) like in comp.py
            img_array = np.array(filtered_paths).reshape(-1, 7)
            
            # Generate the comparison grid
            comparison_path = os.path.join(output_dir, f"{output_name}_comparison.jpg")
            img_grid(img_array, save_path=comparison_path)
            print(f"Generated comparison: {comparison_path}")

    def comp_cross_img_rs(self, txt_file: str, output_name: str):
        """Generate source-target comparison using orz_isolated dataset"""
        # Parse the txt file
        parsed_data = self.parse_txt_file(txt_file)
        
        # Get output directory
        output_dir = get_sub_dir("data", self.output_dir)
        
        # Collect source and target images
        img_path_list = []
        
        # Process each dataset
        for dataset_name, ours_paths in parsed_data.items():
            print(f"Processing dataset: {dataset_name}")
            
            # Process each ours path
            for ours_path in ours_paths:
                data_id, frame_number = self.get_id_and_frame_from_path(ours_path)
                
                try:
                    # Get source and target images using appropriate dataset
                    if dataset_name == "combined":
                        row = RowDataLoader("ours", dataset_name=DataSetNames.ORZ).get_row(data_id)
                        # For orz dataset, remove 'cross' from source path
                        source_img_path = row.source_img_path.replace('_cross', '')
                    else:
                        row = RowDataLoader("ours", dataset_name=DataSetNames.ORZ_ISOLATED).get_row(data_id)
                        source_img_path = row.source_img_path
                    
                    target_img_path = os.path.join(row.target.imgs_dir, f"{frame_number}.jpg")
                    
                    # Check if files exist and print missing ones
                    if not os.path.exists(source_img_path):
                        print(f"Missing source image: {source_img_path}")
                    if not os.path.exists(target_img_path):
                        print(f"Missing target image: {target_img_path}")
                    
                    img_path_list.extend([source_img_path, target_img_path])
                    
                except Exception as e:
                    print(f"Error processing {data_id}: {e}")
                    continue
        
        # Generate source-target comparison
        if img_path_list:
            # Filter out empty paths and create placeholder for missing images
            filtered_paths = []
            placeholder_path = os.path.join(output_dir, "placeholder.jpg")
            
            # Create placeholder image if it doesn't exist
            if not os.path.exists(placeholder_path):
                from PIL import Image
                placeholder_img = Image.new("RGB", (512, 512), (255, 255, 255))
                placeholder_img.save(placeholder_path)
            
            for path in img_path_list:
                if path and os.path.exists(path):
                    filtered_paths.append(path)
                else:
                    filtered_paths.append(placeholder_path)
            
            # Reshape to (-1, 2) for source and target
            img_array = np.array(filtered_paths).reshape(-1, 2)
            
            # Generate the source-target comparison
            source_target_path = os.path.join(output_dir, f"{output_name}_source_target.jpg")
            img_grid(img_array, save_path=source_target_path)
            print(f"Generated source-target: {source_target_path}")

    def run_comparison(self):
        """Run comparison for both p1.txt and p2.txt"""
        print("Generating comparison for p1.txt...")
        self.comp_cross_img(self.p1_file, "p1")
        self.comp_cross_img_rs(self.p1_file, "p1")
        
        print("Generating comparison for p2.txt...")
        self.comp_cross_img(self.p2_file, "p2")
        self.comp_cross_img_rs(self.p2_file, "p2")


def test_parsing():
    """Test the txt file parsing functionality"""
    loader = CompLoaderImageV3()
    
    print("Testing p1.txt parsing:")
    p1_data = loader.parse_txt_file(loader.p1_file)
    for dataset, paths in p1_data.items():
        print(f"  {dataset}: {len(paths)} paths")
        for path in paths[:2]:  # Show first 2 paths
            data_id, frame_name = loader.get_id_and_frame_from_path(path)
            print(f"    {data_id} -> {frame_name}")
    
    print("\nTesting p2.txt parsing:")
    p2_data = loader.parse_txt_file(loader.p2_file)
    for dataset, paths in p2_data.items():
        print(f"  {dataset}: {len(paths)} paths")
        for path in paths[:2]:  # Show first 2 paths
            data_id, frame_name = loader.get_id_and_frame_from_path(path)
            print(f"    {data_id} -> {frame_name}")


def main():
    try:
        loader = CompLoaderImageV3()
        loader.run_comparison()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # test_parsing()  # Uncomment to test parsing
    main()
