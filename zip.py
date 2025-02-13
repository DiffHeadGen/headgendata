from expdataloader import HeadGenLoader
import os


loader = HeadGenLoader("LivePortrait")
dir = loader.output_dir
loader.print_summary()
