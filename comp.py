import os
from expdataloader import *

exp_name_list = ["Protrait4Dv2", "LivePortrait", "VOODOO3D", "GAGAvatar", "FollowYourEmoji", "ROME", "ours", "XPortrait"]

# print(os.listdir("data/orz_output"))

for exp_name in exp_name_list:
    loader = RowDataLoader(exp_name)
    loader.print_summary()