
from expdataloader import *


def main():
    # loader = P4DLoader()
    # loader.clear_all_output()
    loader = P4DRetargetLoader()
    loader.run_all()
def clear():
    loader = P4DLoader()
    loader.clear_all_output()
    
if __name__ == '__main__':
    # main()
    loader = P4DRetargetLoader()
    loader.print_summary()
    