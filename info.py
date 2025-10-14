from expdataloader import HeadGenLoader
import os

def xportrait():
    loader = HeadGenLoader("XPortrait")
    loader.print_summary()

def main():
    # xportrait()
    loader = HeadGenLoader("Protrait4Dv2")
    loader.print_summary()

if __name__ == '__main__':
    main()