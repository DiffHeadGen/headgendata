from expdataloader import HeadGenLoader
import os

def xportrait():
    loader = HeadGenLoader("XPortrait")
    loader.print_summary()

def main():
    xportrait()

if __name__ == '__main__':
    main()