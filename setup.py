from setuptools import setup

setup(
    name='exp_data',
    version='0.1',
    packages=['expdataloader'],
    package_dir={'expdataloader': 'expdataloader'},
    install_requires=[
        'natsort',
        'Pillow',
        'opencv-python',
        'face-alignment',
    ]
)
