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
        'moviepy',
        'numpy==1.26.0'
        'face-alignment',
    ],
    extras_require={
        'face': [
            'insightface',
            'onnxruntime-gpu',
        ]
    }
)