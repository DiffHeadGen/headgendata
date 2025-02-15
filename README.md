# DiffHeadGen

这个仓库主要用于做对比试验，方便批量处理

## Install

按照以下步骤安装 DiffHeadGen

```bash
mkdir diffheadgen
cd diffheadgen
git clone https://github.com/DiffHeadGen/headgendata.git expdata
cd expdata
source create_env.sh

# other repo
git clone https://github.com/DiffHeadGen/Portrait-4D.git

```

## Usage
在其他仓库中使用可编辑方式安装

```bash
pip install -e .
```


## run demo
    
```bash
ml cuDNN/8.7.0.84-CUDA-11.8.0

empty_gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F ', ' '{if ($2 < 512) print $1}' | head -n 1)
if [ -z "$empty_gpu" ]; then
    echo "No empty GPU available"
    exit 1
fi
CUDA_VISIBLE_DEVICES=$empty_gpu python infer.py
```