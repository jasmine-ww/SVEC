This repository implements SVEC for VideoLLaMA, built on top of the original [VideoLLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA?tab=readme-ov-file) codebase.

## Environment Preparation

First, install ffmpeg.

```python
apt update
apt install ffmpeg
```

Then, create a conda environment:

```python
conda env create -f environment.yml
conda activate videollama
```

## Fine-tuning

#### Data Preparation

the fine-tuning dataset consists of:

- 150K image-based instructions from LLaVA [[link](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/raw/main/llava_instruct_150k.json)]
- 11K video-based instructions from VideoChat [[link](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data)]

#### script

Config the checkpoint and dataset paths in visionbranch_stage2_finetune.yaml. Then, run the following script:

```python
conda activate videollama
# for fine-tuning VL branch
torchrun --nproc_per_node=8 train.py --cfg-path  ./train_configs/visionbranch_stage2_finetune.yaml
```

