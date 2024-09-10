# BoViLA: Bootstrapping Video-Language Alignment via LLM-Based Self-Questioning and Answering

This is the official implementation of BoViLA.

<div align="center">
  <img src="asset/main.png" width="900px" />
</div>

## Setup
To install requirements, run:
```
mkdir pretrained
mkdir data
conda create -n bovila python=3.8
conda activate bovila
bash setup.sh
```

## Dataset & LLaMA Preparation

You can download our preprocessed datasets (How2QA, STAR, DramaQA, VLEP and TVQA) at [here](https://drive.google.com/drive/folders/1vFVjMLQWVCKL7KAcLGtuKANv5K-uppeH?usp=sharing). Put them in ```./data```. Also, you can download original LLaMA at [here](https://github.com/facebookresearch/llama/tree/llama_v1), and put the checkpoint in ```./pretrained```.

```
./pretrained
   └─ llama
       |─ 7B
       |   |─ consolidated.00.pth
       |   └─ params.json
       |─ 13B
       |   :
       |─ 33B
       |   :
       └─ tokenizer.model
./data
   |─ how2qa
   |   |─ train.csv
   |   |─ val.csv
   |   └─ clipvitl14.pth
   |─ star
   |   :
   |─ dramaqa
   |   :
   |─ vlep
   |   :
   └─ tvqa
       :
```

## Training BoViLA

### STAR

```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py --model 7B \
--max_seq_len 160 --batch_size 8 --epochs 10 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset star \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/star --accum_iter 2 --vaq 0.5 --aqa 0.25 --aqa_gate
```

### DramaQA

```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py --model 7B \
--max_seq_len 256 --batch_size 8 --epochs 10 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset dramaqa \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/dramaqa --accum_iter 2 --vaq 0.3 --aqa 0.15 --aqa_gate
```

### VLEP

```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py --model 7B \
--max_seq_len 256 --batch_size 8 --epochs 10 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset vlep \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/vlep --sub --accum_iter 2 --vaq 0.5 --aqa 0.25 --aqa_gate
```

### TVQA

```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py --model 7B \
--max_seq_len 160 --batch_size 8 --epochs 5 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset tvqa \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/tvqa --accum_iter 2 --vaq 0.1 --aqa 0.05 --aqa_gate
```

### How2QA

```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 8 train.py --model 7B \
--max_seq_len 160 --batch_size 8 --epochs 3 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset how2qa \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/how2qa --val_split 'public_val' --accum_iter 2 --vaq 0.6 --aqa 0.3 --aqa_gate
```

We provide checkpoints [here](https://huggingface.co/AInsabsw/BoViLA).
## Evaluation
From the training command, simply add ```--resume ./your/checkpoint.pth``` and ```--init_eval```.
