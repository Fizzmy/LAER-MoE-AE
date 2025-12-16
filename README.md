# LAER-MoE Artifact

This repo contains the actifact of paper "LAER-MoE: Load-Adaptive Expert Re-layout for Efficient Mixture-of-Experts Training", including codes and scripts for reproducing all experiments in the paper.

## Requirements

### Hardware dependencies

We conduct our experiments on a 4-node GPU cluster, with each node containing 8 NVIDIA A100 80GB GPUs. Within nodes, GPUs within a node are connected via NVLink, and nodes are interconnected via Infiniband. The peak unidirectional communication bandwidth intra-node is 450 GB/s, and inter-node is 800 Gbps.

### Software dependencies

Following toolkits are requireds: python=3.9.2, CUDA=12.1, torch=2.1.0-cu121. For baseline, cmake >= 3.21 is also required.

## Installation

> For artifact evaluation, users can use existing virtual environment directly.(conda activate AE)

First, we use conda to create a virtual environment and modify environment variables (if necessary).

```
# Create virtual environment
conda create -n laer-moe python=3.9.2
conda activate laer-moe

# Modify environment variables
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

```

Second, to install the artifact, users need to install torch, flash-attn and apex. Moreover, to be able to run the baseline, users also need to install Transformer-Engine.

```
# Install torch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn
pip install packaging
pip install ninja
pip install psutil
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.5.8
python setup.py install
cd csrc/layer_norm
pip install . --no-build-isolation
# or use wheel


# Install apex
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 312acb4
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# or use wheel



# Install Transformer Engine
pip install pybind11
pip install einops
git clone https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git checkout 7f2afaa
git submodule update --init --recursive
NVTE_FRAMEWORK=pytorch pip install --no-build-isolation .
# or use wheel
```

Finally, install laer-moe:

```
cd LAER-MoE
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

## Prepare datasets

We placed the processed dataset in the `datasets/processed` directory and the raw dataset in the `datasets/raw`. The processed dataset was obtained using the following command:
```
# for wikitext
cd Megatron
mkdir -p ../datasets/processed/wikitext
python tools/preprocess_data.py \
       --input "../datasets/raw/wikitext/wikitext.json" \
       --partitions 1 \
       --output-prefix ../datasets/processed/wikitext/mixtral \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model ../tokenizers/mixtral \
       --workers 20 \
       --append-eod
# for C4
cd Megatron
mkdir -p ../datasets/processed/C4
python tools/preprocess_data.py \
       --input "../datasets/raw/C4/c4-train*" \
       --partitions 5 \
       --output-prefix ../datasets/processed/C4/mixtral \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model ../tokenizers/mixtral \
       --workers 20 \
       --append-eod
```
where `../datasets/raw/wikitext/wikitext.json` and `../datasets/raw/C4/c4-train*` are the paths to the wikitext and C4 raw datasets, respectively. `../datasets/processed/wikitext/mixtral` and `../datasets/processed/C4/mixtral` are the paths to the processed wikitext and C4 datasets, respectively. `../tokenizers/mixtral` is the path to the tokenizer.

## Experiment workflow
Please set working directory to `./LAER-MoE`, and follow `./LAER-MoE/README.md` to conduct experiments.