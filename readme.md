# NLP-BIM Framework for Semantic Text-to-Model Alignment.

![](https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black)
![](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)
[![Tech Used](https://skillicons.dev/icons?i=py,docker,windows,linux,collab)](https://skillicons.dev)

## Updates

- As of now we have provided weights and a validation mechanisms.
- We are planning to publish more resource in time for replicability.

## Setup

```sh
git clone https://github.com/ITalab/client-to-bim
```

Install requirements

```sh
pip install -r requirements.txt
```

Conda

```sh
conda env create -f environment.yml
conda activate client-to-bim
```

Poetry

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

```sh
poetry install
poetry shell
```

## Download Model

```python
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(repo_id="ZappyKirby/client-to-bim", filename="model.safetensors")

```

or manually download weights

https://huggingface.co/ZappyKirby/client-to-bim

## Training

### Pretrain

```sh
python mlm-pretrain.py
```
