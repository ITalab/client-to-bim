![](https://i.ibb.co/6RfxQdKz/textanim-vx-S9d.gif)
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif">

> **🚧 Project Type**: NLP × BIM | **📅 Timeline**: Spring 2025 | **🏗️ Domain**: Architecture, Engineering, and Construction

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=ITalab.client-to-bim)
![GitHub stars](https://img.shields.io/github/stars/ITalab/client-to-bim?style=social)
![GitHub forks](https://img.shields.io/github/forks/ITalab/client-to-bim?style=social)
![GitHub issues](https://img.shields.io/github/issues/ITalab/client-to-bim)
![GitHub pull requests](https://img.shields.io/github/issues-pr/ITalab/client-to-bim)
![Last Commit](https://img.shields.io/github/last-commit/ITalab/client-to-bim)
![License](https://img.shields.io/github/license/ITalab/client-to-bim)

<!--
SEO:
client to bim, text to bim, natural language to bim, nlp bim integration, bim automation, bim requirement matching,
semantic bim mapping, ifc language understanding, ai for construction design, ai-assisted bim modeling, transformer-based bim,
language model for bim, huggingface bim application, ifc entity classification, automated design generation,
requirement-driven bim modeling, ml in architecture, construction language parsing, project briefing to model generation,
generative bim from text, aec nlp, nlp for architects, smart bim assistants, bim predesign automation,
ai-driven aec workflows, bim semantic matching, bim client requirements, architecture requirement translation,
nlp-based bim alignment, text2bim deep learning, intelligent design assistants, client intent to ifc mapping,
language-guided bim synthesis, ai in aec industry, bim project scoping from text, ifc automation with transformers,
multi-language support for bim, multilingual client requirements to bim, nlp-enhanced bim tooling, bim chatbot interface,
language understanding in construction, prompt to bim, gpt bim integration, ifc data extraction from text,
llm bim converter, ai for architects and engineers, bim integration pipeline, aec digital transformation
-->

## 🧰 Tech Stack

<div align="center">

<img src="https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white" />

<br><br>
<img src="https://user-images.githubusercontent.com/74038190/212257472-08e52665-c503-4bd9-aa20-f5a4dae769b5.gif" width="50">
[![Tech Used](https://skillicons.dev/icons?i=docker,windows,linux,collab)](https://skillicons.dev)

</div>

## <img src="https://user-images.githubusercontent.com/74038190/216122041-518ac897-8d92-4c6b-9b3f-ca01dcaf38ee.png" width=30> Hugging Face Model Card

[![](https://img.shields.io/badge/%F0%9F%A4%97%20client%20to%20bim%20large-Model-blue)](https://huggingface.co/ZappyKirby/client-to-bim/tree/main)

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif">

## ⬇️ Updates

- July 2025 ✨ Pretraining, Training and Validation

## 🫥 Introduction

BIM is central to AEC workflows, offering structured digital models of built assets. However, aligning unstructured text data such as specs, codes, and requirements with BIM elements remains a key challenge. NLP methods, especially transformer-based LMs, enable semantic mapping between text and BIM components. This work proposes a framework that integrates transformers with BIM to automate compliance, retrieval, and querying
tasks across AEC domains.

### 🔊 Audio Summary

<audio controls>
  <source src="assets/summary.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

## 🔗 BIM Architecture

```mermaid
graph LR
ClientInput["🗣️ Client Requirements"] --> NLPModule
NLPModule --> EntityExtraction["📦 Entity Recognition"]
EntityExtraction --> BIMMapper["🔧 IFC/BIM Mapper"]
BIMMapper --> ModelEngine["🏗️ Model Generator"]
ModelEngine --> Output["📐 BIM Model"]
```

## 🧪 Experiments

- Evaluation using a test suite of 30,000 annotated requirements
- Metrics: Precision, Recall, BIM-conformity accuracy

## Results

<table>
<tr>
  <td>

### ML-Architecture Results

| Model                      | F-1 Score |
| -------------------------- | --------- |
| BERT Baseline              | 87%       |
| BERT Pretrained + Baseline | 95%       |
| BERT Pretrained + Our      | 96%       |

  </td>
  <td style="padding-left:40px">

### Pretraining Results

| Model           | F-1 Score |
| --------------- | --------- |
| BERT English    | 91%       |
| BERT Korean     | 89%       |
| BERT Multingual | 90%       |

  </td>
</tr>
</table>
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif">

## 📈 Applications

- Office design automation
- Early-stage design analysis
- Requirements compliance validation

## ⚙️ Setup

### Pre-requisites

Installing CUDA toolkit using the instructions for your platform.
We recommend using `conda` to install `cudatoolkit`.

[![Conda Version](https://img.shields.io/conda/v/conda-forge/cudatoolkit)](https://anaconda.org/conda-forge/cudatoolkit)

Install the corresponding pytorch version
from the link below. We recommend installing CUDA12.6 with the latest pytorch.

[![](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

### Running the source

```sh
git clone https://github.com/ITalab/client-to-bim
```

Install requirements

```sh
pip install -r requirements.txt
```

Conda

![](https://anaconda.org/conda-forge/mlconjug/badges/version.svg)

```sh
conda env create -f environment.yml
conda activate client-to-bim
```

Poetry

![](https://img.shields.io/badge/packaging-poetry-cyan.svg)

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

```sh
poetry install
poetry shell
```

## 📊 Dataset Overview

Each entry in this dataset is organized by **class name** followed by a list of **text samples**. The structure is designed for easy parsing and use in NLP classification tasks.

### 🧾 Description

- **Keys**: Represent class labels (e.g., `OCR`, `OUU`, `OBC`).
- **Values**: Lists of string texts associated with that class.

## Download Model

```python
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(repo_id="ZappyKirby/client-to-bim", filename="model.safetensors")

```

or manually download weights from hugging face

[![](https://img.shields.io/badge/%F0%9F%A4%97%20client%20to%20bim%20large-Model-blue)](https://huggingface.co/ZappyKirby/client-to-bim/tree/main)

## Training

### Pretrain

```sh
python mlm-pretrain.py
```

## 🤝 Contributing

We welcome contributors! See the [CONTRIBUTING.md]() for guidelines.

## 🗨️ Citation

```bib
@misc{ITalab_client-to-bim,
  author       = {{ITalab}},
  title        = {client-to-bim},
  howpublished = {\url{https://github.com/ITalab/client-to-bim}},
  year         = {2025},
  note         = {GitHub repository, accessed July 23, 2025},
}
```

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif">
Made with ❤️ by ITalab
