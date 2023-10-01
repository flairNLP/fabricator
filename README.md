![Fabricator Logo](resources/logo_fabricator.drawio_dark.png#gh-dark-mode-only)
![Fabricator Logo](resources/logo_fabricator.drawio_white.png#gh-light-mode-only)

<p align="center">A flexible open-source framework to generate datasets with large language models.</p>
<p align="center">
<img alt="version" src="https://img.shields.io/badge/version-0.1-green">
<img alt="python" src="https://img.shields.io/badge/python-3.10-blue">
<img alt="Static Badge" src="https://img.shields.io/badge/license-apache2.0-green">
</p>
<div align="center">
<hr>

[Installation](#installation) | [Basic Concepts](#basic-concepts) | [Examples](#examples) | [Tutorials](tutorials/TUTORIAL-1_OVERVIEW.md) | 
[Paper](https://arxiv.org/abs/2309.09582) | [Citation](#citation)

<hr>
</div>

## Overview

This repository:

- is <b>an easy-to-use open-source library</b> to generate datasets with large language models. If you want to train
a model on a specific domain / label distribution / downstream task, you can use this framework to generate
a dataset for it.
- <b>builds on top of deepset's haystack and huggingface's datasets</b> libraries. Thus, we support a wide range 
of language models and you can load and use the generated datasets as you know it from the Datasets library for your 
model training.
- is <b>highly flexible</b> and offers various adaptions possibilities such as
prompt customization, integration and sampling of fewshot examples or annotation of the unlabeled datasets.

## Installation
Using conda:
```
git clone git@github.com:flairNLP/fabricator.git
cd fabricator
conda create -y -n fabricator python=3.10
conda activate fabricator
pip install -e .
```

## Basic Concepts

This framework is based on the idea of using large language models to generate datasets for specific tasks. To do so, 
we need four basic modules: a dataset, a prompt, a language model and a generator:
- <b>Dataset</b>: We use [huggingface's datasets library](https://github.com/huggingface/datasets) to load fewshot or 
unlabeled datasets and store the generated or annotated datasets with their `Dataset` class. Once
created, you can share the dataset with others via the hub or use it for your model training.
- <b>Prompt</b>: A prompt is the instruction made to the language model. It can be a simple sentence or a more complex
template with placeholders. We provide an easy interface for custom dataset generation prompts in which you can specify 
label options for the LLM to choose from, provide fewshot examples to support the prompt with or annotate an unlabeled 
dataset in a specific way.
- <b>LLM</b>: We use [deepset's haystack library](https://github.com/deepset-ai/haystack) as our LLM interface. deepset
supports a wide range of LLMs including OpenAI, all models from the HuggingFace model hub and many more.
- <b>Generator</b>: The generator is the core of this framework. It takes a dataset, a prompt and a LLM and generates a
dataset based on your specifications.

## Examples

With our library, you can generate datasets for any task you want. You can start as simple
as that:

### Generate a dataset from scratch

```python
import os
from haystack.nodes import PromptNode
from fabricator import DatasetGenerator
from fabricator.prompts import BasePrompt

prompt = BasePrompt(
    task_description="Generate a short movie review.",
)

prompt_node = PromptNode(
    model_name_or_path="gpt-3.5-turbo",
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_length=100,
)

generator = DatasetGenerator(prompt_node)
generated_dataset = generator.generate(
    prompt_template=prompt,
    max_prompt_calls=10,
)

generated_dataset.push_to_hub("your-first-generated-dataset")
```

In our tutorial, we introduce how to create classification datasets with label options to choose from, how to include 
fewshot examples or how to annotate unlabeled data into predefined categories.

## Citation

If you find this repository useful, please cite our work.

```
@misc{golde2023fabricator,
      title={Fabricator: An Open Source Toolkit for Generating Labeled Training Data with Teacher LLMs}, 
      author={Jonas Golde and Patrick Haller and Felix Hamborg and Julian Risch and Alan Akbik},
      year={2023},
      eprint={2309.09582},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


