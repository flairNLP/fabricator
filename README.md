<h1 align="center">Dataset Generator</h1>
<p align="center">A flexible open-source framework to generate datasets with large language models.</p>
<p align="center">
<img alt="version" src="https://img.shields.io/badge/version-0.1-green">
<img alt="python" src="https://img.shields.io/badge/python-3.10-blue">
<img alt="Static Badge" src="https://img.shields.io/badge/license-apache2.0-green">
</p>
<div align="center">
<hr>

[Installation](#installation) - [Basic Concepts](#basic-concepts) - [Tutorials](tutorials/TUTORIAL-1_OVERVIEW.md) - 
Paper - [Citation](#citation)

<hr>
</div>

## Overview

This repository:

- is <b>a easy-to-use open-source framework</b> to generate datasets with large language models. If you want to train
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
git clone git@github.com:whoisjones/ai-dataset-generator.git
cd ai-dataset-generator
conda create -y -n aidatasetgenerator python=3.10
conda activate aidatasetgenerator
pip install -r requirements.txt
```

## Basic Concepts

This framework is based on the idea of using large language models to generate datasets for specific tasks. To do so, 
we need four basic modules: a dataset, a prompt, a language model and a generator:
- <b>Dataset</b>: We use [huggingface's datasets library](https://github.com/huggingface/datasets) to load fewshot or 
unlabeled datasets and store the generated or annotated datasets with their `Dataset` class. Once
created, you can share the dataset with others via the hub or use it for your model training.
- <b>Prompt</b>: A prompt is the instruction made to the language model. It can be a simple sentence or a more complex
template with placeholders. We utilize [langchain](https://github.com/langchain-ai/langchain) `PromptTemplate` classes
and provide an easy interface for custom dataset generation prompts in which you can specify label options
for the LLM to choose from, provide fewshot examples to support the prompt with or annotate an unlabeled dataset
in a specific way.
- <b>LLM</b>: We use [deepset's haystack library](https://github.com/deepset-ai/haystack) as our LLM interface. deepset
supports a wide range of LLMs including OpenAI, all models from the HuggingFace model hub and many more.
- <b>Generator</b>: The generator is the core of this framework. It takes a dataset, a prompt and a LLM and generates a
dataset based on your specifications.

## Citation

If you find this repository useful, please cite our work.

