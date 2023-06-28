# Dataset generation using LLMs

This repository contains the code for generating datasets using LLMs.

## Setup

```
git clone git@github.com:whoisjones/ai-dataset-generator.git
cd ai-dataset-generator
conda create -y -n aidatasetgenerator python=3.10
conda activate aidatasetgenerator
pip install -r requirements.txt
```

## Main components

tbd

### DatasetGenerator

The dataset generator is the main component of this repository. It uses a given LLM to generate prompts for given data points. The generated prompts are then used to generate a dataset.
It also does quality checks and takes care that instances of the input data points are returned.

## Usage

Check out our examples in the `examples` folder. We support two settings: annotate data points and generate unlabeled data.
