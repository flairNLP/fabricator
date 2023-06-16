# Dataset generation using LLMs

This repository contains the code for generating datasets using LLMs.

## Setup

```
conda create -y -n aidatasetgenerator python=3.10
conda activate aidatasetgenerator
pip install -r requirements.txt
```

## Main components

### DataPoints

Data points are the basic building blocks of our dataset. 
They are used to represent a single example in the dataset. The data point interface is important for the dataset generator to work since it matches the structure of our predefined prompts.
We currently support the following data points:
- TextDataPoint: A data point that contains a single text.
- ExtractiveQADataPoint: A data point that contains a question, a context, and an answer.

### PromptTemplates

Prompt templates are used to generate prompts for the dataset generator. In order to properly fromat the prompt, it is necessary to match input variables with class attributes of the used DataPoints.
We currently support the following prompt templates:
- TextGenerationPrompt: A prompt template that generates text.
- AnswerAnnotationPrompt: A prompt template that generates answer for given context and question.
- QuestionAnnotationPrompt: A prompt template that generates questions for given context.
- ContextAnnotationPrompt: A prompt template that generates context for given question and answer.

### DatasetGenerator

The dataset generator is the main component of this repository. It uses a given LLM to generate prompts for given data points. The generated prompts are then used to generate a dataset.
It also does quality checks and takes care that instances of the input data points are returned.

## Usage

Check out our examples in the `examples` folder. We support two settings: annotate data points and generate unlabeled data.

```python
import random

from datasets import load_dataset
from langchain.llms import OpenAI

from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.task_templates import ExtractiveQADataPoint
from ai_dataset_generator.prompt_templates import AnswerAnnotationPrompt

num_support = 10
num_unlabeled = 10
total_examples = num_support + num_unlabeled

dataset = load_dataset("squad_v2", split="train") # load any dataset
dataset = dataset.select(random.sample(range(len(dataset)), total_examples))

extractive_qa_samples = [
    ExtractiveQADataPoint(                  # convert to our data point format
        title=sample["title"],
        question=sample["question"],
        context=sample["context"],
        answer=sample["answers"]["text"][0],
        answer_start=sample["answers"]["answer_start"][0] if sample["answers"]["answer_start"] else None,
    ) for sample in dataset]

unlabeled_examples, support_examples = extractive_qa_samples[:num_unlabeled], extractive_qa_samples[num_unlabeled:]

prompt_template = AnswerAnnotationPrompt()  # use a prompt template to generate prompts
llm = OpenAI(model_name="text-davinci-003") # load an LLM with langchain
generator = DatasetGenerator(llm)           # create dataset generator
generated_dataset = generator.generate(     # generate the dataset
    unlabeled_examples=unlabeled_examples,
    support_examples=support_examples,
    prompt_template=prompt_template,
    max_prompt_calls=1,
)
```
