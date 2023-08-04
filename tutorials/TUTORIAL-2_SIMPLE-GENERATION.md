# Tutorial 2: Generate simple datasets

## Generate without fewshot examples

This example shows how to generate a dataset without fewshot examples. It just take a task
description and returns a dataset with movie reviews which can be pushed
to the HuggingFace Hub.

```python
import os
from haystack.nodes import PromptNode
from fabricator import DatasetGenerator
from fabricator.prompts import BasePrompt

prompt = BasePrompt(
    task_description="Generate a very very short movie review.",
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

## Generate with label options

This example shows how to generate a dataset with label options. As introduced in previous tutorial,
this can be achieved by providing a `label_options` argument to the `BasePrompt` constructor.

```python
import os
from haystack.nodes import PromptNode
from fabricator import DatasetGenerator
from fabricator.prompts import BasePrompt

label_options = ["positive", "negative"]

prompt = BasePrompt(
    task_description="Generate a very very short, {} movie review.",
    label_options=label_options,
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

With label_options specified, the generator will uniformly sample one of the label options and insert it into the task
description which ensure that the generated dataset is balanced.


## Generate with fewshot examples
This example shows how to generate a dataset with fewshot examples. As introduced in previous tutorial, this can be 
achieved by providing a `fewshot_dataset` argument to the `DatasetGenerator.generate()` method.

First, we create an annotated `fewshot_dataset` with two columns: `text` and `label`. In order to generate new movie 
reviews and provide the LLM with examples, we need to specify the `generate_data_for_column` argument in the 
`BasePrompt` constructor. This argument tells the generator which column to generate data for.

Since we are using fewshot examples, we can control the prompt generation through different sampling strategies and 
number of examples per class. We pass the `fewshot_dataset` to the generate function and specify to use one fewshot 
example per class per prompt. The `fewshot_label_sampling_strategy` argument specifies how to sample from the fewshot 
dataset. In this case, we use a uniform sampling strategy which means that the generator will uniformly sample one
example per class from the fewshot dataset. The `fewshot_sampling_column` argument specifies which column to use for
sampling. In this case, we use the `label` column.

```python
import os
from datasets import Dataset
from haystack.nodes import PromptNode
from fabricator import DatasetGenerator
from fabricator.prompts import BasePrompt

label_options = ["positive", "negative"]

fewshot_dataset = Dataset.from_dict({
    "text": ["This movie is great!", "This movie is bad!"],
    "label": label_options
})

prompt = BasePrompt(
    task_description="Generate a {} movie review.",
    label_options=label_options,
    generate_data_for_column="text",
)

prompt_node = PromptNode(
    model_name_or_path="gpt-3.5-turbo",
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_length=100,
)

generator = DatasetGenerator(prompt_node)
generated_dataset = generator.generate(
    prompt_template=prompt,
    fewshot_dataset=fewshot_dataset,
    fewshot_examples_per_class=1,
    fewshot_label_sampling_strategy="uniform",
    fewshot_sampling_column="label",
    max_prompt_calls=10,
)

generated_dataset.push_to_hub("your-first-generated-dataset")
```

## Annotate unlabeled data with fewshot examples

This example shows how to annotate unlabeled data with fewshot examples. In this case, we have a fewshot dataset with
two columns: `text` and `label`. We also have an unlabeled dataset with only a `text` column. We want to annotate the
unlabeled dataset with the fewshot dataset. In order to do this, we need to specify the `unlabeled_dataset` argument
to the `DatasetGenerator.generate()` method. We also need to specify the `fewshot_examples_per_class` argument to
specify how many fewshot examples to use per class. In this case, we use one example per class. The 
`fewshot_label_sampling_strategy` argument specifies how to sample from the fewshot dataset. 
In this case, we use a stratfied sampling strategy which means that the generator will sample exactly one example from 
each class from the fewshot dataset. In this case, we do not need to explicitly specify the `fewshot_sampling_column`
argument since the generator will use the column specified in `generate_data_for_column` by default.

```python
import os
from datasets import Dataset
from haystack.nodes import PromptNode
from fabricator import DatasetGenerator
from fabricator.prompts import BasePrompt

label_options = ["positive", "negative"]

fewshot_dataset = Dataset.from_dict({
    "text": ["This movie is great!", "This movie is bad!"],
    "label": label_options
})

unlabeled_dataset = Dataset.from_dict({
    "text": ["This movie was a blast!", "This movie was not bad!"],
})

prompt = BasePrompt(
    task_description="Annotate movie reviews as either: {}.",
    label_options=label_options,
    generate_data_for_column="label",
    fewshot_example_columns="text",
)

prompt_node = PromptNode(
    model_name_or_path="gpt-3.5-turbo",
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_length=100,
)

generator = DatasetGenerator(prompt_node)
generated_dataset = generator.generate(
    prompt_template=prompt,
    fewshot_dataset=fewshot_dataset,
    fewshot_examples_per_class=1,
    fewshot_label_sampling_strategy="stratified",
    unlabeled_dataset=unlabeled_dataset,
    max_prompt_calls=10,
)

generated_dataset.push_to_hub("your-first-generated-dataset")
```
