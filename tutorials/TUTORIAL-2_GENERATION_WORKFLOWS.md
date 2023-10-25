# Tutorial 2: Generation Workflows

In this tutorial, you will learn:
1. how to generate datasets
2. how to annotate unlabeled datasets
3. how to configure hyperparameters for your generation process

## 1) Generating Datasets

### 1.1) Generating Plain Text

In this example, we demonstrate how to merge components created in previous tutorials by fabricators to create a 
movie review dataset. We don't explicitly direct the Language Learning Model (LLM) to generate movie reviews for 
specific labels (such as binary sentiment) or offer a few examples to guide the LLM in generating similar content. 
Instead, all it requires is a task description. The LLM then produces a dataset containing movie reviews based on the
provided instructions. This dataset can be easily uploaded to the Hugging Face Hub.

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

### 1.2) Generate Label-Conditioned Datasets With Label Options and Few-Shot Examples
To create datasets that are conditioned on specific labels and use few-shot examples, 
we need a few-shot dataset that is already annotated. The prompt should have the same labels as those in the few-shot 
dataset. Additionally, as explained in a previous tutorial, we must set the `generate_data_for_column` 
parameter to specify the column in the dataset for which we want to generate text.

In the dataset generator, we define certain hyperparameters for the generation process. `fewshot_examples_per_class` 
determines how many few-shot examples are incorporated for each class per prompt. `fewshot_sampling_strategy`
can be set to either "uniform" if each label has an equal chance of being sampled, 
or "stratified" if the distribution from the few-shot dataset needs to be preserved. 
`fewshot_sampling_column` specifies the dataset column representing the classes. `max_prompt_calls`
sets the limit for how many prompts should be generated.

Crucially, the prompt instance contains all the details about how a single prompt for a specific data point should 
be structured. This includes information like which few-shot examples should appear alongside which task instruction. 
On the other hand, the dataset generator defines the overall generation process, 
such as determining the label distribution, specified by the `fewshot_sampling_column`.

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
    fewshot_sampling_strategy="uniform",
    fewshot_sampling_column="label",
    max_prompt_calls=10,
)

generated_dataset.push_to_hub("your-first-generated-dataset")
```

## 2) Annotate unlabeled data with fewshot examples

This example demonstrates how to add annotations to unlabeled data using few-shot examples. We have a few-shot dataset containing two columns: `text` and `label`, and an unlabeled dataset with only a `text` column. The goal is to annotate the unlabeled dataset using information from the few-shot dataset.

To achieve this, we utilize the `DatasetGenerator.generate()` method. To begin, we provide the `unlabeled_dataset` argument, indicating the dataset we want to annotate. Additionally, we specify the `fewshot_examples_per_class` argument, determining how many few-shot examples to use for each class. In this scenario, we choose one example per class.

The `fewshot_sampling_strategy` argument dictates how the few-shot dataset is sampled. In this case, we employ a stratified sampling strategy. This means that the generator will select precisely one example from each class within the few-shot dataset.

It's worth noting that there's no need to explicitly specify the `fewshot_sampling_column` argument. By default, the generator uses the column specified in `generate_data_for_column` for this purpose.

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
    fewshot_sampling_strategy="stratified",
    unlabeled_dataset=unlabeled_dataset,
    max_prompt_calls=10,
)

generated_dataset.push_to_hub("your-first-generated-dataset")
```
