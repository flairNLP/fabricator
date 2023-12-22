# Tutorial 3: Advanced Dataset Generation

## Customizing Prompts

Sometimes, you want to customize your prompt to your specific needs. For example, you might want to add a custom 
formatting template (the default takes the column names of the dataset):

```python
from datasets import Dataset
from fabricator.prompts import BasePrompt

label_options = ["positive", "negative"]

fewshot_examples = Dataset.from_dict({
    "text": ["This movie is great!", "This movie is bad!"],
    "label": label_options
})

prompt = BasePrompt(
    task_description="Annotate the sentiment of the following movie review whether it is: {}.",
    generate_data_for_column="label",
    fewshot_example_columns="text",
    fewshot_formatting_template="Movie Review: {text}\nSentiment: {label}",
    target_formatting_template="Movie Review: {text}\nSentiment: ",
    label_options=label_options,
)

print(prompt.get_prompt_text(label_options, fewshot_examples))
```

which yields:

```text
Annotate the sentiment of the following movie review whether it is: positive, negative.

Movie Review: This movie is great!
Sentiment: positive

Movie Review: This movie is bad!
Sentiment: negative

Movie Review: {text}
Sentiment: 
```

## Inferring the Prompt from Dataset Info

Huggingface Dataset objects provide the possibility to infer a prompt from the dataset. This can be achieved by using
the `infer_prompt_from_dataset` function. This function takes a dataset
as input and returns a `BasePrompt` object. The `BasePrompt` object contains the task description, the label options
and the respective columns which can be used to generate a dataset with the `DatasetGenerator` class.

```python
from datasets import load_dataset
from fabricator.prompts import infer_prompt_from_dataset

dataset = load_dataset("imdb", split="train")
prompt = infer_prompt_from_dataset(dataset)

print(prompt.get_prompt_text() + "\n---")

label_options = dataset.features["label"].names
fewshot_example = dataset.shuffle(seed=42).select([0])

print(prompt.get_prompt_text(label_options, fewshot_example))
```

The output of this script is:

```text
Classify the following texts exactly into one of the following categories: {}.

text: {text}
label: 
---
Classify the following texts exactly into one of the following categories: neg, pos.

text: There is no relation at all between Fortier and Profiler but the fact that both are police series [...]
label: 1

text: {text}
label: 
```

This feature is particularly useful, if you have nested structures that follow a common format such as for
extractive question answering:

```python
from datasets import load_dataset
from fabricator.prompts import infer_prompt_from_dataset

dataset = load_dataset("squad_v2", split="train")
prompt = infer_prompt_from_dataset(dataset)

print(prompt.get_prompt_text() + "\n---")

label_options = dataset.features["label"].names
fewshot_example = dataset.shuffle(seed=42).select([0])

print(prompt.get_prompt_text(label_options, fewshot_example))
```

This script outputs:

```text
Given a context and a question, generate an answer that occurs exactly and only once in the text.

context: {context}
question: {question}
answers: 
---
Given a context and a question, generate an answer that occurs exactly and only once in the text.

context: The Roman Catholic Church canon law also includes the main five rites (groups) of churches which are in full union with the Roman Catholic Church and the Supreme Pontiff:
question: What term characterizes the intersection of the rites with the Roman Catholic Church?
answers: {'text': ['full union'], 'answer_start': [104]}

context: {context}
question: {question}
answers: 
```

## Preprocess datasets

In the previous example, we highlighted the simplicity of generating prompts using Hugging Face Datasets information. 
However, for optimal utilization of LLMs in generating text, it's essential to incorporate label names instead of IDs
for text classification. Similarly, for question answering tasks, plain substrings are preferred over JSON-formatted 
strings. We'll elaborate on these limitations in the following example.

```text
Classify the following texts exactly into one of the following categories: **neg, pos**.

text: There is no relation at all between Fortier and Profiler but the fact that both are police series [...]
**label: 1**

---

Given a context and a question, generate an answer that occurs exactly and only once in the text.

context: The Roman Catholic Church canon law also includes the main five rites (groups) of churches which are in full union with the Roman Catholic Church and the Supreme Pontiff:
question: What term characterizes the intersection of the rites with the Roman Catholic Church?
answers: **{'text': ['full union'], 'answer_start': [104]}**
```

To overcome this, we provide a range of preprocessing functions for various downstream tasks.

### Text Classification

The `convert_label_ids_to_texts` function transforms your text classification dataset with label IDs into textual
labels. The default will be the label names specified in the features column.

```python
from datasets import load_dataset
from fabricator.prompts import infer_prompt_from_dataset
from fabricator.dataset_transformations.text_classification import convert_label_ids_to_texts

dataset = load_dataset("imdb", split="train")
prompt = infer_prompt_from_dataset(dataset)
dataset, label_options = convert_label_ids_to_texts(
    dataset=dataset,
    label_column="label",
    return_label_options=True
)

fewshot_example = dataset.shuffle(seed=42).select([0])
print(prompt.get_prompt_text(label_options, fewshot_example))
```

Which yields:

```text
Classify the following texts exactly into one of the following categories: neg, pos.

text: There is no relation at all between Fortier and Profiler but the fact that both are police series [...]
label: pos

text: {text}
label: 
```

If we want to provide more meaningful names we can do so by specifying `expanded_label_mapping`.
Remember to update the label options accordingly in the `BasePrompt` class.

```python
extended_mapping = {0: "negative", 1: "positive"}
dataset, label_options = convert_label_ids_to_texts(
    dataset=dataset,
    label_column="label",
    expanded_label_mapping=extended_mapping,
    return_label_options=True
)
prompt.label_options = label_options
```

This yields:

```text
Classify the following texts exactly into one of the following categories: positive, negative.

text: There is no relation at all between Fortier and Profiler but the fact that both are police series [...]
label: positive

text: {text}
label: 
```

Once the dataset is generated, one can easily convert the string labels back to label IDs by
using huggingface's `class_encode_labels` function.

```python
dataset = dataset.class_encode_column("label")
print("Labels: " + str(dataset["label"][:5]))
print("Features: " + str(dataset.features["label"]))
```

Which yields:

```text
Labels: [0, 1, 1, 0, 0]
Features: ClassLabel(names=['negative', 'positive'], id=None)
```

<ins>Note:</ins> While generating the dataset, the model is supposed to assign labels based on the specific options 
provided. However, we do not filter the data if it doesn't adhere to these predefined labels. 
Therefore, it's important to double-check if the annotations match the expected label options.
If they don't, you should make corrections accordingly.

### Question Answering (Extractive)

In question answering tasks, we offer two functions to handle dataset processing: preprocessing and postprocessing. 
The preprocessing function is responsible for transforming datasets from SQuAD format into flat strings. 
On the other hand, the postprocessing function reverses this process by converting flat predictions back into 
SQuAD format. It achieves this by determining the starting point of the answer and checking if the answer cannot be 
found in the given context or if it occurs multiple times.

```python
from datasets import load_dataset
from fabricator.prompts import infer_prompt_from_dataset
from fabricator.dataset_transformations.question_answering import preprocess_squad_format, postprocess_squad_format

dataset = load_dataset("squad_v2", split="train")
prompt = infer_prompt_from_dataset(dataset)

dataset = preprocess_squad_format(dataset)

print(prompt.get_prompt_text(None, dataset.select([0])) + "\n---")

dataset = postprocess_squad_format(dataset)
print(dataset[0]["answers"])
```

Which yields:

```text
Given a context and a question, generate an answer that occurs exactly and only once in the text.

context: Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".
question: When did Beyonce start becoming popular?
answers: in the late 1990s

context: {context}
question: {question}
answers: 
---
{'start': 269, 'text': 'in the late 1990s'}
```

### Named Entity Recognition

If you attempt to create a dataset for named entity recognition without any preprocessing, the prompt might be 
challenging for the language model to understand.

```python
from datasets import load_dataset
from fabricator.prompts import BasePrompt

dataset = load_dataset("conll2003", split="train")
prompt = BasePrompt(
    task_description="Annotate each token with its named entity label: {}.",
    generate_data_for_column="ner_tags",
    fewshot_example_columns=["tokens"],
    label_options=dataset.features["ner_tags"].feature.names,
)

print(prompt.get_prompt_text(prompt.label_options, dataset.select([0])))
```

Which outputs:

```text
Annotate each token with its named entity label: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC.

tokens: ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
ner_tags: [3, 0, 7, 0, 0, 0, 7, 0, 0]

tokens: {tokens}
ner_tags: 
```

To enhance prompt clarity, we can preprocess the dataset by converting labels into spans. This conversion can be 
accomplished using the `convert_token_labels_to_spans` function. Additionally, the function will provide the 
available label options:

```python
from datasets import load_dataset
from fabricator.prompts import BasePrompt
from fabricator.dataset_transformations import convert_token_labels_to_spans

dataset = load_dataset("conll2003", split="train")
dataset, label_options = convert_token_labels_to_spans(dataset, "tokens", "ner_tags", return_label_options=True)
prompt = BasePrompt(
    task_description="Annotate each token with its named entity label: {}.",
    generate_data_for_column="ner_tags",
    fewshot_example_columns=["tokens"],
    label_options=label_options,
)

print(prompt.get_prompt_text(prompt.label_options, dataset.select([0])))
```
Output:
```text
Annotate each token with its named entity label: MISC, ORG, PER, LOC.

tokens: EU rejects German call to boycott British lamb . 
ner_tags: EU is ORG entity.
German is MISC entity.
British is MISC entity.

tokens: {tokens}
ner_tags: 
```

As in text classification, we can also specify more semantically precise labels with the `expanded_label_mapping`:

```python
expanded_label_mapping = {
    0: "O",
    1: "B-person",
    2: "I-person",
    3: "B-location",
    4: "I-location",
    5: "B-organization",
    6: "I-organization",
    7: "B-miscellaneous",
    8: "I-miscellaneous",
}

dataset, label_options = convert_token_labels_to_spans(
    dataset=dataset,
    token_column="tokens",
    label_column="ner_tags",
    expanded_label_mapping=expanded_label_mapping,
    return_label_mapping=True
)
```

Output:

```text
Annotate each token with its named entity label: organization, person, location, miscellaneous.

tokens: EU rejects German call to boycott British lamb . 
ner_tags: EU is organization entity.
German is miscellaneous entity.
British is miscellaneous entity.

tokens: {tokens}
ner_tags: 
```

Once the dataset is created, we can use the `convert_spans_to_token_labels` function to convert spans back to labels 
IDs. This function will only add spans the occur only once in the text. If a span occurs multiple times, it will be
ignored. Note: this takes rather long is currently build on spacy. We are working on a faster implementation or welcome
contributions.

```python
from fabricator.dataset_transformations import convert_spans_to_token_labels

dataset = convert_spans_to_token_labels(
    dataset=dataset.select(range(20)),
    token_column="tokens",
    label_column="ner_tags",
    id2label=expanded_label_mapping
)
```

Outputs:

```Text
{'id': '1', 'tokens': ['Peter', 'Blackburn'], 'pos_tags': [22, 22], 'chunk_tags': [11, 12], 'ner_tags': [1, 2]}
```

## Adapt to arbitrary datasets

The `BasePrompt` class is designed to be easily adaptable to arbitrary datasets. Just like in the examples for text 
classification, token classification or question answering, you can specify the task description, the column to generate
data for and the fewshot example columns. The only difference is that you have to specify the optional label options 
yourself.

### Translation

```python
import os
from datasets import Dataset
from haystack.nodes import PromptNode
from fabricator import DatasetGenerator
from fabricator.prompts import BasePrompt

fewshot_dataset = Dataset.from_dict({
    "german": ["Der Film ist großartig!", "Der Film ist schlecht!"],
    "english": ["This movie is great!", "This movie is bad!"],
})

unlabeled_dataset = Dataset.from_dict({
    "english": ["This movie was a blast!", "This movie was not bad!"],
})

prompt = BasePrompt(
    task_description="Translate to german:",  # Since we do not have a label column, 
    # we can just specify the task description
    generate_data_for_column="german",
    fewshot_example_columns="english",
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
    fewshot_examples_per_class=2,  # Take both fewshot examples per prompt
    fewshot_sampling_strategy=None,  # Since we do not have a class label column, we can just set this to None 
    # (default)
    unlabeled_dataset=unlabeled_dataset,
    max_prompt_calls=2,
)

generated_dataset.push_to_hub("your-first-generated-dataset")
```

### Textual similarity

```python
import os
from datasets import load_dataset
from haystack.nodes import PromptNode
from fabricator import DatasetGenerator
from fabricator.prompts import BasePrompt
from fabricator.dataset_transformations.text_classification import convert_label_ids_to_texts

dataset = load_dataset("glue", "mrpc", split="train")
dataset, label_options = convert_label_ids_to_texts(dataset, "label", return_label_options=True)  # convert the
# label ids to text labels and return the label options

fewshot_dataset = dataset.select(range(10))
unlabeled_dataset = dataset.select(range(10, 20))

prompt = BasePrompt(
    task_description="Annotate the sentence pair whether it is: {}",
    label_options=label_options,
    generate_data_for_column="label",
    fewshot_example_columns=["sentence1", "sentence2"],  # we can pass an array of columns to use for the fewshot
)

prompt_node = PromptNode(
    model_name_or_path="gpt-3.5-turbo",
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_length=100,
)

generator = DatasetGenerator(prompt_node)
generated_dataset, original_dataset = generator.generate(
    prompt_template=prompt,
    fewshot_dataset=fewshot_dataset,
    fewshot_examples_per_class=1,  # Take 1 fewshot examples per class per prompt
    fewshot_sampling_column="label",  # We want to sample fewshot examples based on the label column
    fewshot_sampling_strategy="stratified",  # We want to sample fewshot examples stratified by class
    unlabeled_dataset=unlabeled_dataset,
    max_prompt_calls=2,
    return_unlabeled_dataset=True,  # We can return the original unlabelled dataset which might be interesting in this
    # case to compare the annotation quality
)

generated_dataset = generated_dataset.class_encode_column("label")

generated_dataset.push_to_hub("your-first-generated-dataset")
```

You can also easily switch out columns to be annotated if you want, for example, to generate a second sentence given a
first sentence and a label like:

```python
import os
from datasets import load_dataset
from haystack.nodes import PromptNode
from fabricator import DatasetGenerator
from fabricator.prompts import BasePrompt
from fabricator.dataset_transformations.text_classification import convert_label_ids_to_texts

dataset = load_dataset("glue", "mrpc", split="train")
dataset, label_options = convert_label_ids_to_texts(dataset, "label", return_label_options=True)  # convert the
# label ids to text labels and return the label options

fewshot_dataset = dataset.select(range(10))
unlabeled_dataset = dataset.select(range(10, 20))

prompt = BasePrompt(
    task_description="Generate a sentence that is {} to sentence1.",
    label_options=label_options,
    generate_data_for_column="sentence2",
    fewshot_example_columns=["sentence1", "label"],  # we can pass an array of columns to use for the fewshot
)

prompt_node = PromptNode(
    model_name_or_path="gpt-3.5-turbo",
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_length=100,
)

generator = DatasetGenerator(prompt_node)
generated_dataset, original_dataset = generator.generate(
    prompt_template=prompt,
    fewshot_dataset=fewshot_dataset,
    fewshot_examples_per_class=1,  # Take 1 fewshot examples per class per prompt
    fewshot_sampling_column="label",  # We want to sample fewshot examples based on the label column
    fewshot_sampling_strategy="stratified",  # We want to sample fewshot examples stratified by class
    unlabeled_dataset=unlabeled_dataset,
    max_prompt_calls=2,
    return_unlabeled_dataset=True,  # We can return the original unlabelled dataset which might be interesting in this
    # case to compare the annotation quality
)

generated_dataset = generated_dataset.class_encode_column("label")

generated_dataset.push_to_hub("your-first-generated-dataset")
```
