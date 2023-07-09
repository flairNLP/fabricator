# Dataset generation using LLMs

We use LLMs to generate and annotate unlabeled data for supervised learning.

## Setup

```
git clone git@github.com:whoisjones/ai-dataset-generator.git
cd ai-dataset-generator
conda create -y -n aidatasetgenerator python=3.10
conda activate aidatasetgenerator
pip install -r requirements.txt
```

## Usage

### Generate unlabeled data
Create some prompt to generate unlabeled data with fewshot examples:

```python
from datasets import load_dataset
from ai_dataset_generator.prompts import DataGenerationPrompt

dataset = load_dataset("imdb", split="train")
fewshot_examples = dataset.select([1, 2, 3])

input_variables = ["text"] # Column names as they occur in the dataset
output_format = "text" # indicates the output format of the LLM is text
prompt = DataGenerationPrompt(
    input_variables=input_variables,
    output_format=output_format,
    task_description="Generate similar texts.",
)
raw_prompt = prompt.get_prompt_text(fewshot_examples)
print(raw_prompt)
```
Output: 
```commandline
Generate similar texts.
Text: "I Am Curious: Yellow" is a risible and pretentious steaming pile. It doesn't matter what one's political views are because this film can hardly be taken seriously on any level. As for the claim that frontal male nudity is an automatic NC-17, that isn't true. I've seen R-rated films with male nudity. Granted, they only offer some fleeting views, but where are the R-rated films with gaping vulvas and flapping labia? Nowhere, because they don't exist. The same goes for those crappy cable shows: schlongs swinging in the breeze but not a clitoris in sight. And those pretentious indie movies like The Brown Bunny, in which we're treated to the site of Vincent Gallo's throbbing johnson, but not a trace of pink visible on Chloe Sevigny. Before crying (or implying) "double-standard" in matters of nudity, the mentally obtuse should take into account one unavoidably obvious anatomical difference between men and women: there are no genitals on display when actresses appears nude, and the same cannot be said for a man. In fact, you generally won't see female genitals in an American film in anything short of porn or explicit erotica. This alleged double-standard is less a double standard than an admittedly depressing ability to come to terms culturally with the insides of women's bodies.
Text: If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />One might feel virtuous for sitting thru it because it touches on so many IMPORTANT issues but it does so without any discernable motive. The viewer comes away with no new perspectives (unless one comes up with one while one's mind wanders, as it will invariably do during this pointless film).<br /><br />One might better spend one's time staring out a window at a tree growing.<br /><br />
Text: This film was probably inspired by Godard's Masculin, féminin and I urge you to see that film instead.<br /><br />The film has two strong elements and those are, (1) the realistic acting (2) the impressive, undeservedly good, photo. Apart from that, what strikes me most is the endless stream of silliness. Lena Nyman has to be most annoying actress in the world. She acts so stupid and with all the nudity in this film,...it's unattractive. Comparing to Godard's film, intellectuality has been replaced with stupidity. Without going too far on this subject, I would say that follows from the difference in ideals between the French and the Swedish society.<br /><br />A movie of its time, and place. 2/10.
Text: {text}
```
Use our DatasetGenerator to generate unlabeled examples:
```python
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator

prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"), max_length=100)
generator = DatasetGenerator(prompt_node)
generated_dataset = generator.generate(
    support_examples=fewshot_examples, # from above
    prompt_template=prompt, # from above
    max_prompt_calls=3, # max number of calls to the LLM
    support_examples_per_prompt=1, # number of support examples per prompt
)

generated_dataset.push_to_hub("your-first-generated-dataset")
```

### Annotate unlabeled data
Create some prompt to annotate unlabeled data using few-shot examples:

```python
from datasets import load_dataset
from ai_dataset_generator.prompts import DataGenerationPrompt

dataset = load_dataset("imdb", split="train")
fewshot_examples = dataset.select([1, 2, 3])

input_variables = ["text"]  # Column name from dataset
target_variable = "label"  # Also column name from dataset, indicates the variable needs to be annotated
output_format = "single_label_classification" # Annotation format can be "text", "single_label", "multi_label", "token_classification" and determines how the LLM is prompted for the annotation
idx2label = {idx: key for idx, key in enumerate(fewshot_examples.features[target_variable].names)}

prompt = DataGenerationPrompt(
    input_variables=input_variables,
    output_format=output_format,
    target_variable=target_variable,
    classification_labels=idx2label,
    task_description="Classify the review whether it's positive or negative",
)
raw_prompt = prompt.get_prompt_text(fewshot_examples)
print(raw_prompt)
```
Output:
```commandline
Classify the review whether it's positive or negative Your prediction must be exactly one of the following labels: 0: neg, 1: pos.
Text: "I Am Curious: Yellow" is a risible and pretentious steaming pile. It doesn't matter what one's political views are because this film can hardly be taken seriously on any level. As for the claim that frontal male nudity is an automatic NC-17, that isn't true. I've seen R-rated films with male nudity. Granted, they only offer some fleeting views, but where are the R-rated films with gaping vulvas and flapping labia? Nowhere, because they don't exist. The same goes for those crappy cable shows: schlongs swinging in the breeze but not a clitoris in sight. And those pretentious indie movies like The Brown Bunny, in which we're treated to the site of Vincent Gallo's throbbing johnson, but not a trace of pink visible on Chloe Sevigny. Before crying (or implying) "double-standard" in matters of nudity, the mentally obtuse should take into account one unavoidably obvious anatomical difference between men and women: there are no genitals on display when actresses appears nude, and the same cannot be said for a man. In fact, you generally won't see female genitals in an American film in anything short of porn or explicit erotica. This alleged double-standard is less a double standard than an admittedly depressing ability to come to terms culturally with the insides of women's bodies.
Label: 0
Text: If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />One might feel virtuous for sitting thru it because it touches on so many IMPORTANT issues but it does so without any discernable motive. The viewer comes away with no new perspectives (unless one comes up with one while one's mind wanders, as it will invariably do during this pointless film).<br /><br />One might better spend one's time staring out a window at a tree growing.<br /><br />
Label: 0
Text: This film was probably inspired by Godard's Masculin, féminin and I urge you to see that film instead.<br /><br />The film has two strong elements and those are, (1) the realistic acting (2) the impressive, undeservedly good, photo. Apart from that, what strikes me most is the endless stream of silliness. Lena Nyman has to be most annoying actress in the world. She acts so stupid and with all the nudity in this film,...it's unattractive. Comparing to Godard's film, intellectuality has been replaced with stupidity. Without going too far on this subject, I would say that follows from the difference in ideals between the French and the Swedish society.<br /><br />A movie of its time, and place. 2/10.
Label: 0
Text: {text}
Label: 
```

Use our dataset generator to produce datasets and upload to huggingface hub.

```python
import os
from haystack.nodes import PromptNode
from ai_dataset_generator import DatasetGenerator

unlabeled_examples = dataset.select([10, 11, 12])

prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"))
generator = DatasetGenerator(prompt_node)
generated_dataset, original_dataset = generator.generate(
    support_examples=fewshot_examples, # from above
    unlabeled_examples=unlabeled_examples,
    prompt_template=prompt, # from above
    max_prompt_calls=3, # stop when this number of prompts has been called
    support_examples_per_prompt=1, # how many fewshot examples per prompt should be used
    return_original_dataset=True,
)

generated_dataset.push_to_hub("your-first-generated-dataset")
original_dataset.push_to_hub("original-dataset-to-compare")
```

### Pre- and postprocessing of datasets

We require a flat structure of dataset for the LLM prompting. You can easily adjust your datasets with huggingface's Dataset transformations.
At the example of Extractive QA with Squad v2.0:
- We pre-process by flatten the structure, rename the answer column and convert the type to str if the question can be answered.
- We post-process by calculating the answer_start of the generated answer which should occur exactly (substring) and only once in the text.
```python
from datasets import load_dataset
from haystack.nodes import PromptNode

from ai_dataset_generator import DatasetGenerator
from ai_dataset_generator.prompts import DataGenerationPrompt

def preprocess(example):
    if example["answer"]:
        example["answer"] = example["answer"].pop()
    else:
        example["answer"] = ""
    return example

dataset = load_dataset("squad_v2", split="train").flatten().rename_column("answers.text", "answer").map(preprocess)
unlabeled_examples = dataset.select([100, 105, 110])
fewshot_examples = dataset.select([50, 55, 60, 65])

# Column names as they occur in the dataset
input_variables = ["context", "question"]  # Inputs can be many texts, so either be a List[str] or a str
target_variable = "answer"  # Target / annotation variable needs to be a str
output_format = "text" # Annotation format can be "text", "single_label", "multi_label", "token_classification" and determines how the LLM is prompted for the annotation

prompt = DataGenerationPrompt(
    input_variables=input_variables,
    output_format=output_format,
    target_variable=target_variable,
    task_description="Given a context and a question, generate, if possible, an answer to this question that occurs exactly and only once in the text (substring).",
)
raw_prompt = prompt.get_prompt_text(fewshot_examples)
print(raw_prompt)

prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"))
generator = DatasetGenerator(prompt_node)
generated_dataset, original_dataset = generator.generate(
    support_examples=fewshot_examples,
    unlabeled_examples=unlabeled_examples,
    prompt_template=prompt,
    max_prompt_calls=3,
    support_examples_per_prompt=2,
    return_original_dataset=True,
)

def postprocess(example):
    answer_start = example["context"].find(example["answer"])
    if answer_start < 0:
        print(
            f'Could not calculate the answer start because the context "{example["context"]}" '
            f'does not contain the answer "{example["answer"]}".'
        )
        answer_start = -1
    else:
        # check that the answer doesn't occur more than once in the context
        second_answer_start = example["context"].find(example["answer"], answer_start + 1)
        if second_answer_start >= 0:
            print(
                "Could not calculate the answer start because the context contains the answer more than once."
            )
            answer_start = -1
        else:
            answer_start = answer_start
    example["answer_start"] = answer_start
    return example

generated_dataset = generated_dataset.map(postprocess)
generated_dataset.push_to_hub("your-first-generated-dataset")
```
