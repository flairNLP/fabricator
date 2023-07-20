# Evaluation of LLMs NLP Downstream Task capabilities through instruction-fine-tuning


## Settings

Evaluate the "downstream" capabilities through instructions.

1. Compare different existing LLMs on the same task.
2. Compare a baseline LLM with a fine-tuned LLM on the same task.
3. Compare a not-downstream fine-tuned LLM with a downstream fine-tuned LLM on the same task.



## Results

FT-LLama on evaluation split of Conll03

| Model | Prompt Type | Generated | Sampled | Accuracy | f1-score | 
| --- | --- | --- | --- | --- | --- |
| FT-LLama | zero-shot | 100  | 100 | 0.664 | 0.89 |
| FT-LLama | zero-shot | 1000 | 1000 | 0.47 | 0.81 |
| FT-LLama | zero-shot | 3250 | 3200 | 0.54 | 0.82 |
| LLama | zero-shot | 10000 | - | - | - |
| LLama | 3-shot | 100 | - | - | - |
| falcon-7B-instruct | zero-shot | 30 | - | - | - |
| falcon-7B-instruct | 3-shot | 30 | - | 0.13 | 0.18 |
| guanaco-33b-merged | 3-shot | 20 | 10 | 0.0 | 0.01 |
| chatGPT | 3-shot | 30 | 30 | 0.31 | 0.72|


<img width="568" alt="Screenshot 2023-07-20 at 09 56 50" src="https://github.com/flairNLP/ai-dataset-generator/assets/22773355/273c3dbf-1e8d-481e-9485-9cad4024aca4">

