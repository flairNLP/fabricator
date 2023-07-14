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
| FT-LLama | non-verbal | 100  | 100 | 0.664 | 0.89 |
| FT-LLama | non-verbal | 1000 | 1000 | 0.47 | - |
| LLama | non-verbal | 10000 | - | - | - |
| falcon-7B-instruct | non-verbal | 30 | - | - | - |
| falcon-7B-instruct | verbal | 30 | - | 0.13 | 0.18 |
| guanaco-33b-merged | verbal | 20 | 10 | 0.0 | 0.01 |
| chatGPT | verbal | 30 | 30 | 0.31 | 0.72|