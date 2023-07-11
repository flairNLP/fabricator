# Evaluation of LLMs NLP Downstream Task capabilities through instruction-fine-tuning


## Settings

Evaluate the "downstream" capabilities through instructions.

1. Compare different existing LLMs on the same task.
2. Compare a baseline LLM with a fine-tuned LLM on the same task.
3. Compare a not-downstream fine-tuned LLM with a downstream fine-tuned LLM on the same task.



## Results

FT-LLama on evaluation  split of Conll03


100 Samples

Accuracy: 0.664

Classification Report
            precision   recal    f1-score   support
0           1.00       1.00     1.00       143
1           1.00       0.90     0.95       31
2           0.00       0.00     0.00       24
3           0.91       0.91     0.91       46
4           0.00       0.00     0.00       7
5           0.99       0.89     0.93       80
6           0.00       0.00     0.00       10
7           0.77       0.77     0.77       13
8           0.00       0.00     0.00       5
                                           
mic (avg)   0.97       0.82     0.89       359
mac (avg)   0.52       0.50     0.51       359


1000 Samples: 

Accuracy: 0.47


