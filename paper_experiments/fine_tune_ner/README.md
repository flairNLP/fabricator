# Evaluation of LLMs NLP Downstream Task capabilities through instruction-fine-tuning

## Setup

Install library

```bash
# In root of project
python -m pip install -e .
```

Install relevant requirements for experiment

```bash
python -m pip install -r requirements.txt
```

## Fine-tune Model

```bash
torchrun --nproc_per_node=2 train.py \
    --model_name_or_path "../llama_hf" \
    --bf16 True \
    --output_dir dar_llama_big_noinp_clean \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
```

## Evaluate LLM with library

This will generate NER tokens based on the CONLL03 evaluation split and evaluate at it against the gold labels.

```bash
python evaluate.py --model_name_or_path "<HF_MODEL>"
```
