# Part 2: T5 Fine-tuning Baseline Guide

This guide will help you achieve baseline results and answer Q4 and Q5.

## Implementation Summary

All TODO items have been implemented:
- ✅ `T5Dataset` class with data processing
- ✅ `normal_collate_fn` and `test_collate_fn` for batching
- ✅ `initialize_model` for T5 model initialization
- ✅ `save_model` and `load_model_from_checkpoint` for checkpointing
- ✅ `eval_epoch` for evaluation during training
- ✅ `test_inference` for test set inference

## Q4: Data Statistics and Processing

### Step 1: Compute Data Statistics

Run the following command to compute statistics for Table 1 and Table 2:

```bash
cd /Users/surajmishra/Downloads/release/part-2
python compute_data_statistics.py
```

This will output:
- **Before Preprocessing**: Raw text statistics (for Table 1)
- **After Preprocessing**: T5 tokenized statistics (for Table 2)

### Filling Table 1 (Before Preprocessing)

Use the "BEFORE PREPROCESSING" section from the script output:
- Number of examples
- Mean sentence length (in words)
- Mean SQL query length (in words)
- Vocabulary size (natural language)
- Vocabulary size (SQL)

### Filling Table 2 (After Preprocessing)

Use the "AFTER PREPROCESSING" section from the script output:
- Model name: `google-t5/t5-small`
- Mean sentence length (in tokens)
- Mean SQL query length (in tokens)
- Vocabulary size (natural language tokens)
- Vocabulary size (SQL tokens)

## Q5: T5 Fine-tuning Details

### Baseline Configuration

Here's a recommended baseline configuration for Table 3:

#### Data Processing
- **Description**: No additional preprocessing beyond tokenization. Raw natural language queries and SQL queries are tokenized using the T5 tokenizer.

#### Tokenization
- **Encoder Input**: Natural language queries tokenized with `T5TokenizerFast.from_pretrained('google-t5/t5-small')` with `add_special_tokens=True`
- **Decoder Input**: SQL queries tokenized with the same tokenizer. During training, decoder inputs are shifted by one position (teacher forcing).
- **Special Tokens**: T5's default EOS token is used. Decoder input starts with the first token of the tokenized SQL query.

#### Architecture
- **Model**: `google-t5/t5-small` (60M parameters)
- **Fine-tuning Strategy**: Full model fine-tuning (all parameters are trainable)
- **Components**: Both encoder and decoder are fine-tuned

#### Hyperparameters (Baseline)

| Hyperparameter | Value | Command Line Argument |
|----------------|-------|----------------------|
| Learning Rate | 1e-4 | `--learning_rate 1e-4` |
| Batch Size | 16 | `--batch_size 16` |
| Test Batch Size | 16 | `--test_batch_size 16` |
| Max Epochs | 10 | `--max_n_epochs 10` |
| Patience | 3 | `--patience_epochs 3` |
| Optimizer | AdamW | `--optimizer_type AdamW` |
| Weight Decay | 0.01 | `--weight_decay 0.01` |
| Scheduler | Cosine | `--scheduler_type cosine` |
| Warmup Epochs | 1 | `--num_warmup_epochs 1` |

## Running the Baseline

### Training Command

```bash
cd /Users/surajmishra/Downloads/release/part-2

python train_t5.py \
  --finetune \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 10 \
  --patience_epochs 3 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --experiment_name baseline_v1
```

### What This Does

1. **Loads Data**: Loads train/dev/test splits with T5 tokenization
2. **Initializes Model**: Loads pretrained T5-small and moves to GPU (if available)
3. **Training Loop**: 
   - Trains for up to 10 epochs
   - Evaluates on dev set after each epoch
   - Saves best model based on Record F1 score
   - Early stops if no improvement for 3 epochs
4. **Final Evaluation**: Loads best checkpoint and evaluates on dev set
5. **Test Inference**: Generates predictions for test set

### Output Files

After training, you'll find:
- `checkpoints/ft_experiments/baseline_v1/best_model.pt` - Best model checkpoint
- `results/t5_ft_baseline_v1_dev.sql` - Dev set predictions
- `records/t5_ft_baseline_v1_dev.pkl` - Dev set database records
- `results/t5_ft_baseline_v1_test.sql` - Test set predictions
- `records/t5_ft_baseline_v1_test.pkl` - Test set database records

### Evaluating Results

To evaluate your dev set predictions:

```bash
python evaluate.py \
  --predicted_sql results/t5_ft_baseline_v1_dev.sql \
  --predicted_records records/t5_ft_baseline_v1_dev.pkl \
  --development_sql data/dev.sql \
  --development_records records/dev_gt_records.pkl
```

## Expected Baseline Performance

With the baseline configuration, you should expect:
- **Record F1**: 0.40 - 0.60 (40-60%)
- **Record EM**: 0.30 - 0.50 (30-50%)
- **SQL EM**: 0.05 - 0.15 (5-15%)
- **Error Rate**: 0.10 - 0.30 (10-30%)

## Hyperparameter Tuning Tips

To improve performance, try:

1. **Learning Rate**: Try `5e-5`, `1e-4`, `3e-4`
2. **Batch Size**: Larger batches (32, 64) if GPU memory allows
3. **Max Length**: Increase `max_length` in generation (currently 512)
4. **Beam Search**: Change `num_beams=1` to `num_beams=4` or `num_beams=8`
5. **More Epochs**: Increase `max_n_epochs` to 15-20
6. **Warmup**: Try 2-3 warmup epochs

## Implementation Details for Q5

### Data Processing
"No additional preprocessing was applied. Natural language queries and SQL queries were loaded from `.nl` and `.sql` files respectively and tokenized using the T5 tokenizer."

### Tokenization
"Both encoder inputs (natural language) and decoder outputs (SQL) were tokenized using `T5TokenizerFast` from the `google-t5/t5-small` checkpoint with `add_special_tokens=True`. For training, decoder inputs were created by shifting the tokenized SQL sequence by one position to implement teacher forcing. The T5 tokenizer's default EOS token was used to mark sequence ends."

### Architecture
"The entire T5-small model (60M parameters) was fine-tuned, including both the encoder and decoder components. All model parameters were trainable. The model uses a standard encoder-decoder transformer architecture with 6 layers in both encoder and decoder, 8 attention heads, and a hidden size of 512."

### Hyperparameters
"See the table above for the complete list of hyperparameters used in the baseline configuration."

## Troubleshooting

### Out of Memory
- Reduce `batch_size` to 8 or 4
- Reduce `test_batch_size` to 8 or 4
- Reduce `max_length` in generation to 256

### Slow Training
- Increase `batch_size` if GPU memory allows
- Use gradient accumulation (requires code modification)
- Skip generation during training (only compute loss)

### Poor Performance
- Increase `max_n_epochs` to 15-20
- Try different learning rates (5e-5, 3e-4)
- Use beam search (`num_beams=4`)
- Check for data loading issues

## Next Steps

After achieving baseline results:
1. Run `compute_data_statistics.py` to fill Table 1 and Table 2
2. Fill Table 3 with your baseline configuration
3. Report dev set metrics (Record F1, Record EM, SQL EM)
4. Experiment with hyperparameters to improve performance
5. Submit test set predictions for leaderboard evaluation
