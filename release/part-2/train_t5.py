import os
import argparse
from contextlib import nullcontext
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from transformers import GenerationConfig

from t5_utils import (
    count_parameters,
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from load_data import (
    DEFAULT_DATA_DIR,
    TASK_PREFIX,
    build_data_config_from_args,
    canonicalize_sql_query,
    get_t5_tokenizer,
    load_t5_data,
)
from utils import compute_metrics, compute_records, save_queries_and_records, set_random_seeds

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DEFAULT_RECORDS_DIR = os.path.join(BASE_DIR, 'records')
DEFAULT_CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
DEFAULT_GOLD_DEV_RECORDS_PATH = os.path.join(BASE_DIR, 'records', 'ground_truth_dev.pkl')


def normalize_sql(query):
    return ' '.join(query.strip().split())


def get_generation_config(args, model):
    return GenerationConfig(
        max_new_tokens=args.max_generation_length,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        early_stopping=args.num_beams > 1,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


def postprocess_decoded_query(query, args):
    normalized_query = normalize_sql(query)
    if args.canonicalize_sql:
        return canonicalize_sql_query(normalized_query)
    return normalized_query


def decode_queries(tokenizer, generated_ids, args):
    return [postprocess_decoded_query(query, args) for query in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]


def get_autocast_context(args):
    if DEVICE.type != 'cuda' or args.mixed_precision == 'none':
        return nullcontext()

    dtype = torch.float16 if args.mixed_precision == 'fp16' else torch.bfloat16
    return torch.autocast(device_type='cuda', dtype=dtype)


def get_grad_scaler(args):
    enabled = DEVICE.type == 'cuda' and args.mixed_precision == 'fp16'
    return torch.cuda.amp.GradScaler(enabled=enabled)


def generate_candidate_queries(args, model, tokenizer, encoder_input, encoder_mask):
    generation_config = get_generation_config(args, model)
    generation_outputs = model.generate(
        input_ids=encoder_input,
        attention_mask=encoder_mask,
        generation_config=generation_config,
        return_dict_in_generate=args.num_return_sequences > 1,
        output_scores=args.num_return_sequences > 1,
    )

    generated_ids = generation_outputs.sequences if hasattr(generation_outputs, 'sequences') else generation_outputs
    return decode_queries(tokenizer, generated_ids, args)


def select_queries_from_candidates(args, candidate_queries):
    if args.num_return_sequences == 1:
        return candidate_queries, None, None

    grouped_queries = [
        candidate_queries[idx: idx + args.num_return_sequences]
        for idx in range(0, len(candidate_queries), args.num_return_sequences)
    ]

    if args.beam_selection_strategy == 'top':
        return [group[0] for group in grouped_queries], None, None

    candidate_records, candidate_error_msgs = compute_records(candidate_queries)
    selected_queries = []
    selected_records = []
    selected_error_msgs = []

    for group_idx, group in enumerate(grouped_queries):
        start_idx = group_idx * args.num_return_sequences
        end_idx = start_idx + args.num_return_sequences
        group_records = candidate_records[start_idx:end_idx]
        group_error_msgs = candidate_error_msgs[start_idx:end_idx]

        selected_idx = 0
        for offset, error_msg in enumerate(group_error_msgs):
            if not error_msg:
                selected_idx = offset
                break

        selected_queries.append(group[selected_idx])
        selected_records.append(group_records[selected_idx])
        selected_error_msgs.append(group_error_msgs[selected_idx])

    return selected_queries, selected_records, selected_error_msgs


def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    parser.add_argument('--model_name', type=str, default='google-t5/t5-small',
                        help="T5 checkpoint/config name to use for the model and tokenizer")
    parser.add_argument('--freeze_encoder', action='store_true',
                        help="Freeze all encoder parameters during training")
    parser.add_argument('--freeze_decoder', action='store_true',
                        help="Freeze all decoder parameters during training")
    parser.add_argument('--freeze_embeddings', action='store_true',
                        help="Freeze the shared token embeddings during training")

    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW", "Adafactor"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--mixed_precision', type=str, default='none', choices=['none', 'fp16', 'bf16'])
    parser.add_argument('--gradient_checkpointing', action='store_true')

    parser.add_argument('--scheduler_type', type=str, default="linear", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=20,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=2,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")
    parser.add_argument('--load_from_experiment_name', type=str, default=None,
                        help="If set, load checkpoints from this experiment while saving outputs under experiment_name")
    parser.add_argument('--skip_training', action='store_true',
                        help="Skip training and only run evaluation/inference from a saved checkpoint")
    parser.add_argument('--skip_test_inference', action='store_true',
                        help="Skip test generation. Useful for quick dev sweeps.")

    # Data / generation hyperparameters
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--task_prefix', type=str, default=TASK_PREFIX)
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=512)
    parser.add_argument('--lowercase_inputs', action='store_true')
    parser.add_argument('--include_schema_in_input', action='store_true')
    parser.add_argument('--schema_prompt_mode', type=str, choices=['none', 'tables', 'full'], default=None)
    parser.add_argument('--canonicalize_sql', action='store_true')
    parser.add_argument('--no_normalize_whitespace', dest='normalize_whitespace', action='store_false')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--beam_selection_strategy', type=str, default='top', choices=['top', 'first_valid'])
    parser.add_argument('--max_generation_length', type=int, default=256)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--records_dir', type=str, default=DEFAULT_RECORDS_DIR)
    parser.add_argument('--checkpoint_root', type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument('--gold_dev_records_path', type=str, default=DEFAULT_GOLD_DEV_RECORDS_PATH)

    parser.set_defaults(normalize_whitespace=True)

    args = parser.parse_args()
    return args


def train(args, model, train_loader, dev_loader, optimizer, scheduler, scaler):
    best_f1 = -1
    epochs_since_improvement = 0

    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join(args.checkpoint_root, f'{model_type}_experiments', experiment_name)
    gt_sql_path = os.path.join(args.data_dir, 'dev.sql')
    gt_record_path = args.gold_dev_records_path
    model_sql_path = os.path.join(args.results_dir, f't5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(args.records_dir, f't5_{model_type}_{experiment_name}_dev.pkl')

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler, scaler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args,
            model,
            dev_loader,
            gt_sql_path,
            model_sql_path,
            gt_record_path,
            model_record_path,
        )
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate * 100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            wandb.log(
                {
                    'train/loss': tr_loss,
                    'dev/loss': eval_loss,
                    'dev/record_f1': record_f1,
                    'dev/record_em': record_em,
                    'dev/sql_em': sql_em,
                    'dev/error_rate': error_rate,
                },
                step=epoch,
            )

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break


def train_epoch(args, model, train_loader, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0
    total_tokens = 0
    optimizer.zero_grad(set_to_none=True)

    for step_idx, (encoder_input, encoder_mask, decoder_input, decoder_targets, _) in enumerate(tqdm(train_loader), start=1):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)
        labels = decoder_targets.masked_fill(decoder_targets == PAD_IDX, -100)

        with get_autocast_context(args):
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                labels=labels,
            )
            loss = outputs.loss
            scaled_loss = loss / args.gradient_accumulation_steps

        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = step_idx % args.gradient_accumulation_steps == 0 or step_idx == len(train_loader)
        if should_step:
            if args.max_grad_norm and args.max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            num_tokens = torch.sum(decoder_targets != PAD_IDX).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate.

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    tokenizer = get_t5_tokenizer(args.model_name)
    previous_use_cache = model.config.use_cache
    model.config.use_cache = True

    total_loss = 0
    total_tokens = 0
    candidate_queries = []

    for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs in tqdm(dev_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)
        labels = decoder_targets.masked_fill(decoder_targets == PAD_IDX, -100)

        with torch.no_grad():
            with get_autocast_context(args):
                outputs = model(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    labels=labels,
                )
            loss = outputs.loss
            num_tokens = torch.sum(decoder_targets != PAD_IDX).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            batch_candidate_queries = generate_candidate_queries(args, model, tokenizer, encoder_input, encoder_mask)

        candidate_queries.extend(batch_candidate_queries)

    generated_queries, selected_records, selected_error_msgs = select_queries_from_candidates(args, candidate_queries)

    save_queries_and_records(
        generated_queries,
        model_sql_path,
        model_record_path,
        records=selected_records,
        error_msgs=selected_error_msgs,
    )
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth,
        model_sql_path,
        gt_record_path,
        model_record_path,
    )
    error_rate = float(np.mean([bool(msg) for msg in model_error_msgs]))
    eval_loss = total_loss / max(total_tokens, 1)
    model.config.use_cache = previous_use_cache

    return eval_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    tokenizer = get_t5_tokenizer(args.model_name)
    previous_use_cache = model.config.use_cache
    model.config.use_cache = True
    candidate_queries = []

    for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)

        with torch.no_grad():
            batch_candidate_queries = generate_candidate_queries(args, model, tokenizer, encoder_input, encoder_mask)

        candidate_queries.extend(batch_candidate_queries)

    generated_queries, selected_records, selected_error_msgs = select_queries_from_candidates(args, candidate_queries)

    save_queries_and_records(
        generated_queries,
        model_sql_path,
        model_record_path,
        records=selected_records,
        error_msgs=selected_error_msgs,
    )
    model.config.use_cache = previous_use_cache
    print(f"Saved test SQL queries to {model_sql_path}")
    print(f"Saved test query records to {model_record_path}")


def main():
    # Get key arguments
    args = get_args()
    set_random_seeds(args.seed)
    if args.num_return_sequences > args.num_beams:
        raise ValueError('--num_return_sequences cannot be larger than --num_beams')
    if args.gradient_accumulation_steps < 1:
        raise ValueError('--gradient_accumulation_steps must be at least 1')

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.records_dir, exist_ok=True)
    os.makedirs(args.checkpoint_root, exist_ok=True)

    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    data_config = build_data_config_from_args(args, data_folder=args.data_dir)
    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size,
        args.test_batch_size,
        data_config=data_config,
        data_folder=args.data_dir,
    )
    model = initialize_model(args)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    total_parameters, trainable_parameters = count_parameters(model)
    scaler = get_grad_scaler(args)
    print(f"Model checkpoint: {args.model_name}")
    print(
        "Preprocessing configuration: "
        f"task_prefix={args.task_prefix!r}, "
        f"schema_prompt_mode={args.schema_prompt_mode or ('full' if args.include_schema_in_input else 'none')}, "
        f"lowercase_inputs={args.lowercase_inputs}, "
        f"normalize_whitespace={args.normalize_whitespace}, "
        f"canonicalize_sql={args.canonicalize_sql}, "
        f"max_input_length={args.max_input_length}, "
        f"max_target_length={args.max_target_length}"
    )
    print(
        "Freezing configuration: "
        f"freeze_encoder={args.freeze_encoder}, "
        f"freeze_decoder={args.freeze_decoder}, "
        f"freeze_embeddings={args.freeze_embeddings}, "
        f"gradient_checkpointing={args.gradient_checkpointing}, "
        f"mixed_precision={args.mixed_precision}, "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}"
    )
    print(
        "Generation configuration: "
        f"num_beams={args.num_beams}, "
        f"num_return_sequences={args.num_return_sequences}, "
        f"beam_selection_strategy={args.beam_selection_strategy}, "
        f"length_penalty={args.length_penalty}, "
        f"repetition_penalty={args.repetition_penalty}, "
        f"no_repeat_ngram_size={args.no_repeat_ngram_size}"
    )
    print(f"Trainable parameters: {trainable_parameters:,}/{total_parameters:,}")
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train
    if not args.skip_training:
        train(args, model, train_loader, dev_loader, optimizer, scheduler, scaler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    model.config.use_cache = True

    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(args.data_dir, 'dev.sql')
    gt_record_path = args.gold_dev_records_path
    model_sql_path = os.path.join(args.results_dir, f't5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(args.records_dir, f't5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args,
        model,
        dev_loader,
        gt_sql_path,
        model_sql_path,
        gt_record_path,
        model_record_path,
    )
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate * 100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    if not args.skip_test_inference:
        model_sql_path = os.path.join(args.results_dir, f't5_{model_type}_{experiment_name}_test.sql')
        model_record_path = os.path.join(args.records_dir, f't5_{model_type}_{experiment_name}_test.pkl')
        test_inference(args, model, test_loader, model_sql_path, model_record_path)


if __name__ == "__main__":
    main()
