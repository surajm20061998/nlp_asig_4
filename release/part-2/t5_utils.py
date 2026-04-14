import os

import torch

import transformers
from transformers import Adafactor, T5Config, T5ForConditionalGeneration
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_NAME = 'google-t5/t5-small'

def setup_wandb(args):
    wandb.init(
        project='nlp-hw4-text-to-sql',
        name=args.experiment_name,
        config=vars(args),
    )

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    model_name = getattr(args, 'model_name', MODEL_NAME)

    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id

    apply_parameter_freezing(args, model)
    model.to(DEVICE)
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    mkdir(checkpoint_dir)
    checkpoint_name = 'best' if best else 'last'
    model.save_pretrained(os.path.join(checkpoint_dir, checkpoint_name))

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_experiment_name = getattr(args, 'load_from_experiment_name', None) or args.experiment_name
    checkpoint_root = getattr(args, 'checkpoint_root', 'checkpoints')
    checkpoint_dir = os.path.join(checkpoint_root, f'{model_type}_experiments', checkpoint_experiment_name)
    checkpoint_name = 'best' if best else 'last'
    model = T5ForConditionalGeneration.from_pretrained(os.path.join(checkpoint_dir, checkpoint_name))
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id
    model.to(DEVICE)
    return model

def freeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def apply_parameter_freezing(args, model):
    if getattr(args, 'freeze_embeddings', False):
        freeze_module(model.shared)

    if getattr(args, 'freeze_encoder', False):
        freeze_module(model.encoder)

    if getattr(args, 'freeze_decoder', False):
        freeze_module(model.decoder)


def count_parameters(model):
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total_parameters, trainable_parameters

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    elif args.optimizer_type == "Adafactor":
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise NotImplementedError(f'Unsupported optimizer type: {args.optimizer_type}')

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
