from dataclasses import dataclass
import simple_parsing as sp


@dataclass
class TrainingArguments:
    """Help string for this group of command-line arguments"""

    seed: int = 42  # all seed
    local_rank: int = None  # ddp local rank
    accelerator: str = "cpu"
    strategy: str = None  # ddp, ddp2, ddp_spawn, ddp_cpu, deepspeed, etc...
    data_dir: str = "nihcc"  # target pytorch lightning data dirs
    ratio: float = 0.2  # train/valid split ratio
    output_dir: str = "output"  # model output path
    config_path: str = "config/dense_model.json"
    num_workers: int = 4 # how many proc map?
    learning_rate: float = 0.001  # learning rate
    warmup_ratio: float = 0.2  # learning rate scheduler warmup ratio per EPOCH
    max_lr: float = 0.01  # lr_scheduler max learning rate
    div_factor: int = 25  # initial_lr = max_lr/div_factor
    final_div_factor: int = 1e4  # (max_lr/div_factor)*final_div_factor is final lr
    weight_decay: float = 0.0001  # weigth decay
    per_device_train_batch_size: int = 1  # The batch size per GPU/TPU core/CPU for training.
    per_device_eval_batch_size: int = 1  # The batch size per GPU/TPU core/CPU for evaluation.
    valid_on_cpu: bool = False  # If you want to run validation_step on cpu -> true
    model_select: str = "linear"  # linear or rnn
    truncated_bptt_steps: int = 1  # TBPTT step size
    deepspeed_config: str = "ds_config/zero2.json"
    dropout_p: float = 0.0  # Drop path rate (default: 0.0)
    cutoff_epoch: int = 0  # if drop_mode is early / late, this is the epoch where dropout ends / starts
    drop_mode: str = sp.field(default="standard", choices=["standard", "early", "late"])  # drop mode
    drop_schedule: str = sp.field(
        default="constant", choices=["constant", "linear"]
    )  # drop schedule for early dropout / s.d. only
    dimension: int = sp.field(default=2048, choices=[2048, 4096]) # dimension of the model
    attention_type: str = sp.field(default="cbam", choices=["se", "eca", "cbam", "external", "coordatt"])
    fusion_method: str = sp.field(default="add", choices=["concat", "add", "max"])
    data_aug: bool = False
    dataset: str = "nih"