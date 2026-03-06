from dataclasses import dataclass, field
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import NormalizationMode
from lerobot.configs.policies import PreTrainedConfig

@PreTrainedConfig.register_subclass("dual_diffusion")
@dataclass
class DualDiffusionConfig(DiffusionConfig):
    """Configuration class for DualDiffusionPolicy."""
    
    # This identifies the policy in the factory
    type: str = "dual_diffusion"

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 7

    # Architecture
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    
    # Unet
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    
    # Noise scheduler
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        # Run standard validation from DiffusionConfig
        super().__post_init__()

        # Custom validation for Dual architecture
        # We must ensure action dimension is divisible by 2 for the split
        action_dim = self.output_features["action"].shape[0]
        if action_dim % 2 != 0:
            raise ValueError(f"Action dimension ({action_dim}) must be even for DualDiffusion to split it.")