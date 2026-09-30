"""Typed schema of ``configs/*.yaml``.

The schema declares types only; values come from ``configs/default.yaml``, so
there is exactly one place that states each hyperparameter.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PathsConfig:
    """Artifact locations."""

    output_root: str
    scene_dir: str
    reference_video_dir: str
    multiview_dir: str
    run_dir: str
    render_dir: str


@dataclass
class TriplaneConfig:
    """Triplane resolution and channel count."""

    resolution: list[int]
    multires: list[int]
    channels: int


@dataclass
class PeriodicConfig:
    """Periodic Deformation Field P."""

    heads: list[str]
    harmonics: int
    is_periodic: bool
    width: int
    max_log_scale_delta: float
    motion_grid: TriplaneConfig
    appearance_grid: TriplaneConfig


@dataclass
class DriftConfig:
    """Grounded Drift Field Δ."""

    enabled: bool
    heads: list[str]
    is_periodic: bool
    width: int
    view_embedding_dim: int
    grid: TriplaneConfig
    zero_reference_view: bool
    zero_t0: bool
    warmup_iterations: int


@dataclass
class ModelConfig:
    """The cinemagraph model."""

    sh_degree: int
    frozen: list[str]
    point_chunk_size: int
    activation_checkpointing: bool
    periodic: PeriodicConfig
    drift: DriftConfig


@dataclass
class LearningRatesConfig:
    """Adam learning rates."""

    xyz: list[float]
    field_mlp: list[float]
    field_grid: list[float]
    decay_steps: int
    sh_dc: float
    sh_rest: float
    opacity: float
    log_scale: float
    rotation: float


@dataclass
class MotionWeightsConfig:
    """Relative weights inside the motion smoothness term."""

    position: float
    rotation: float
    scale: float


@dataclass
class DriftChannelWeightsConfig:
    """Relative weights of Δ's output channels in its magnitude penalty."""

    translation: float
    rotation: float
    scale: float
    opacity: float
    sh: float


@dataclass
class LossesConfig:
    """Regulariser weights."""

    sample_gaussians: int
    appearance_magnitude: float
    appearance_smoothness: float
    appearance_delta: float
    scale_change: float
    motion_smoothness: float
    motion_delta: float
    motion_weights: MotionWeightsConfig
    twist_smoothness: float
    twist_delta: float
    drift_magnitude: float
    drift_channel_weights: DriftChannelWeightsConfig
    drift_view_chunk: int
    plane_smoothness: float


@dataclass
class RefinementConfig:
    """Diffusion refinement."""

    enabled: bool
    start_after: int
    weight: float
    stable_diffusion: str
    lcm_lora: str
    inference_steps: int
    rectification_weights: list[float]
    prompt: str
    dtype: str


@dataclass
class DensificationConfig:
    """Adaptive density control."""

    enabled: bool
    start_iteration: int
    stop_iteration: int
    interval: int
    gradient_threshold: float
    percent_dense: float
    min_opacity: float
    max_prune_fraction: float


@dataclass
class TrainConfig:
    """The 4D optimisation."""

    iterations: int
    batch_size: int
    seed: int
    supervision: str
    reference_view_probability: float
    field_grad_clip: float
    max_consecutive_skips: int
    strike_limit: int
    checkpoint_iterations: list[int]
    log_interval: int
    learning_rates: LearningRatesConfig
    losses: LossesConfig
    refinement: RefinementConfig
    densification: DensificationConfig


@dataclass
class RenderConfig:
    """Loop videos."""

    checkpoint: str
    n_loops: int
    fps: int
    cycle_seconds: float
    orbit_degrees: list[float]
    orbit_radius_scale: float
    orbit_camera_loops: int


@dataclass
class CandidatesConfig:
    """Grid of candidate reference cameras around the pivot."""

    horizontal_degrees: float
    vertical_degrees: float
    horizontal_steps: int
    vertical_steps: int
    forward_fractions: list[float]
    thumbnail_width: int
    min_coverage: float


@dataclass
class ReferenceConfig:
    """Pivot and reference view of the scene package."""

    selected: str
    width: int
    height: int
    pivot_patch_radius: int
    center_patch_fraction: float
    candidates: CandidatesConfig


@dataclass
class PromptConfig:
    """Motion prompt from the reference image (OpenAI Responses API)."""

    model: str
    reasoning_effort: str
    image_detail: str
    max_output_tokens: int
    timeout_seconds: float
    additional_instruction: str
    api_key_env: str
    base_url_env: str
    default_base_url: str


@dataclass
class SeedanceConfig:
    """Seedance 2.0 task settings."""

    base_url: str
    api_key_env: str
    model: str
    resolution: str
    ratio: str
    width: int
    height: int
    duration_seconds: int
    additional_instruction: str
    poll_interval_seconds: float
    task_timeout_seconds: float
    request_timeout_seconds: float


@dataclass
class WanConfig:
    """Wan 2.2 I2V settings."""

    model: str
    cache_dir: str
    memory_mode: str
    max_area: int
    frame_count: int
    inference_steps: int
    guidance_scale: float
    fps: int
    seed: int


@dataclass
class ContinuityConfig:
    """Largest accepted global change between the first two generated frames."""

    enabled: bool
    max_scale_change_percent: float
    max_translation_pixels: float
    max_rotation_degrees: float


@dataclass
class ReferenceVideoConfig:
    """Looping reference video generation."""

    frame_count: int
    backend: str
    prompt: PromptConfig
    seedance: SeedanceConfig
    wan: WanConfig
    continuity: ContinuityConfig


@dataclass
class OrbitConfig:
    """Training cameras around the pivot."""

    view_count: int
    max_yaw_degrees: float
    axis_tilt_degrees: float


@dataclass
class LiftingConfig:
    """VGGT-Omega depth and its alignment to the input 3DGS."""

    checkpoint: str
    image_resolution: int
    confidence_drop_percentile: float
    alignment_iterations: int
    alignment_huber_k: float
    min_valid_pixels: int


@dataclass
class FusionConfig:
    """Blend of the warp over the video model's output."""

    known_threshold: float
    max_warp_alpha: float
    erode_pixels: int
    feather_pixels: int
    feather_sigma: float


@dataclass
class InpaintingConfig:
    """TrajectoryCrafter and its condition."""

    base_model: str
    transformer: str
    caption_model: str
    caption_suffix: str
    negative_prompt: str
    inference_steps: int
    guidance_scale: float
    seed: int
    cpu_offload: str
    attention_query_chunk: int
    sweep_frames: int
    appearance_frames: int
    fusion: FusionConfig


@dataclass
class GenerationConfig:
    """Multi-view video generation."""

    frame_count: int
    orbit: OrbitConfig
    lifting: LiftingConfig
    inpainting: InpaintingConfig
    save_diagnostics: bool


@dataclass
class Config:
    """Root of the configuration."""

    dataset: str
    scene: str
    paths: PathsConfig
    reference: ReferenceConfig
    reference_video: ReferenceVideoConfig
    generation: GenerationConfig
    model: ModelConfig
    train: TrainConfig
    render: RenderConfig


@dataclass
class EvaluatedGroup:
    """One row of a results table."""

    group: str
    label: str
    cycle_seconds: float


@dataclass
class VividnessConfig:
    """Vividness Degree."""

    sample_fps: float
    motion_threshold_percent: float
    illumination_threshold_percent: float
    illumination_epsilon: float
    sea_raft: str


@dataclass
class NaturalnessConfig:
    """KVD against the reference videos."""

    reference_root: str
    reference_pattern: str
    reference_dataset_dirs: dict[str, str]
    i3d: str
    feature_cache: str
    clip_seconds: float
    clip_stride_seconds: float
    clip_frames: int
    frame_size: int
    max_clips_per_video: int
    subset_size: int
    subset_count: int
    kernel_degree: int
    kernel_coefficient: float
    seed: int


@dataclass
class LoopSeamConfig:
    """Seam SSIM and MALF."""

    samples_per_cycle: int
    max_ring_lags: int


@dataclass
class SharpnessConfig:
    """Variance of the Laplacian."""

    frames: int
    period_frames: int


@dataclass
class SceneQualityConfig:
    """VBench dimensions."""

    vbench_weights: str
    prompts: str


@dataclass
class EvaluationConfig:
    """Root of ``configs/evaluation.yaml``."""

    results_root: str
    output_root: str
    views: list[str]
    device: str
    methods: list[EvaluatedGroup]
    ablations: list[EvaluatedGroup]
    period_ablation: EvaluatedGroup
    vividness: VividnessConfig
    naturalness: NaturalnessConfig
    loop_seam: LoopSeamConfig
    sharpness: SharpnessConfig
    scene_quality: SceneQualityConfig
