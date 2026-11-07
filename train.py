import contextlib
import copy
import gc
import glob
import math
import os
import random
import re
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import time

import cv2
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.amp import GradScaler, autocast
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.io import decode_image
from torchvision.utils import make_grid, save_image

# ====================================== GLOBAL ======================================
VERBOSE = False
start_time = time()
time_limit = 9 * 60 * 60
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

checkpoint_path = "checkpoint.pt"
checkpoint_load_path = '/kaggle/input/notebooks/afsalali/stylegan/checkpoint.pt'

save_best_path = "save_best.pt"
visualization = False


@dataclass
class Config:
    batch_size: int = 512
    N_DIM: int = 512
    leaky_relu_negative_slope: float = 0.2
    n_blocks: int = 8
    H: int = 4
    W: int = 4
    num_mapping_layers: int = 8
    output_pixel = 512

    fmap_base_power = 12
    fmap_decay = 1.0
    fmap_max_power = math.log2(N_DIM)


config = Config()

# ====================================== DDP ======================================
ddp = os.environ.get("RANK", -1) != -1
world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
device = "cuda" if torch.cuda.is_available() else "cpu"
local_rank = 0
MASTER_PROC = True

if ddp:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    MASTER_PROC = local_rank == 0

random_gen = torch.Generator(device=device)
random_gen.seed()

# ====================================== FUNCTIONS ======================================



def nf(config: Config, stage: int) -> int:
    power = min(
        (config.fmap_base_power - ((stage + 1) * config.fmap_decay)),
        config.fmap_max_power,
    )
    return int(2**power)


def clear_ram_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_shards(
    image_paths: list[str],
    output_path: str,
    shard_size: int,
    num_workers: int,
    local_rank: int,
    world_size: int
) -> None:
    os.makedirs(output_path, exist_ok=True)

    def load_image(path: str) -> torch.Tensor:
        img = decode_image(path)
        return img

    log(f"Creating shards with {shard_size} batches", header=True, master_proc=MASTER_PROC)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        shard_batches = list(range(0, len(image_paths), shard_size))[
            local_rank::world_size
        ]

        for start_index in shard_batches:
            shard_t1 = time()
            
            end_index = min(start_index + shard_size, len(image_paths)) - 1

            shard_path = os.path.join(
                output_path, f"shard_from_{start_index}_to_{end_index}.pt"
            )

            if os.path.exists(shard_path):
                continue

            batch_paths = image_paths[start_index : end_index + 1]
            
            # Decode images concurrently
            images = list(executor.map(load_image, batch_paths))
            image_shard = torch.stack(images)
            torch.save(image_shard, shard_path)

            log(f"Saved {shard_path} ({len(images)} images) Took {(time()-shard_t1):.2f}")
            
            del images
            del image_shard

    log("Shards Saved Successfully...", master_proc=MASTER_PROC)


def balance_shards(shard_bounds: list[tuple[int, int]], batch_size: int, world_size: int) -> list[tuple[int, int]]:
    # 1. Keep ONLY shards that have a full batch_size (drop incomplete tail shards)
    complete_shards = [s for s in shard_bounds if (s[1] - s[0]) == batch_size]
    # 2. Trim so total shards is a multiple of world_size
    n = (len(complete_shards) // world_size) * world_size
    return complete_shards[:n]

def get_pixel(stage: int)->int:
    return 4 * (2 ** (stage - 1))

def save_model(
    models: dict[str, nn.Module],
    optimizers: dict[str, optim.Optimizer],
    n_images: int,
    fid_losses: list[float],
    epoch: int,
    epoch_summaries: list[dict],
    stage: int,
    path: str = "checkpoint.pt"
) -> None:
    
    torch.save(
        {
            "gen_state_dict": models["gen"].state_dict(),
            "disc_state_dict": models["disc"].state_dict(),
            "gen_ema_state_dict": models["gen_ema"].state_dict(),
            "optimizer_gen_state_dict": optimizers["gen"].state_dict(),
            "optimizer_disc_state_dict": optimizers["disc"].state_dict(),
            "fid_loss": fid_losses[-1],
            "n_images": n_images,
            "fid_losses": fid_losses,
            "epoch": epoch,
            "stage": stage,
            'epoch_summaries': epoch_summaries
        },
        path,
    )
    gc.collect()


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    images = images * 127.5 + 127.5
    return images.clamp(0, 255).to(torch.uint8)


def from_uint8(images: torch.Tensor) -> torch.Tensor:
    return images.float() * (2.0 / 255.0) - 1.0

    
def lr_update_scheduler(optimizer: optim.Optimizer, default_lr: float, stage: int, previous_stage: int | None) -> int:
    schedule_dict = {
        6: 0.0015,
        7: 0.002,
        8: 0.003,
        9: 0.005,
    }
    if previous_stage != stage:
        lr = schedule_dict.get(stage, default_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return stage

def log(string: str, master_proc: bool = True, verbose: bool = True, header: bool = False, end: str = '\n') -> None:
    header_limit = 90
    string_length = len(str(string))
    
    if header:
        separator_length = ((header_limit - string_length) // 2) - 1
        string = ("="*separator_length) + ' ' + string + ' ' + ("="*separator_length)

    if master_proc and verbose:
        print(string, end=end, flush=True)

def get_batch_params(n_images: int, stage: int, fade_map: dict[int, int], config: Config) -> tuple[int, int, float]:
    fade_limit = fade_map[stage] # gets appropriate fade-limit for stage
    fade_sum = sum([fade_map[n+1] for n in range(stage)]) # sum of fade limits 
    fade_previous = sum([fade_map[n] for n in range(1, stage)]) # sum of prev fade limits, on stage 1 it will be 0

    
    if (stage < config.n_blocks) and (n_images > fade_sum*2): 
        stage += 1

        # Recalculate fade limit for alpha 
        fade_limit = fade_map[stage] 
        fade_previous = sum([fade_map[n] for n in range(1, stage)]) # on stage it will be 0


    alpha = min(((n_images - (fade_previous*2)) / fade_limit), 1)
    pixels = get_pixel(stage)
    n_batch = pixel_batches.get(pixels, 32) // world_size

    return stage, n_batch, alpha


def get_real_at_stage(disc: nn.Module, real_images: torch.Tensor, config: Config, stage: int) -> torch.Tensor:
    target_res = get_pixel(stage)
    factor = config.output_pixel // target_res
    real_images = from_uint8(real_images)
    real_images = disc.downscale2d(real_images, factor)
    return real_images

def dist_barrier(ddp: bool) -> None:
    if ddp:
        torch.cuda.synchronize()
        dist.barrier()

def calculate_fid_authentic(
    real_loader: DataLoader,
    eval_gen: nn.Module,
    raw_disc: nn.Module,
    stage: int,
    fid_mini_batch: int = 64,
    feature: int = 768
) -> float:
    """
    real_loader / fake_loader: iterables yielding batches of images (N, 3, H, W).
      - If normalize=True: images should be float in [0, 1]
      - If normalize=False: images should be uint8 in [0, 255]
    """
    fid = FrechetInceptionDistance(feature=feature, normalize=True).to(device)

    # Clear Loading Head
    clear_ram_gpu()
    eval_gen.eval()

    
    log(f"GPU {local_rank} Starting FID", verbose=VERBOSE, header=True)

    for idx, batch in enumerate(real_loader):
        
        log(f"GPU {local_rank} batch {idx}", verbose=VERBOSE)
        
        for start_index in range(0, batch.shape[0], fid_mini_batch):
            
            end_index = min(start_index + fid_mini_batch, batch.shape[0])
            cur_mini = end_index - start_index
            if cur_mini == 0:
                continue

            # Real Standardized Images (scaled from [-1, 1] to [0, 1])
            real_images = get_real_at_stage(
                raw_disc, batch[start_index:end_index], config, stage
            )
            real_images = real_images.to(device)
            real_images = (real_images * 0.5 + 0.5).clamp(0.0, 1.0)
            fid.update(real_images, real=True)

            # Fake Images
            latent_z = torch.randn(
                (cur_mini, config.N_DIM), generator=random_gen, device=device
            ).to(dtype=dtype, non_blocking=True)

            with autocast(device_type=device_type, dtype=dtype), torch.inference_mode():
                fake_images = eval_gen(
                    latent_z, alpha=1, step=stage, inference_mode=True, psi=0.5
                )
            fake_images = fake_images.float()
            fake_images = (fake_images * 0.5 + 0.5).clamp(0.0, 1.0)
            fid.update(fake_images, real=False)

    # dist_barrier(ddp)
    log(f"GPU {local_rank} FID loop done", verbose=VERBOSE)
    fid_score = fid.compute().item()

    log(f"FID Finished Computation GPU {local_rank}", verbose=VERBOSE, header=True)
    
    del fid
    clear_ram_gpu()
    
    return fid_score


def r_penalty(discriminator_output: torch.Tensor, images: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:

    grads = torch.autograd.grad(
        outputs=discriminator_output.sum(),
        inputs=images,
        create_graph=True,
        only_inputs=True,
    )[0]

    penalty = grads.square().sum([1, 2, 3])  # sum over C, H, W -> shape [B]
    r1_loss = penalty.mean() * (gamma / 2)
    return r1_loss

def get_gen_ema(gen_ema: nn.Module | None, raw_gen: nn.Module) -> nn.Module:
    if gen_ema is None:
        gen_ema = copy.deepcopy(raw_gen)
    
    for p in gen_ema.parameters():
        p.requires_grad_(False)

    return gen_ema.eval()
      

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, mini_batch: int):
    G_smoothing_kimg = 10
    beta = 0.5 ** (mini_batch / (G_smoothing_kimg * 1000.0))
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.lerp_(param, 1.0 - beta)
    for eb, b in zip(ema_model.buffers(), model.buffers()):
        eb.copy_(b) 

def post_epoch_process(
    models: dict[str, nn.Module],
    optimizers: dict[str, optim.Optimizer],
    n_images: int,
    stage: int,
    best_fid: float,
    start_time: float,
    epoch: int,
    fid_score: float,
    epoch_start_time: float,
    epoch_summaries: list[dict],
    checkpoint_path: str = "checkpoint.pt",
    save_best_path: str = "save_best.pt",
) -> tuple[float, list[dict]]:
    # Free RAM load images to
    gc.collect()

    # Time Taken
    time_taken = time() - start_time
    epoch_time = time() - epoch_start_time
    
    pixels = get_pixel(stage)
    is_model_best = all((fid_score < score) for score in fid_losses)
    fid_losses.append(fid_score)

    # Norm Calculation
    norm = torch.nn.utils.clip_grad_norm_(
        models["gen"].parameters(), max_norm=float("inf")
    )
    epoch_summary = {
        'epoch': epoch,
        'fid_score': fid_score,
        'norm': norm,
        'time_taken': time_taken,
        'epoch_time': epoch_time,
        'n_images': n_images,
        'stage': stage,
        'pixels': pixels
    }
    epoch_summaries.append(epoch_summary)

    # Save Model Checkpoint
    save_model(models, optimizers,  n_images, fid_losses, epoch, epoch_summaries, stage, path=checkpoint_path)

    # Save Best Model
    if is_model_best:
        save_model(models, optimizers,  n_images, fid_losses, epoch, epoch_summaries, stage, path=save_best_path)

    log(
        f"Epoch {epoch} | FID: {fid_score} | Norm: {norm} | Pixels: {pixels} | Epoch Time: {epoch_time:.2f}s | Total Time: {time_taken:.2f}s",
        master_proc=MASTER_PROC
    )

    if is_model_best:
        best_fid = fid_score
    return best_fid, epoch_summaries
    

# =================================== Visualization Functions ===================================

def visualize_network(model: nn.Module, x: torch.Tensor | None = None, stage: int = 8, alpha: float = 1) -> None:
    if x is None:
        dev = next(model.parameters()).device
        if hasattr(model, "mapping_network"):
            x = torch.randn(4, model.config.N_DIM, device=dev)
        else:
            res = get_pixel(stage)
            x = torch.randn(4, 3, res, res, device=dev)

    m_dict = {
        k: []
        for k in [
            "PixelNorm", "EqualizedLinear",
            "LeakyReLU", "InstanceNorm2d",
            "UpscaleConv2d", "Blur",
            "Noise", "EqualizedConv2d",
            "ToRGB", "DownsamplingBlock",
            "MappingNetwork", "Block",
            "FinalBlock", "MinibatchStdDev",
        ]
    }

    def _forward_hook(
        module: nn.Module, inp: tuple[torch.Tensor, ...],
        out: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> None:
        n = module.__class__.__name__
        if n in m_dict and isinstance(out, torch.Tensor):
            t = out.detach().float().cpu().view(-1)
            hy, hx = torch.histogram(t, density=True)
            m_dict[n].append(
                (
                    hx[:-1],
                    hy,
                    t.mean().item(),
                    t.std().item(),
                    (t.abs() <= 1e-3).float().mean().item(),
                )
            )

    hooks = [m.register_forward_hook(_forward_hook) for m in model.modules()]
    with torch.no_grad():
        model(x, alpha=alpha, step=stage)
    for h in hooks:
        h.remove()

    for name, itm in m_dict.items():
        if not itm:
            continue
        print(f"----------- {name} -----------")
        for hx, hy, mean, std, sat in itm:
            plt.plot(hx, hy)
            print(f"Mean: {mean:.4f}, Std: {std:.4f}, Saturation: {sat:.4f}")
        plt.title(name)
        plt.show()


def save_images(
    generator: nn.Module,
    n_images: int,
    stage: int,
    output_dir: str,
    latent_dim: int = 512,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    latent_z: None | torch.Tensor = None
) -> None:
    """Generate n_images images at n_pixels resolution and save them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    pixels = get_pixel(stage)

    generator.eval()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.inference_mode(), autocast(dtype=dtype, device_type=device_type):
        if latent_z is not None:
            z = latent_z.clone()
        else:
            z = torch.randn(4, latent_dim, device=device, dtype=dtype, generator=random_gen)
        img = generator(z, alpha=1, step=stage, inference_mode=True, psi=0.5)
        img = to_uint8(img)
        img = vutils.make_grid(img, nrow=2)
        img = img.permute(1, 2, 0).to("cpu").numpy()
        plt.imsave(os.path.join(output_dir, f"{pixels}px_{n_images:04d}.png"), img)
    
    # Remove if gen_ema
    # generator.train()


def _sort_key(path: str) -> tuple[int, int] | str:
    match = re.match(r"(\d+)px_(\d+)", os.path.basename(path))
    return (int(match.group(1)), int(match.group(2))) if match else path

def create_video(
    images_saved_path: str,
    output_pixel: int = 512,
    output_video: str = "output.mp4",
    fps: int = 10
) -> None:
    """Combine saved images into a video, resizing every frame to output_pixel x output_pixel."""
    paths = sorted(glob.glob(os.path.join(images_saved_path, "*.png")), key=_sort_key)

    size = (output_pixel, output_pixel)
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    for path in paths:
        img = cv2.imread(path)
        frame = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

        # Add title and image number
        pixel_match = re.match(r"(\d+)px_(\d+)", os.path.basename(path))
        if pixel_match:
            pixel = pixel_match.group(1)
            n_images = pixel_match.group(2)

            cv2.putText(
                frame,
                f"Pixels: {pixel}",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Images: {n_images}",
                (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

    writer.release()
    log(f"Saved video to {output_video}", verbose=VERBOSE)


def plot_on_pixels(epoch_summaries: list[dict], save_path: str ='fid_scores.png'):
    plots = {}

    for summary in epoch_summaries:
        pixels = summary['pixels']
        if pixels not in plots:
            plots[pixels] = [[], []]

        plots[pixels][0].append(summary['epoch'])
        plots[pixels][1].append(summary['fid_score'])

    if not(plots and (len(plots) % 2 == 0)):
        log("Miss Match in Plots Unable to show subplots")
        return 
    

    rows, cols = 2, math.ceil(len(plots) / 2)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    fig.suptitle("FID Score Across Pixel Resolutions", fontsize=16)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax, (pixels, (epochs, fid_scores)) in zip(axes, plots.items()):
        ax.plot(epochs, fid_scores)

        ax.set_title(f"Pixels {pixels}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("FID Score")

        # X-axis ticks
        x_step = max(1, math.ceil((max(epochs) - min(epochs)) / 8))
        ax.set_xticks(range(min(epochs), max(epochs) + 1, x_step))

        # Y-axis ticks
        y_min, y_max = min(fid_scores), max(fid_scores)
        y_step = (y_max - y_min) / 6 if y_max != y_min else 1
        ax.set_yticks([
            round(y_min + i * y_step, 2)
            for i in range(7)
        ])

    # Hide unused subplots
    for ax in axes[len(plots):]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(
        save_path,
        format="png",
        dpi=200,
        bbox_inches="tight"
    )
    
    plt.show()
    plt.close(fig)


def show_images(gen_ema: nn.Module, rows: int = 3, cols: int = 3, save_path: str = 'final_images.png'):
    images = None
    
    for i in range(rows*cols):
        image = gen_ema.generate_image(device)
        if images is not None:
            images = torch.cat([images, image])
        else:
            images = image
    
    grid = make_grid(
        images.detach().float().cpu(),
        nrow=cols,
        padding=2,
        normalize=True
    )

    save_image(
        images.detach().float().cpu(),
        save_path,
        format="PNG",
        nrow=cols,
        padding=2,
        normalize=True
    )
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.show()


# ====================================== Architecture ======================================


class EqualizedLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, gain: float = math.sqrt(2), lrmul: float = 1.0, bias: bool = True
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lrmul)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.bias_mul = lrmul if bias else None
        fan_in = in_features
        self.scale = (gain / math.sqrt(fan_in)) * lrmul  # applied at forward time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias * self.bias_mul if self.bias is not None else None
        return F.linear(x, self.weight * self.scale, bias=bias)


class EqualizedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        gain: float = math.sqrt(2),
        lrmul: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) / lrmul
        )
        self.out_channels = out_channels
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.bias_mul = lrmul if bias else None

        fan_in = in_channels * kernel_size * kernel_size
        self.scale = (gain / math.sqrt(fan_in)) * lrmul
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = (
            (self.bias * self.bias_mul).type(x.dtype) if self.bias is not None else None
        )
        return F.conv2d(
            x,
            self.weight * self.scale,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
        )


class Blur(nn.Module):
    """Fixed [1,2,1]⊗[1,2,1] low-pass filter, applied per channel."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
        k = k / k.sum()
        self.register_buffer("k", k[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.k, padding=1, groups=self.channels)


class UpscaleConv2d(nn.Module):
    """Official StyleGAN upscale+conv. Fused (transposed conv) when big enough,
    else plain upsample+conv. Matches networks_progan.upscale2d_conv2d."""

    def __init__(self, in_c: int, out_c: int, kernel: int = 3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn((in_c, out_c, kernel, kernel)))
        self.scale = math.sqrt(2 / (in_c * kernel * kernel))  # equalized LR
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = min(x.shape[2:]) * 2 >= 128 

        if not fused:
            # --- separate path: nearest upsample, then normal conv ---
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            w = self.weight.permute(1, 0, 2, 3) * self.scale  # [out,in,k,k]
            return F.conv2d(x, w, padding=self.kernel // 2)

        # --- fused path: pad + sum 4 shifts of the weight, one transposed conv ---
        w = self.weight * self.scale  # [in,out,k,k]
        w = F.pad(w, (1, 1, 1, 1))  # grow kernel by 1 each side
        w = (
            w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
        )  # sum 4 shifted copies (NO /4)
        return F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)


class MappingNetwork(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.pixel_norm = PixelNorm()
        mapping_network = []

        # Creating Module List for Mapping Network linear + leaky relu
        for layer_index in range(config.num_mapping_layers):
            mapping_network.append(
                EqualizedLinear(config.N_DIM, config.N_DIM, lrmul=0.01)
            )
            mapping_network.append(nn.LeakyReLU(config.leaky_relu_negative_slope))

        self.mapping_network = nn.Sequential(*mapping_network)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixel_norm(x)
        x = self.mapping_network(x)
        return x


class Block(nn.Module):
    def __init__(self, config: Config, block_stage: int) -> None:
        super().__init__()
        self.config = config
        self.block_stage = block_stage

        previous_C = nf(config, block_stage - 1)
        C = nf(config, block_stage)

        self.activation = nn.LeakyReLU(config.leaky_relu_negative_slope)
        self.instance_norm = nn.InstanceNorm2d(C, eps=1e-8)

        if block_stage == 0:
            self.constant = nn.Parameter(torch.ones(1, C, config.H, config.W))
        else:
            self.upsample2d_conv2d = nn.Sequential(
                UpscaleConv2d(previous_C, C), Blur(C)
            )

        # A across all layers
        self.A1 = EqualizedLinear(config.N_DIM, C * 2, gain=1)
        self.A2 = EqualizedLinear(config.N_DIM, C * 2, gain=1)

        # (Noise * learned_scale) + learned_bias
        self.noise_block_1 = Noise(C)
        self.noise_block_2 = Noise(C)
        self.layer_bias_1 = nn.Parameter(torch.zeros(1, C, 1, 1))  # (1, N_DIM, 1, 1)
        self.layer_bias_2 = nn.Parameter(torch.zeros(1, C, 1, 1))  # (1, N_DIM, 1, 1)

        self.conv = EqualizedConv2d(C, C, kernel_size=3, padding=1, bias=False)

    def forward(
        self,
        styles: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        x: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        if isinstance(styles, (list, tuple)):
            s1, s2 = styles
        else:
            s1 = s2 = styles

        B = s1.shape[0]

        # ======================= Layer 1 =======================

        # if initial block then use constant else use x
        if self.block_stage == 0:
            x = self.constant.expand(B, -1, -1, -1)
        else:
            # upsample + conv + blur
            x = self.upsample2d_conv2d(
                x
            )  # B, C, H, W -> B, C/2 (if<fmap_max), 2*H, 2*W

        A1 = self.A1(s1)  # (B, C*2)
        scale_1, bias_1 = A1.unsqueeze(2).unsqueeze(2).chunk(2, dim=1) # (B, N_DIM, 1, 1) * 2

        learned_noise_1 = self.noise_block_1(x) + self.layer_bias_1
        normalization_input_1 = self.activation(x + learned_noise_1)
        x = (scale_1 + 1) * self.instance_norm(normalization_input_1) + bias_1

        # ======================= Layer 2 =======================

        A2 = self.A2(s2)  # (B, N_DIM*2)
        scale_2, bias_2 = A2.unsqueeze(2).unsqueeze(2).chunk(2, dim=1) # (B, N_DIM, 1, 1) * 2

        x = self.conv(x)
        learned_noise_2 = self.noise_block_2(x)
        learned_noise_2 = learned_noise_2 + self.layer_bias_2
        normalization_input_2 = self.activation(x + learned_noise_2)
        output = (scale_2 + 1) * self.instance_norm(normalization_input_2) + bias_2
        return output


class Noise(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.learned_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]
        noise = torch.randn(
            B, 1, height, width, device=x.device, dtype=x.dtype, generator=random_gen
        )  # (B, 1, H, W)
        return noise * self.learned_scale  # (B, 1, H, W) * (1, C, 1, 1) = (B, C, H, W)


class PixelNorm(nn.Module):
    """Normalize each sample's feature vector to unit length, per pixel."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C]  (mapping net)  or  [B, C, H, W]  (conv feature map)
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class ToRGB(nn.Module):
    def __init__(self, in_c: int) -> None:
        super().__init__()
        self.in_c = in_c
        self.weight = nn.Parameter(torch.randn(in_c, 3, 1, 1))
        self.bias = nn.Parameter(torch.zeros(3))
        self.scale = math.sqrt(1 / in_c)  # equalized LR, gain=1 (not sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.scale
        return F.conv2d(x, w.permute(1, 0, 2, 3), bias=self.bias)


class Generator(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.mapping_network = MappingNetwork(config)
        self.blocks = nn.ModuleList(
            [Block(config, block_stage=idx) for idx in range(config.n_blocks)]
        )
        self.to_rgb_layers = nn.ModuleList(
            [ToRGB(nf(config, idx)) for idx in range(config.n_blocks)]
        )
        self.config = config
        self.register_buffer("w_avg", torch.zeros(config.N_DIM))
        self.w_avg_beta = 0.995

    def fade_in(self, upscaled_img: torch.Tensor, x_img: torch.Tensor, alpha: float = 1) -> torch.Tensor:
        return x_img * alpha + (1 - alpha) * upscaled_img

    def progressive_step(self, old: torch.Tensor, new: torch.Tensor, step: int, alpha: float) -> torch.Tensor:
        old_img = self.to_rgb_layers[step - 2](old)
        old_upscaled = F.interpolate(old_img, scale_factor=2, mode="nearest")

        new = self.to_rgb_layers[step - 1](new)
        new = self.fade_in(old_upscaled, new, alpha)
        return new

    def forward(
        self,
        x: torch.Tensor | None = None,
        w: torch.Tensor | None = None,
        alpha: float = 1.0,
        step: int = 1,
        inference_mode: bool = False,
        psi: float | None = 0.5,
        truncation_cutoff: int | None = 8,
        style_mixing_prob: float = 0.9,
    ) -> torch.Tensor:
        assert x is not None or w is not None, "Provide either `x` (latent z) or `w` (precomputed style)"
    
        # Mapping Network (skip if w is given directly)
        if w is not None:
            w1 = w
            batch_size, ref_device, ref_dtype = w1.shape[0], w1.device, w1.dtype
        else:
            w1 = self.mapping_network(x)  # (B, N_DIM)
            batch_size, ref_device, ref_dtype = x.shape[0], x.device, x.dtype
    
            if self.training and not inference_mode:
                with torch.no_grad():
                    batch_avg = w1.detach().mean(dim=0).float()
                    if (self.w_avg == 0).all():
                        self.w_avg.copy_(batch_avg)
                    else:
                        self.w_avg.lerp_(batch_avg, 1.0 - self.w_avg_beta)
    
        cur_layers = step * 2
        if (random.random() < style_mixing_prob) and not inference_mode and self.training:
            z2 = torch.randn(
                (batch_size, self.config.N_DIM),
                device=ref_device,
                dtype=ref_dtype,
                generator=random_gen
            )
            w2 = self.mapping_network(z2)
            crossover = random.randint(1, cur_layers - 1)
            layer_styles = [w1 if i < crossover else w2 for i in range(cur_layers)]
        else:
            layer_styles = [w1] * cur_layers
    
        if inference_mode and psi is not None and psi != 1.0:
            cutoff = truncation_cutoff if truncation_cutoff is not None else cur_layers
            truncated_styles = []
            for idx, w_style in enumerate(layer_styles):
                if idx < cutoff:
                    w_t = self.w_avg.unsqueeze(0) + psi * (w_style - self.w_avg.unsqueeze(0))
                    truncated_styles.append(w_t)
                else:
                    truncated_styles.append(w_style)
            layer_styles = truncated_styles
    
        # Group 2 styles per resolution block: (s1 for A1, s2 for A2)
        block_styles = [
            (layer_styles[2 * i], layer_styles[2 * i + 1]) for i in range(step)
        ]
    
        x_out, prev = None, None
        for i in range(step):
            if i == 0:  # If initial Layer
                x_out = self.blocks[i](block_styles[i])
            else:
                prev = x_out
                x_out = self.blocks[i](block_styles[i], x_out)
    
        x_out = (
            self.progressive_step(prev, x_out, step, alpha)
            if alpha < 1 and step != 1
            else self.to_rgb_layers[step - 1](x_out)
        )
        return x_out
    
    
    @torch.no_grad()
    def generate_image(
        self,
        device: torch.device | str,
        *,
        w: torch.Tensor | None = None,
        batch_size: int = 1,
        step: int = 8,
        alpha: float = 1.0,
        psi: float = 0.5,
        truncation_cutoff: int | None = 8,
    ) -> torch.Tensor:
        """
        Convenience wrapper for inference-time sampling (EMA generator, assumed
        already in eval mode — no train/eval toggling here).
    
        If `w` is provided, it's used directly as the style vector and the
        mapping network is skipped. Otherwise a random z ~ N(0, I) is sampled
        and run through the mapping network for you.
        """
        if w is None:
            x = torch.randn(
                (batch_size, self.config.N_DIM),
                device=device,
            )
            w = self.mapping_network(x)
        else:
            w = w.to(device)
    
        return self.forward(
            w=w,
            alpha=alpha,
            step=step,
            inference_mode=True,
            psi=psi,
            truncation_cutoff=truncation_cutoff,
        )


class MinibatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4, eps: float = 1e-8) -> None:
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        G = min(self.group_size, N)
        if N % G != 0:
            G = N  # fallback if not divisible
        M = N // G
        y = x.view(M, G, C, H, W)  # [M, G, C, H, W]
        y = y - y.mean(dim=1, keepdim=True)  # center within group
        y = (y**2).mean(dim=1)  # variance [M, C, H, W]
        y = torch.sqrt(y + self.eps)  # stddev [M, C, H, W]
        y = y.mean(dim=[1, 2, 3], keepdim=True)  # average over C,H,W -> [M, 1, 1, 1]
        y = y.repeat_interleave(G, dim=0)  # broadcast back across samples -> [N, 1, 1, 1]
        y = y.expand(-1, -1, H, W)  # [N, 1, H, W]
        return torch.cat([x, y], dim=1)  # concat as extra channel


class DownsamplingBlock(nn.Module):
    def __init__(self, config: Config, block_stage: int) -> None:
        super().__init__()
        in_channels = nf(config, config.n_blocks - block_stage - 1)
        out_channels = nf(config, config.n_blocks - block_stage - 2)

        self.conv_1 = EqualizedConv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.blur = Blur(in_channels)
        self.conv_2 = EqualizedConv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.activation = nn.LeakyReLU(config.leaky_relu_negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.blur(x)
        x = self.conv_2(x)
        x = self.activation(x)
        return x


class FinalBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        # stddev layer
        self.stddev_layer = MinibatchStdDev(group_size=4)

        self.conv = EqualizedConv2d(
            config.N_DIM + 1, config.N_DIM, kernel_size=3, padding=1, bias=True
        )
        self.activation = nn.LeakyReLU(config.leaky_relu_negative_slope)

        flattened_size = config.N_DIM * 4 * 4
        self.dense_1 = EqualizedLinear(flattened_size, config.N_DIM)
        self.dense_2 = EqualizedLinear(config.N_DIM, 1, gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stddev_layer(x)
        x = self.conv(x)
        x = self.activation(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        # Input (B, 3, 512, 512)
        self.config = config
        self.from_rgb_layers = nn.ModuleList(
            [
                EqualizedConv2d(3, nf(config, config.n_blocks - 1 - i), kernel_size=1)
                for i in range(config.n_blocks)
            ]
        )
        self.activation = nn.LeakyReLU(negative_slope=config.leaky_relu_negative_slope)
        self.blocks = nn.ModuleList(
            [DownsamplingBlock(config, idx) for idx in range(config.n_blocks - 1)]
            + [FinalBlock(config)]
        )

    def fade_in(self, old: torch.Tensor, new: torch.Tensor, alpha: float = 1) -> torch.Tensor:
        return new * alpha + (1 - alpha) * old

    def downscale2d(self, x: torch.Tensor, factor: int = 2) -> torch.Tensor:
        """Average-pool downsample by an integer factor."""
        if factor == 1:
            return x
        return F.avg_pool2d(x, kernel_size=factor, stride=factor)

    def progressive_step(self, old: torch.Tensor, new: torch.Tensor, block_idx: int, alpha: float) -> torch.Tensor:
        downscaled = self.downscale2d(old)
        old_feat = self.activation(self.from_rgb_layers[block_idx + 1](downscaled))
        x = self.fade_in(old_feat, new, alpha)
        return x

    def forward(self, x: torch.Tensor, alpha: float, step: int) -> torch.Tensor:
        old = x
        x = self.activation(self.from_rgb_layers[self.config.n_blocks - step](x))

        for i in range(self.config.n_blocks - step, self.config.n_blocks):
            x = self.blocks[i](x)
            progressive_condition = (
                alpha < 1
                and i != self.config.n_blocks - 1
                and i == self.config.n_blocks - step
            )
            x = self.progressive_step(old, x, i, alpha) if progressive_condition else x

        return x

# ======================================== DATASET CLASS ========================================


class SyntheticDatasetBatched(Dataset):
    def __init__(
        self,
        folder_path: str,
        shards_path: str = "/kaggle/tmp/image_shards",
        batch_size: int = 512,
        num_workers: int = 2,
        batch_shuffle: bool = True,
        size: int | None = None,
        shard_workers: int = 16,
    ) -> None:
        self.image_paths = sorted(
            [os.path.join(folder_path, path) for path in os.listdir(folder_path)]
        )[:size]
        self.shards_path = shards_path
        self.size = len(self.image_paths)

        save_shards(
            self.image_paths,
            shards_path,
            shard_size=batch_size,
            num_workers=shard_workers,
            local_rank=local_rank,
            world_size=world_size,
        )

        # Synchronize so all ranks finish saving before any rank lists the directory
        dist_barrier(ddp)

        self.shard_bounds = []
        for shard in sorted(os.listdir(shards_path)):
            if not shard.endswith(".pt") or not shard.startswith("shard_from_"):
                continue
            start_index, end_index = map(
                int, shard.removeprefix("shard_from_").removesuffix(".pt").split("_to_")
            )
            # Only include shards within this dataset's size
            if end_index < self.size:
                self.shard_bounds.append((start_index, end_index + 1))

        self.batch_shuffle = batch_shuffle
        self.shard_bounds.sort(key=lambda x: x[0])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = decode_image(self.image_paths[idx])
        return image

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, ...]:
        """Fetch multiple items at once — enables vectorized generation.

        Instead of N individual __getitem__ calls (each with its own
        overhead), this generates the entire batch in one shot using
        vectorized tensor operations.
        """
        # will be sorted by default
        start_index = indices[0]
        end_index = indices[-1]

        log(f"Data Sample {start_index} - {end_index}, GPU: {local_rank}", verbose=VERBOSE)

        file_name = f"shard_from_{start_index}_to_{end_index}.pt"
        chunk_path = os.path.join(shards_path, file_name)
        chunk = torch.load(chunk_path)
        if self.batch_shuffle:
            return chunk[torch.randperm(len(chunk))].unbind(0)
        return chunk.unbind(0)


class ShardBatchSampler(Sampler):
    def __init__(self, shard_bounds: list[tuple[int, int]], batch_size: int, rank: int, world_size: int, drop_last: bool = True, shuffle: bool = False) -> None:
        self.shard_bounds = shard_bounds
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.epoch = 0
        self.shuffle = shuffle

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.g = torch.Generator()
        self.g.manual_seed(self.epoch)

    def __iter__(self) -> Iterator[list[int]]:
        my_shards = self.shard_bounds[self.rank :: self.world_size]
        
        if self.shuffle:
            my_shards = torch.tensor(my_shards)
            my_shards = my_shards[torch.randperm(len(my_shards), generator=self.g)].tolist()
            
        for start, end in my_shards:
            idxs = list(range(start, end))
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self) -> int:
        my_shards = self.shard_bounds[self.rank :: self.world_size]
        total = 0
        for start, end in my_shards:
            n = end - start
            total += (
                n // self.batch_size if self.drop_last else -(-n // self.batch_size)
            )
        return total


# ====================================== Train Parameters ======================================


# Dataset Params
folder_path = "/kaggle/input/datasets/arnaud58/flickrfaceshq-dataset-ffhq"
prefetch_factor = 2
num_workers = 2
fid_size = 10240

# Progressive Training
fade_map = {(i + 1): (300 * 1000) + (28036 * (i + 1)) for i in range(config.n_blocks)} # progressively grows from 300k - 500k
pixel_batches = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}

stage = 1
n_images = 0
previous_stage = None


# Optimizer Params
lr = 0.001
betas = (0.00, 0.99)
eps = 1e-8

# Training Params
start_epoch = 0
epochs = 1000
dtype = torch.float16
fid_losses = []
best_fid = float("inf")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

# Other Params
image_save_path = "processed_images"
fixed_image_save_path = 'processed_fixed_images'
fixed_video_path = "output_fixed.mp4"
shards_path = "/kaggle/tmp/image_shards"
shard_workers = 16 # rest will be in queue
epoch_summaries = []
save_image_after = 10
print_after = 10
fixed_z = torch.randn(
    4, config.N_DIM,
    generator=torch.Generator().manual_seed(1234)
).to(device=device, dtype=dtype)


# ==================================== DATASET AND LOADERS ====================================


dataset = SyntheticDatasetBatched(
    folder_path, shards_path, config.batch_size, num_workers
)
fid_dataset = SyntheticDatasetBatched(
    folder_path, shards_path, config.batch_size, num_workers, size=fid_size
)


shard_bounds = balance_shards(dataset.shard_bounds, config.batch_size, world_size)
fid_shard_bounds = balance_shards(
    fid_dataset.shard_bounds, config.batch_size, world_size
)

sampler = ShardBatchSampler(shard_bounds, config.batch_size, local_rank, world_size, shuffle=True)
fid_sampler = ShardBatchSampler(
    fid_shard_bounds, config.batch_size, local_rank, world_size
)

data_loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    prefetch_factor=prefetch_factor,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

fid_loader = DataLoader(
    fid_dataset,
    prefetch_factor=1,
    num_workers=num_workers,
    batch_sampler=fid_sampler,
    pin_memory=True,
    persistent_workers=True,
)

gen = Generator(config).to(device)
disc = Discriminator(config).to(device)
gen_ema = None

# ======================================== Visualize ===========================================
if visualization and MASTER_PROC:
    log('Generator', header=True, end='\n\n')

    visualize_network(gen)
    log(gen)

    log('Generator', header=True, end='\n\n')
    visualize_network(disc)
    log(disc)
    sys.exit()


G_adam = optim.Adam(gen.parameters(), lr=lr, betas=betas, eps=eps)
D_adam = optim.Adam(disc.parameters(), lr=lr, betas=betas, eps=eps)

# Overrides values if training exist
if os.path.exists(checkpoint_load_path):
    log("Loading Checkpoint...", master_proc=MASTER_PROC)
    checkpoint = torch.load(checkpoint_load_path, map_location=device)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    disc.load_state_dict(checkpoint["disc_state_dict"])

    # Load optimizer state
    G_adam.load_state_dict(checkpoint["optimizer_gen_state_dict"])
    D_adam.load_state_dict(checkpoint["optimizer_disc_state_dict"])

    n_images = checkpoint["n_images"]
    fid_losses = checkpoint["fid_losses"]

    start_epoch = checkpoint["epoch"] + 1
    epoch_summaries = checkpoint['epoch_summaries']
    stage = checkpoint["stage"]
    
    if 'gen_ema_state_dict' in checkpoint: 
        log("Loading Gen Ema...", master_proc=MASTER_PROC)
        gen_ema = Generator(config).to(device)
        gen_ema.load_state_dict(checkpoint['gen_ema_state_dict'])
    
    del checkpoint
    clear_ram_gpu()
    log(f"Checkpoint Exist continuing from Epoch {start_epoch}, Stage: {stage}......", master_proc=MASTER_PROC)
else:
    log("Checkpoint Doesnt Exist Training with initial setup", master_proc=MASTER_PROC)
    
softplus = nn.Softplus()

if ddp:
    disc = DDP(disc, device_ids=[local_rank], find_unused_parameters=True).to(device)
    raw_disc = disc.module

    gen = DDP(gen, device_ids=[local_rank], find_unused_parameters=True).to(device)
    raw_gen = gen.module
else:
    raw_disc = disc
    raw_gen = gen

gen_ema = get_gen_ema(gen_ema, raw_gen)

G_scaler = GradScaler(device)
D_scaler = GradScaler(device)

log('TRAINING INFO', header=True, master_proc=MASTER_PROC)
log(f"Device: {device}, Dtype: {dtype}, DDP: {ddp}, World Size: {world_size}", master_proc=MASTER_PROC)
log(f"Length of Dataset: {len(dataset)}", master_proc=MASTER_PROC)
log(f"Image Shape: {tuple(dataset[0].shape)}", master_proc=MASTER_PROC)
log(f"Stabilization/Fade-In with {fade_map} images", master_proc=MASTER_PROC, end='\n\n')
log(f"Pixel Batches {pixel_batches} images", master_proc=MASTER_PROC, end='\n\n')
log("MODEL TRAINING STARTED", header=True, master_proc=MASTER_PROC)
# device type, dtype, device, etc params

# =================================== Model Training =======================================

for epoch in range(start_epoch, epochs):
    epoch_start_time = time()
    
    log(f"EPOCH: {epoch}", header=True, master_proc=MASTER_PROC)
    
    if ddp:
        sampler.set_epoch(epoch)
    

    for idx, data in enumerate(data_loader):
        stage, mini_batch, alpha = get_batch_params(n_images, stage, fade_map, config)
        iterations = config.batch_size // mini_batch
        pixels = get_pixel(stage)
        data_percentage = ((idx+1)*100) / len(data_loader)

        # ================================ Discriminator ================================
        log(f"Starting Discriminator Iteration for GPU: {local_rank}", verbose=VERBOSE)
        
        for iteration in range(iterations):
            D_adam.zero_grad()
            start_index = iteration * mini_batch
            end_index = start_index + mini_batch
            # is_last = (iteration == iterations - 1)
            is_last = True # Always do backprop no accum
            
            latent_vector = torch.randn(
                (mini_batch, config.N_DIM),
                device=device,
                generator=random_gen,
                dtype=dtype,
            )

            disc_ctx = (
                contextlib.nullcontext() if (is_last or (not ddp)) else disc.no_sync()
            )

            with disc_ctx:
                with autocast(device_type=device_type, dtype=dtype):
                    with torch.no_grad():
                        fake_images = raw_gen(latent_vector, alpha=alpha, step=stage)
                    real_images = data[start_index:end_index].to(
                        device, dtype, non_blocking=True
                    )
                    real_images = get_real_at_stage(
                        raw_disc, real_images, config, stage
                    ).requires_grad_(True)

                    fake_images_score = disc(fake_images, alpha=alpha, step=stage)
                    real_images_score = disc(real_images, alpha=alpha, step=stage)

                    log(f"Disc Mini Batch {mini_batch*(iteration+1)}/{len(data)} Batch: {idx} GPU: {local_rank}", verbose=VERBOSE)
                    
                    # Non-saturating logistic loss
                    loss = (
                        softplus(-real_images_score) + softplus(fake_images_score)
                    ).mean()

                    # R1 gradient penalty on real images (gamma=10.0)
                    r1_loss = (
                        r_penalty(real_images_score.float(), real_images, gamma=10.0)
                    )
                    total_d_loss = loss + r1_loss
                    D_scaler.scale(total_d_loss).backward()

            lr_update_scheduler(D_adam, lr, stage, previous_stage)
    
            D_scaler.step(D_adam)
            D_scaler.update()
            log(f"Discriminator Iteration Finished for GPU: {local_rank}", verbose=VERBOSE)

        # clear_ram_gpu()
        
            # ================================ Generator ================================
            log(f"Starting Generator Iteration for GPU: {local_rank}", verbose=VERBOSE)
        
            # Freeze discriminator weights
            raw_disc.requires_grad_(False)
            G_adam.zero_grad()
            # is_last = iteration == iterations - 1
            is_last = True

            gen_ctx = (
                contextlib.nullcontext() if (is_last or (not ddp)) else gen.no_sync()
            )

            latent_vector = torch.randn(
                (mini_batch, config.N_DIM),
                device=device,
                generator=random_gen,
                dtype=dtype
            )

            log(f"Gen Mini Batch {mini_batch*(iteration+1)}/{len(data)} Batch: {idx} GPU: {local_rank}", verbose=VERBOSE)

            with gen_ctx:
                with autocast(device_type=device_type, dtype=dtype):
                    # Forward pass on Generator
                    fake_images = gen(latent_vector, alpha=alpha, step=stage)
                    fake_images_score = raw_disc(fake_images, alpha=alpha, step=stage)

                    loss = softplus(-fake_images_score).mean()
                    total_g_loss = loss
                    G_scaler.scale(loss).backward()

            # Unfreeze discriminator weights
            raw_disc.requires_grad_(True)

            previous_stage = lr_update_scheduler(G_adam, lr, stage, previous_stage)
            G_scaler.step(G_adam)
            G_scaler.update()


            # final update n_images
            n_images += (mini_batch * world_size)

            update_ema(gen_ema, gen, mini_batch * world_size)
            
            log(f"Generator Iteration Finished for GPU: {local_rank}", verbose=VERBOSE)

        if ((idx+1) % print_after) == 0:
            log(
                f'Pixels: {pixels}, Mini Batch: {mini_batch}, Alpha: {alpha}, G_loss: {total_g_loss.item():.3f}, D_loss: {total_d_loss.item():.3f}, Epoch Percentage: {data_percentage}',
                master_proc=MASTER_PROC
               )

        if ((idx % save_image_after) == 0) and MASTER_PROC:
            log(f"Saving and Writing video using GPU: {local_rank}", verbose=VERBOSE)

            save_images(
                gen_ema, n_images, stage, image_save_path, device=device, dtype=dtype, 
            )

            if stage == config.n_blocks:
                save_images(
                    gen_ema,
                    n_images,
                    stage,
                    fixed_image_save_path,
                    device=device,
                    dtype=dtype,
                    latent_z=fixed_z
                )


    
        
    clear_ram_gpu()
    
    if MASTER_PROC:
        create_video(image_save_path, config.output_pixel, fps=5)
        if stage == config.n_blocks:
            create_video(fixed_image_save_path, config.output_pixel, fps=5, output_video=fixed_video_path)
        
        
    models = {"gen": raw_gen, "disc": raw_disc, 'gen_ema': gen_ema}
    optimizers = {"gen": G_adam, "disc": D_adam}

    fid_score = calculate_fid_authentic(fid_loader, gen_ema, raw_disc, stage)

    if MASTER_PROC:
        best_fid, epoch_summaries = post_epoch_process(
            models,
            optimizers,
            n_images,
            stage,
            best_fid,
            start_time,
            epoch,
            fid_score,
            epoch_start_time,
            epoch_summaries
        )
    
        
    if (time() - start_time) >= time_limit:
        log("Time Limit Reached Exiting", header=True, master_proc=MASTER_PROC)
        break


# ======================================= Final Summary =======================================
log(f"Trained on {epoch} number of Epochs", master_proc=MASTER_PROC)

if MASTER_PROC:
    log(f"Trained on {epoch} number of Epochs", master_proc=MASTER_PROC)
    plot_on_pixels(epoch_summaries)
    show_images(gen_ema, rows=3, cols=4)


if ddp:
    destroy_process_group()