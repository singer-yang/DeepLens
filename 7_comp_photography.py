"""An example for computational photography with realistic lens and sensor model.

# The code uses distributed data parallel (DDP) scheme. To run experiments on multiple GPUs, use the following command:
# torchrun --nproc_per_node=4 7_comp_photography.py

Reference:
    [1] Brooks, Tim, et al. "Unprocessing images for learned raw denoising." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""

import logging
import os
import random
import shutil
import string
from datetime import datetime

import lpips
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image
from tqdm import tqdm

from deeplens import Camera
from deeplens.network import NAFNet, PerceptualLoss, PhotographicDataset
from deeplens.utils import batch_psnr, batch_ssim, set_logger, set_seed


def config():
    """Load and prepare configuration."""
    # Load config files
    with open("configs/7_comp_photography.yml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Set up result directory
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(4))
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = f"{current_time}-Comp-Photography-{random_string}"

    result_dir = f"./results/{exp_name}"
    os.makedirs(result_dir, exist_ok=True)
    args["result_dir"] = result_dir

    # Set random seed
    if args["seed"] is None:
        args["seed"] = random.randint(0, 1000)
    set_seed(args["seed"])

    # Configure logging
    set_logger(result_dir)
    logging.info(f"Experiment: {args['exp_name']}")
    if not args["is_debug"]:
        raise Exception("Add your wandb logging config here.")

    # Configure device
    if torch.cuda.is_available():
        args["device"] = torch.device("cuda")
        args["num_gpus"] = torch.cuda.device_count()
        args["ddp"] = args["num_gpus"] > 1
        logging.info(f"Using {args['num_gpus']} {torch.cuda.get_device_name(0)} GPU(s)")
    else:
        args["device"] = torch.device("cpu")
        logging.info("Using CPU")

    # Save config and code
    with open(f"{result_dir}/config.yml", "w") as f:
        yaml.dump(args, f)
    shutil.copy("7_comp_photography.py", f"{result_dir}/7_comp_photography.py")

    return args


def setup():
    """
    Initialize the distributed environment using environment variables set by torchrun.
    """
    # When using torchrun, these environment variables are automatically set
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()


class Trainer:
    """Class for training models with DDP."""

    def __init__(self, local_rank, world_size, args):
        """Initialize the trainer.

        Args:
            local_rank: Local GPU rank
            world_size: Total number of GPUs
            args: Dictionary with training configuration
            dataset_args: Dictionary with dataset configuration
        """
        self.local_rank = local_rank
        self.rank = int(os.environ["RANK"])
        self.world_size = world_size
        self.args = args
        self.device = torch.device(f"cuda:{local_rank}")

        # Initialize model, renderer, and data
        self._init_data(
            train_set_config=args["train_set"], eval_set_config=args["eval_set"]
        )
        self._init_model(net_args=args["network"], train_args=args["train"])
        self._init_camera(camera_args=args["camera"])

    def _init_camera(self, camera_args):
        """Initialize the camera."""
        self.camera = Camera(
            lens_file=camera_args["lens_file"],
            sensor_size=camera_args["sensor_size"],
            sensor_res=camera_args["sensor_res"],
            device=self.device,
        )

    def _init_model(self, net_args, train_args):
        """Initialize the model and optimizer."""
        # Create model
        self.model = NAFNet(
            in_chan=net_args["in_chan"],
            out_chan=net_args["out_chan"],
            width=net_args["width"],
            middle_blk_num=net_args["middle_blk_num"],
            enc_blk_nums=net_args["enc_blk_nums"],
            dec_blk_nums=net_args["dec_blk_nums"],
        )
        self.model.to(self.device)

        # Load checkpoint if provided
        if net_args.get("ckpt_path"):
            state_dict = torch.load(net_args["ckpt_path"], map_location=self.device)
            try:
                self.model.load_state_dict(state_dict["model"])
            except:
                self.model.load_state_dict(state_dict)

        # Wrap with DDP
        self.ddp_model = DDP(self.model, device_ids=[self.local_rank])

        # Create optimizer
        self.optimizer = optim.AdamW(
            self.ddp_model.parameters(), lr=float(train_args["lr"])
        )

        # Create scheduler
        total_steps = train_args["epochs"] * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-7,
        )

        # Create loss
        self.lpips_loss = PerceptualLoss(device=self.device)

        # Create metrics
        self.lpips_metric = lpips.LPIPS(net="alex").to(self.device)

    def _init_data(self, train_set_config, eval_set_config):
        """Initialize data loaders."""
        # Create datasets
        train_dataset = PhotographicDataset(
            train_set_config["dataset"],
            output_type=train_set_config["output_type"],
            img_res=train_set_config["res"],
            is_train=True,
        )
        val_dataset = PhotographicDataset(
            eval_set_config["dataset"],
            output_type=eval_set_config["output_type"],
            img_res=eval_set_config["res"],
            is_train=False,
        )

        # Create distributed samplers
        self.train_sampler = DistributedSampler(
            train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )

        val_sampler = DistributedSampler(
            val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False
        )

        # Create data loaders
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=train_set_config["batch_size"],
            sampler=self.train_sampler,
            num_workers=train_set_config["num_workers"],
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=eval_set_config["batch_size"],
            sampler=val_sampler,
            num_workers=eval_set_config["num_workers"],
            pin_memory=True,
        )

    def compute_loss(self, inputs, targets):
        """Compute loss between model outputs and targets.

        Args:
            inputs: Input blurred images [B, C, H, W]
            targets: Target clean images [B, C, H, W]

        Returns:
            loss: The computed loss value
            loss_dict: Dictionary with loss components for logging
        """
        # Forward pass
        outputs = self.ddp_model(inputs)
        outputs = outputs.clamp(0, 1)

        # Convert to RGB for loss computation
        sensor = self.camera.sensor
        sensor.sample_augmentation()
        outputs_rgb = sensor.process2rgb(outputs)
        targets_rgb = sensor.process2rgb(targets)

        # RGB Loss
        l1_loss = F.l1_loss(outputs_rgb, targets_rgb)
        perceptual_loss = self.lpips_loss(outputs_rgb, targets_rgb)
        rgb_loss = l1_loss + 0.5 * perceptual_loss

        # RAW Loss
        raw_loss = F.l1_loss(outputs, targets)

        # Total loss
        loss = rgb_loss + raw_loss
        loss_dict = {
            "rgb_loss": rgb_loss.item(),
            "raw_loss": raw_loss.item(),
            "total_loss": loss.item(),
        }
        return loss, loss_dict

    def compute_metrics(self, outputs, targets=None):
        """Compute metrics between model outputs and targets."""
        # Convert to RGB
        sensor = self.camera.sensor
        sensor.reset_augmentation()
        outputs_rgb = sensor.process2rgb(outputs)
        targets_rgb = sensor.process2rgb(targets)

        # Calculate metrics
        lpips_score = self.lpips_metric(outputs_rgb * 2 - 1, targets_rgb * 2 - 1)
        psnr_score = batch_psnr(outputs_rgb, targets_rgb)
        ssim_score = batch_ssim(outputs_rgb, targets_rgb)

        metrics = {
            "psnr": psnr_score,
            "ssim": ssim_score,
            "lpips": lpips_score,
        }
        return metrics

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.train_sampler.set_epoch(epoch)
        self.ddp_model.train()

        # Training loop
        for i, data_dict in enumerate(tqdm(self.train_loader, disable=self.rank != 0)):
            # Image simulation, training data synthesis
            inputs, targets = self.camera.render(data_dict)

            # Compute loss
            loss, loss_dict = self.compute_loss(inputs, targets)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            # Log progress
            if (
                i % self.args["train"]["log_every_n_steps"]
                == self.args["train"]["log_every_n_steps"] - 1
                and self.rank == 0
            ):
                print(
                    f"Epoch: {epoch + 1}/{self.args['train']['epochs']}, "
                    f"Batch: {i + 1}/{len(self.train_loader)}, "
                    f"Loss: {loss_dict['total_loss']:.4f}"
                )

                # Save sample images
                with torch.no_grad():
                    outputs = self.ddp_model(inputs)

                    sensor = self.camera.sensor
                    sensor.reset_augmentation()
                    inputs_rgb = sensor.process2rgb(inputs[:, :4, :, :])
                    outputs_rgb = sensor.process2rgb(outputs.detach()[:, :4, :, :])
                    targets_rgb = sensor.process2rgb(targets[:, :4, :, :])
                    save_image(
                        torch.cat([inputs_rgb, outputs_rgb, targets_rgb], dim=2),
                        f"{self.args['result_dir']}/train_epoch{epoch}_batch{i}.png",
                    )

        return loss_dict["total_loss"] / len(self.train_loader)

    def validate(self, epoch):
        """Run validation."""
        # Set model to eval mode
        self.ddp_model.eval()

        # Initialize metrics
        val_psnr = 0.0
        val_ssim = 0.0
        val_lpips = 0.0
        val_samples = 0

        # Validation loop
        with torch.no_grad():
            for i, data_dict in enumerate(
                tqdm(self.val_loader, desc="Validating", disable=self.rank != 0)
            ):
                # Apply blur to create inputs using camera
                inputs, targets = self.camera.render(data_dict)

                # Forward pass
                outputs = self.ddp_model(inputs)
                outputs = outputs.clamp(0, 1)

                # Compute metrics
                metrics = self.compute_metrics(outputs, targets)
                val_psnr += metrics["psnr"] * inputs.size(0)
                val_ssim += metrics["ssim"] * inputs.size(0)
                val_lpips += metrics["lpips"] * inputs.size(0)
                val_samples += inputs.size(0)

                # Save sample validation images
                if i == 0 and self.rank == 0:
                    # Convert to RGB
                    sensor = self.camera.sensor
                    sensor.reset_augmentation()
                    inputs_rgb = sensor.process2rgb(inputs[:, :4, :, :])
                    outputs_rgb = sensor.process2rgb(outputs.detach()[:, :4, :, :])
                    targets_rgb = sensor.process2rgb(targets[:, :4, :, :])

                    # Save images
                    save_image(
                        torch.cat([inputs_rgb, outputs_rgb, targets_rgb], dim=2),
                        f"{self.args['result_dir']}/val_epoch{epoch}_{i}.png",
                    )

        # Gather validation metrics from all processes
        val_psnr_tensor = torch.tensor([val_psnr]).to(self.device)
        val_ssim_tensor = torch.tensor([val_ssim]).to(self.device)
        val_lpips_tensor = torch.tensor([val_lpips]).to(self.device)
        val_samples_tensor = torch.tensor([val_samples]).to(self.device)

        dist.all_reduce(val_psnr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_ssim_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_lpips_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_samples_tensor, op=dist.ReduceOp.SUM)

        # Calculate average metrics
        metrics = {}
        if val_samples_tensor.item() > 0:
            metrics["val_psnr"] = val_psnr_tensor.item() / val_samples_tensor.item()
            metrics["val_ssim"] = val_ssim_tensor.item() / val_samples_tensor.item()
            metrics["val_lpips"] = val_lpips_tensor.item() / val_samples_tensor.item()
        return metrics

    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        if self.rank != 0:
            return

        # Save model state
        torch.save(
            self.ddp_model.module.state_dict(),
            f"{self.args['result_dir']}/network_epoch{epoch}.pth",
        )

        # Save optimizer state
        torch.save(
            {
                "epoch": epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"{self.args['result_dir']}/optimizer_epoch{epoch}.pth",
        )

    def train(self):
        """Run the full training process."""
        for epoch in range(self.args["train"]["epochs"]):
            # Train one epoch
            train_loss = self.train_epoch(epoch)

            if self.rank == 0:
                print(f"Epoch {epoch + 1}/{self.args['train']['epochs']} completed.")
                print(f"Train Loss: {train_loss:.4f}")

            # Validate and save checkpoint
            if (epoch + 1) % self.args["train"]["eval_every_n_epochs"] == 0:
                self.save_checkpoint(epoch + 1)

                # Validate
                val_metrics = self.validate(epoch + 1)

                # Log epoch results
                if self.rank == 0:
                    if val_metrics:
                        print(f"Validation Loss: {val_metrics.get('val_loss', 'N/A')}")
                        print(
                            f"Validation PSNR: {val_metrics.get('val_psnr', 'N/A')} dB"
                        )
                        print(f"Validation SSIM: {val_metrics.get('val_ssim', 'N/A')}")
                        print(
                            f"Validation LPIPS: {val_metrics.get('val_lpips', 'N/A')}"
                        )

                    print("-" * 50)

        # Save final model
        if self.rank == 0:
            self.save_checkpoint(self.args["train"]["epochs"] - 1)
            print("Training completed!")


def main():
    """Main function to start the distributed training."""
    # Initialize the distributed environment
    setup()

    # Get local rank and world size
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Training configuration
    args = config()

    try:
        # Create trainer and start training
        trainer = Trainer(local_rank, world_size, args)
        trainer.train()
    finally:
        # Make sure to clean up even if there's an error
        cleanup()


if __name__ == "__main__":
    main()
