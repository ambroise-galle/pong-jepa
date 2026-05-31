import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.jepa import JEPAWorldModel
from models.decoder import SpatialDecoder
from utils.buffer import ReplayBuffer

def main():
    parser = argparse.ArgumentParser(description="Train Spatial Decoder on top of frozen JEPA representations")
    parser.add_argument("--jepa_path", type=str, default="checkpoints/jepa_best.pt", help="Path to pre-trained JEPA model")
    parser.add_argument("--data_path", type=str, default="data/pong_transitions_100k.npz", help="Path to collected transition dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs/decoder", help="Directory for TensorBoard logs")
    parser.add_argument("--device", type=str, default="auto", help="Execution device")
    parser.add_argument("--use_wandb", action="store_true", help="Log metrics and reconstruction grids to Weights & Biases")
    parser.add_argument("--project_name", type=str, default="pong-jepa", help="Wandb project name")
    args = parser.parse_args()

    # Device configuration
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Training Spatial Decoder on device: {device}")

    # 1. Load pre-trained JEPA model and extract the online encoder
    print(f"Loading pre-trained JEPA from {args.jepa_path}...")
    if not os.path.exists(args.jepa_path):
        raise FileNotFoundError(f"JEPA checkpoint not found at {args.jepa_path}. Please train the JEPA model first!")
    
    checkpoint = torch.load(args.jepa_path, map_location=device, weights_only=False)
    jepa_model = JEPAWorldModel(in_channels=4, latent_dim=256, action_dim=6).to(device)
    jepa_model.load_state_dict(checkpoint["model_state_dict"])
    
    encoder = jepa_model.online_encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    print("Pre-trained Encoder successfully loaded and frozen.")

    # 2. Load dataset
    print(f"Loading transition dataset from {args.data_path}...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset file {args.data_path} not found. Please run collect_data.py first!")
        
    data_archive = np.load(args.data_path)
    dataset_size = int(data_archive['size'])
    buffer = ReplayBuffer(capacity=dataset_size, state_shape=(4, 64, 64), device=device)
    buffer.load(args.data_path)

    # 3. Initialize Decoder
    decoder = SpatialDecoder(latent_dim=256).to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)

    # 4. Logger Setup
    writer = SummaryWriter(log_dir=args.log_dir)
    if args.use_wandb:
        try:
            import wandb
            # Since the user already logged in, we init the run
            wandb.init(project=args.project_name, config=vars(args), name="spatial-decoder-training")
            print("Successfully initialized Wandb logging.")
        except ImportError:
            print("Wandb not installed. Defaulting to TensorBoard only.")
            args.use_wandb = False

    steps_per_epoch = dataset_size // args.batch_size
    best_loss = float("inf")

    # Sample a fixed batch of example frames to visualize progress across epochs
    eval_states, _, _, _, _ = buffer.sample(8)
    with torch.no_grad():
        eval_z = encoder(eval_states)
    # The target is the most recent frame in the stack (channel index -1)
    eval_targets = eval_states[:, -1, :, :].unsqueeze(1) # Shape (8, 1, 64, 64)

    print("Beginning Spatial Decoder training loop...")
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        epoch_losses = []

        for step in range(steps_per_epoch):
            # Sample transitions
            states, _, _, _, _ = buffer.sample(args.batch_size)
            
            # The target is the most recent frame in the stack (channel index -1)
            targets = states[:, -1, :, :].unsqueeze(1) # Shape (batch, 1, 64, 64)

            # Get representation and reconstruct
            with torch.no_grad():
                z = encoder(states)
            
            recon = decoder(z)

            # Compute Reconstruction Loss (MSE)
            loss = F.mse_loss(recon, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            global_step = (epoch - 1) * steps_per_epoch + step
            if global_step % 50 == 0:
                writer.add_scalar("Decoder/ReconstructionLoss", loss.item(), global_step)

        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Reconstruction Loss (MSE): {mean_loss:.6f}")

        # Live Reconstruction Grid to W&B
        if args.use_wandb:
            decoder.eval()
            with torch.no_grad():
                eval_recon = decoder(eval_z)
            
            # Prepare a comparison grid: top row = Real, bottom row = Reconstructed
            # Convert tensors to numpy arrays in [0, 255]
            real_np = (eval_targets.cpu().numpy() * 255.0).astype(np.uint8)
            recon_np = (eval_recon.cpu().numpy() * 255.0).astype(np.uint8)
            
            # Concatenate horizontally
            real_row = np.concatenate([real_np[i, 0] for i in range(8)], axis=1)
            recon_row = np.concatenate([recon_np[i, 0] for i in range(8)], axis=1)
            
            # Concatenate vertically
            comparison_grid = np.concatenate([real_row, recon_row], axis=0)
            
            # Log grid to W&B
            wandb.log({
                "reconstruction_loss": mean_loss,
                "reconstructions": wandb.Image(
                    comparison_grid, 
                    caption="Live Reconstruction (Top row: True frames, Bottom row: Latent reconstructions)"
                )
            }, step=epoch)

        # Save Checkpoints
        os.makedirs(args.save_dir, exist_ok=True)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_path = os.path.join(args.save_dir, "decoder_best.pt")
            torch.save(decoder.state_dict(), best_path)
            print(f"--> Saved new best decoder to {best_path} with loss {best_loss:.6f}")

    print("Decoder training completed successfully!")
    writer.close()
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
