import os
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.jepa import JEPAWorldModel
from utils.buffer import ReplayBuffer

def main():
    parser = argparse.ArgumentParser(description="Train JEPA World Model on collected Pong transitions")
    parser.add_argument("--data_path", type=str, default="data/pong_transitions_100k.npz", help="Path to collected dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--latent_dim", type=int, default=256, help="Dimension of latent space")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA decay rate for target encoder")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs/jepa", help="Directory for TensorBoard logs")
    parser.add_argument("--device", type=str, default="auto", help="Device to train on (auto, cuda, mps, cpu)")
    parser.add_argument("--use_wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--project_name", type=str, default="pong-jepa", help="Wandb project name")

    args = parser.parse_args()

    # Determine training device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Training on device: {device}")

    # Load dataset
    print(f"Loading transitions from {args.data_path}...")
    # Read size first to allocate buffer
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset file {args.data_path} not found. Please run collect_data.py first!")
    
    data_archive = np.load(args.data_path)
    dataset_size = int(data_archive['size'])
    print(f"Dataset contains {dataset_size} transitions.")
    
    buffer = ReplayBuffer(capacity=dataset_size, state_shape=(4, 64, 64), device=device)
    buffer.load(args.data_path)

    # Initialize JEPA model
    print(f"Initializing JEPA World Model (latent_dim={args.latent_dim})...")
    model = JEPAWorldModel(
        in_channels=4,
        latent_dim=args.latent_dim,
        action_dim=6,  # Standard Atari Pong action space size
        ema_decay=args.ema_decay
    ).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = dataset_size // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)

    # Logging setup
    writer = SummaryWriter(log_dir=args.log_dir)
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.project_name, config=vars(args))
            print("Successfully initialized Wandb logging.")
            
            # Log an example input observation stack
            try:
                states, _, _, _, _ = buffer.sample(1)
                frame_stack = states[0].detach().cpu().numpy()
                frame_grid = np.concatenate([frame_stack[i] for i in range(4)], axis=1)
                frame_grid = (frame_grid * 255.0).astype(np.uint8)
                wandb.log({
                    "example_observation_stack": wandb.Image(
                        frame_grid, 
                        caption="Example preprocessed frame stack (4 consecutive timesteps concatenated horizontally)"
                    )
                }, step=0)
                print("Successfully logged example observation stack to Wandb.")
            except Exception as img_err:
                print(f"Warning: Failed to log example observation stack to Wandb: {img_err}")
                
        except ImportError:
            print("Wandb not installed. Defaulting to TensorBoard only.")
            args.use_wandb = False

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")

    print("Beginning training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_pred_losses = []
        epoch_reward_losses = []
        epoch_done_losses = []

        # We define an "epoch" as completing steps equivalent to one full pass over the buffer
        for step in range(steps_per_epoch):
            # Sample transitions
            states, actions, next_states, rewards, dones = buffer.sample(args.batch_size)

            # Compute losses
            loss_dict = model.compute_jepa_loss(states, actions, next_states, rewards, dones)
            loss = loss_dict["loss"]

            # Backward pass & step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update target encoder via EMA
            model.update_target_encoder()
            scheduler.step()

            # Record metrics
            epoch_losses.append(loss.item())
            epoch_pred_losses.append(loss_dict["pred_loss"].item())
            epoch_reward_losses.append(loss_dict["reward_loss"].item())
            epoch_done_losses.append(loss_dict["done_loss"].item())

            # Log to TensorBoard/Wandb periodically
            global_step = (epoch - 1) * steps_per_epoch + step
            if global_step % 50 == 0:
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/PredLoss", loss_dict["pred_loss"].item(), global_step)
                writer.add_scalar("Train/RewardLoss", loss_dict["reward_loss"].item(), global_step)
                writer.add_scalar("Train/DoneLoss", loss_dict["done_loss"].item(), global_step)
                writer.add_scalar("Train/LearningRate", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("Stats/z_t_norm", loss_dict["z_t_norm"].item(), global_step)
                
                if args.use_wandb:
                    wandb.log({
                        "loss": loss.item(),
                        "pred_loss": loss_dict["pred_loss"].item(),
                        "reward_loss": loss_dict["reward_loss"].item(),
                        "done_loss": loss_dict["done_loss"].item(),
                        "lr": scheduler.get_last_lr()[0],
                        "z_t_norm": loss_dict["z_t_norm"].item()
                    }, step=global_step)

        # Epoch summary
        mean_loss = np.mean(epoch_losses)
        mean_pred = np.mean(epoch_pred_losses)
        mean_rew = np.mean(epoch_reward_losses)
        mean_done = np.mean(epoch_done_losses)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {mean_loss:.4f} [Pred: {mean_pred:.4f}, Rew: {mean_rew:.4f}, Done: {mean_done:.4f}]")

        # Periodically log validation plots (t-SNE latent projection and multi-step predictor drift) to Wandb
        if args.use_wandb and (epoch % 10 == 0 or epoch == args.epochs):
            print(f"Epoch {epoch:02d} | Generating and logging latent space & drift plots to Wandb...")
            try:
                from utils.env import make_pong
                from visualize import get_rollout_data, plot_latent_space, evaluate_dynamics_drift
                
                eval_env = make_pong(env_id="PongNoFrameskip-v4")
                eval_steps = min(500, dataset_size)
                
                states_list, z_list, actions, _ = get_rollout_data(
                    eval_env, 
                    model.online_encoder, 
                    num_steps=eval_steps, 
                    device=device
                )
                eval_env.close()
                
                tsne_path = os.path.join(args.save_dir, f"latent_space_tsne_epoch_{epoch}.png")
                drift_path = os.path.join(args.save_dir, f"dynamics_drift_epoch_{epoch}.png")
                
                plot_latent_space(z_list, states_list, save_path=tsne_path)
                evaluate_dynamics_drift(model, z_list, actions, device=device, save_path=drift_path)
                
                wandb.log({
                    "latent_space_tsne": wandb.Image(tsne_path, caption=f"t-SNE Latent Space at Epoch {epoch}"),
                    "dynamics_drift": wandb.Image(drift_path, caption=f"Predictor Multi-Step Drift at Epoch {epoch}")
                }, step=global_step)
                print("Successfully logged latent space and drift plots to Wandb.")
            except Exception as e:
                print(f"Warning: Failed to generate or log periodic plots to Wandb: {e}")

        # Save latest checkpoint
        latest_path = os.path.join(args.save_dir, "jepa_latest.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": mean_loss,
        }, latest_path)

        # Save best checkpoint
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_path = os.path.join(args.save_dir, "jepa_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, best_path)
            print(f"--> Saved new best checkpoint to {best_path} with loss {best_loss:.4f}")

    print("Training finished.")
    writer.close()
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
