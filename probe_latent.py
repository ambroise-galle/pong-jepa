import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from models.jepa import JEPAWorldModel

class CoordinateProbe(nn.Module):
    """
    Lightweight 2-layer MLP regression probe to map the 256-dimensional 
    latent representation (z_t) to the game's physical coordinates:
    [ball_x, ball_y, paddle_y_left, paddle_y_right]
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Output: [ball_x, ball_y, paddle_y_left, paddle_y_right]
        )

    def forward(self, z):
        return self.net(z)

def extract_ground_truth_coordinates(states):
    """
    Heuristically extracts ground truth positions from the raw image pixel grids.
    Input states: numpy array of shape (N, 4, 64, 64) in [0, 255]
    Returns:
        targets: numpy array of shape (M, 4) containing normalized coordinates [ball_x, ball_y, paddle_y_left, paddle_y_right]
        valid_indices: list of indices that contained fully visible/located elements
    """
    targets = []
    valid_indices = []
    
    for i in range(len(states)):
        frame = states[i][-1]  # Get most recent frame in stack
        
        # 1. Locate Ball (value > 140)
        # Search area: y from 10 to 60, x from 8 to 55
        ball_y_idxs, ball_x_idxs = np.where(frame[10:60, 8:55] > 140)
        if len(ball_y_idxs) > 0:
            ball_y = ball_y_idxs.mean() + 10
            ball_x = ball_x_idxs.mean() + 8
        else:
            continue  # Exclude frames where the ball isn't visible (e.g. scoring resetting)
            
        # 2. Locate Right Paddle (value > 100, x columns 56-59)
        paddle_y_right_idxs, _ = np.where(frame[10:60, 56:59] > 100)
        if len(paddle_y_right_idxs) > 0:
            paddle_y_right = paddle_y_right_idxs.mean() + 10
        else:
            continue
            
        # 3. Locate Left Paddle (value > 100, x columns 4-7)
        paddle_y_left_idxs, _ = np.where(frame[10:60, 4:7] > 100)
        if len(paddle_y_left_idxs) > 0:
            paddle_y_left = paddle_y_left_idxs.mean() + 10
        else:
            continue
            
        # Normalize coordinates to [0.0, 1.0] relative to 64x64 board size
        targets.append([ball_x / 64.0, ball_y / 64.0, paddle_y_left / 64.0, paddle_y_right / 64.0])
        valid_indices.append(i)
        
    return np.array(targets), valid_indices

def main():
    parser = argparse.ArgumentParser(description="Train a linear/MLP probe to extract board coordinates from latent space")
    parser.add_argument("--jepa_path", type=str, default="checkpoints/jepa_best.pt", help="Path to pre-trained JEPA model")
    parser.add_argument("--data_path", type=str, default="data/pong_transitions_100k.npz", help="Path to transition dataset")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to train the probe")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device to execute on")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu"
    print(f"Running latent coordinate probe on: {device}")

    # 1. Load pre-trained Encoder
    print(f"Loading pre-trained JEPA from {args.jepa_path}...")
    if not os.path.exists(args.jepa_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.jepa_path}.")
    checkpoint = torch.load(args.jepa_path, map_location=device, weights_only=False)
    jepa_model = JEPAWorldModel(in_channels=4, latent_dim=256, action_dim=6).to(device)
    jepa_model.load_state_dict(checkpoint["model_state_dict"])
    encoder = jepa_model.online_encoder
    encoder.eval()
    
    # 2. Load dataset transitions
    print(f"Loading transitions from {args.data_path}...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found at {args.data_path}.")
    data = np.load(args.data_path)
    states = data["states"]
    print(f"Extracting board coordinates from {len(states)} transitions...")
    
    # Extract coordinates via pixel intensities
    coords, valid_idxs = extract_ground_truth_coordinates(states)
    print(f"Located active gameplay coordinates in {len(coords)} transitions.")

    # 3. Generate Latent Representations (z) for the active transitions
    print("Generating latent representations from frozen Encoder...")
    active_states = states[valid_idxs]
    z_list = []
    
    # Process in batches to avoid GPU/CPU memory overflows
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(active_states), batch_size):
            batch_s = torch.as_tensor(active_states[i:i+batch_size], dtype=torch.float32, device=device) / 255.0
            z_batch = encoder(batch_s).cpu().numpy()
            z_list.append(z_batch)
            
    X = np.concatenate(z_list, axis=0)
    y = coords

    # Split into train & test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Coordinate Regression Probe
    print("Training Coordinate Probe MLP...")
    probe = CoordinateProbe(latent_dim=256).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    dataset_size = len(X_train)
    steps_per_epoch = dataset_size // args.batch_size

    for epoch in range(1, args.epochs + 1):
        probe.train()
        idxs = np.arange(dataset_size)
        np.random.shuffle(idxs)
        
        for step in range(steps_per_epoch):
            batch_idxs = idxs[step*args.batch_size:(step+1)*args.batch_size]
            bx = X_train_t[batch_idxs]
            by = y_train_t[batch_idxs]
            
            pred = probe(bx)
            loss = loss_fn(pred, by)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 5. Evaluate Probe Accuracy
    probe.eval()
    with torch.no_grad():
        preds_t = probe(X_test_t)
        preds = preds_t.cpu().numpy()

    # Calculate metrics
    # Multiplying by 64.0 maps normalized [0, 1] coordinates back to raw pixels
    pixel_errors = np.abs(preds - y_test) * 64.0
    mean_pixel_errors = np.mean(pixel_errors, axis=0)

    # Compute R2 score (Coefficient of Determination)
    r2_scores = [r2_score(y_test[:, i], preds[:, i]) for i in range(4)]

    print("\n" + "="*50)
    print("      JEPA Latent Space Coordinate Probe Results")
    print("="*50)
    print(f"1. Ball X-Coordinate:   R² Score = {r2_scores[0]:.4f} | Avg Pixel Error = {mean_pixel_errors[0]:.2f} px")
    print(f"2. Ball Y-Coordinate:   R² Score = {r2_scores[1]:.4f} | Avg Pixel Error = {mean_pixel_errors[1]:.2f} px")
    print(f"3. Left Paddle Y-Coord: R² Score = {r2_scores[2]:.4f} | Avg Pixel Error = {mean_pixel_errors[2]:.2f} px")
    print(f"4. Right Paddle Y-Coord:R² Score = {r2_scores[3]:.4f} | Avg Pixel Error = {mean_pixel_errors[3]:.2f} px")
    print("="*50)
    print("Interpretation:")
    print("R² score represents the proportion of variance explained by the latent features.")
    print("R² > 0.90 demonstrates that the pre-trained, self-supervised JEPA latent space")
    print("perfectly encodes the absolute physical coordinates of active game elements!")
    print("="*50)

if __name__ == "__main__":
    main()
