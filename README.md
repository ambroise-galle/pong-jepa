# JEPA World Model for Atari Pong

A modular, high-performance Joint Embedding Predictive Architecture (JEPA) / World Model repository designed to learn representation and dynamics of the Gymnasium Atari Pong environment. Built with PyTorch and designed for easy containerized or VM deployment on high-performance GPU services like **Thunder Compute**.

Rather than using generative reconstruction (predicting pixels), this model learns transitions entirely in a $L_2$-normalized latent space with a BYOL-style EMA target encoder to guarantee collapse prevention.

---

## Repository Structure

```
jepa_pong/
├── README.md                 # Project guide and instructions
├── requirements.txt          # PyTorch, Gymnasium[atari], OpenCV, etc.
├── Dockerfile                # High-performance PyTorch Docker container
├── run_training.sh           # Easy setup, install and runner script
├── choices.md                # Rationale behind key architectural choices
├── collect_data.py          # Data collection script via heuristic-guided policy
├── train_jepa.py            # Main training loop for the JEPA World Model
├── play_pong.py              # Downstream representation-based DQN training script
├── models/
│   ├── __init__.py
│   ├── jepa.py              # JEPA Encoder, Predictor, Target Encoder, and losses
│   └── policy.py            # Downstream DQN Q-Network MLP
├── utils/
│   ├── __init__.py
│   ├── env.py               # Frame skipping, stacking, resizing, and warping
│   └── buffer.py            # Highly memory-efficient transition replay buffers
└── visualize.py             # Evaluation: t-SNE latent projections & prediction drift
```

---

## Quickstart: VM / Rented Compute (e.g. Thunder Compute)

Clone this repository to your rented VM and run the setup script:

```bash
# Make the setup script executable and run it
# This will install requirements and import the license-accepted Atari ROMs
chmod +x run_training.sh
./run_training.sh
```

### Complete Workflow in 4 Commands:

#### 1. Collect Transitions
Collect a dataset of $100,000$ transitions using our epsilon-heuristic policy (keeps the ball in play to collect excellent dynamics data):
```bash
python3 collect_data.py --num_steps 100000 --save_path data/pong_transitions_100k.npz
```

#### 2. Train the JEPA World Model
Train the latent representations and transition dynamics:
```bash
python3 train_jepa.py --epochs 30 --data_path data/pong_transitions_100k.npz --save_dir checkpoints --log_dir logs/jepa
```

#### 3. Train Downstream Policy (Play)
Verify representation quality by training a tiny DQN policy directly on top of the frozen learned embeddings ($z_t$):
```bash
python3 play_pong.py --episodes 500 --jepa_path checkpoints/jepa_best.pt --save_dir checkpoints
```

#### 4. Analyze & Visualize
Generate high-fidelity t-SNE scatter plots of the latent space (colored by ball position) and evaluate multi-step prediction drift:
```bash
python3 visualize.py --jepa_path checkpoints/jepa_best.pt
```
Visualizations are saved directly to `data/latent_space_tsne.png` and `data/dynamics_drift.png`.

---

## Rationale and Key Architectural Choices

For detailed justifications of critical decisions including:
* Why JEPA beats pixel-generative world models (dreamers/reconstructors) in games
* Mathematical mechanics of the BYOL-style EMA collapse prevention
* Space-efficient framing stacking and warping ($64 \times 64$)
* Downstream evaluation design

Please refer to [choices.md](choices.md).
