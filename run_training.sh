#!/bin/bash
# High-Performance Runner and Setup Script for JEPA World Model on Atari Pong
# Optimized for quick deployment on Thunder Compute and remote GPU instances.

set -e # Exit immediately on error

echo "================================================================="
echo "        JEPA / World Model Atari Pong - Deployment Setup"
echo "================================================================="

# 1. Setup Directories
echo "--> Creating directory structure..."
mkdir -p data checkpoints logs/jepa

# 2. Verify and install dependencies
echo "--> Verifying Python environment and dependencies..."
pip install -r requirements.txt

# 3. Import Atari ROMs (Required by Gymnasium)
echo "--> Importing Atari ROMs with AutoROM (License accepted)..."
AutoROM --accept-license

echo "--> Environment successfully prepared!"
echo "================================================================="
echo "Available Commands (uncomment in this script or run manually):"
echo "================================================================="
echo "1. Collect Transition Data (runs heuristic-guided policy with 20% exploration):"
echo "   python3 collect_data.py --num_steps 100000 --save_path data/pong_transitions_100k.npz"
echo ""
echo "2. Train the JEPA World Model (representation & dynamics learning):"
echo "   python3 train_jepa.py --epochs 30 --data_path data/pong_transitions_100k.npz --save_dir checkpoints --log_dir logs/jepa"
echo ""
echo "3. Train downstream DQN Policy on frozen JEPA representations:"
echo "   python3 play_pong.py --episodes 500 --jepa_path checkpoints/jepa_best.pt --save_dir checkpoints"
echo ""
echo "4. Generate Latent Space (t-SNE) and Predictor Drift visualizations:"
echo "   python3 visualize.py --jepa_path checkpoints/jepa_best.pt"
echo "================================================================="

# Provide a fast dry-run option
if [ "$1" == "--dry-run" ]; then
    echo "Running quick dry-run test to verify full script structure..."
    python3 collect_data.py --num_steps 1000 --save_path data/pong_dryrun.npz
    python3 train_jepa.py --epochs 1 --data_path data/pong_dryrun.npz --batch_size 64
    echo "Dry run completed successfully! Ready for full training."
fi
