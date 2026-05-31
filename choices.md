# Architectural Decisions and Rationale

This document details and justifies the critical design decisions made during the implementation of the Joint Embedding Predictive Architecture (JEPA) / World Model for Gymnasium Atari Pong.

---

## 1. Why Joint Embedding Predictive Architecture (JEPA)?

Most traditional model-based reinforcement learning systems (e.g. World Models by Ha & Schmidhuber, Dreamer by Hafner et al.) are **generative**. They predict the future state in *pixel space* using a decoder (e.g., VAE or GAN) to reconstruct the next observation $\hat{x}_{t+1}$. 

While pixel reconstruction is visually satisfying, it exhibits three major drawbacks:
1. **Computational Inefficiency:** Generative decoders are massive, slow, and expensive to train.
2. **The "Resource Trap":** Reconstructing fine-grained background details and noise consumes a significant portion of the model's capacity, while these details are completely irrelevant to agent decision-making.
3. **Severe Blurring in Long Horizons:** Pixel-level predictions tend to average out uncertainty, leading to extremely blurry predictions over long rollouts where crucial small elements (like the Pong ball!) vanish completely.

**JEPA** solves these problems by predicting transitions entirely in a **latent representation space** ($z_t, a_t \to \hat{z}_{t+1}$). The network is forced to learn a compressed, abstract representation of the environment, ignoring visual clutter (like the scores at the top of the screen) and focusing exclusively on predictable physics (the movement of the ball and paddles).

---

## 2. Preventing Representation Collapse: BYOL-style EMA

A primary challenge in training any joint embedding architecture is preventing **representation collapse**, where the encoder learns the trivial solution: mapping *all* inputs to a constant vector, yielding a perfect but useless predictor loss of $0$.

To solve this without needing contrastive learning (which requires computationally expensive negative pairs and complex batching), we implement a **BYOL-style (Bootstrap Your Own Latent) framework**:
* **Online Encoder & Predictor:** The online parameters are updated via regular SGD.
* **Target Encoder (EMA):** The target encoder is a structural copy of the online encoder, but its weights $\theta^-$ are updated via an **Exponential Moving Average (EMA)** of the online encoder's weights:
  $$\theta^- \leftarrow \tau \theta^- + (1-\tau)\theta \quad (\text{with } \tau = 0.99)$$
* **Stop-Gradient:** No gradients are backpropagated through the target encoder.
* **Predictor Bottleneck & Asymmetry:** The predictor is only placed on the online path.
* **Normalization:** Both predicted and target vectors are $L_2$-normalized before calculating the MSE loss.

Mathematically, this asymmetry prevents collapse because the target encoder provides a stable, slowly-moving representation target that the online encoder must actively predict via a non-linear dynamics network. The network cannot collapse to a constant because a constant representation would completely fail to predict the state transitions under different active actions.

---

## 3. Environment Design & Preprocessing

Atari frames are originally RGB and sized $210 \times 160$. To make training exceptionally fast and lightweight, we implement custom gymnasium wrappers in `utils/env.py`:
1. **Frame Skipping (Skip=4):** The action is repeated for 4 frames. This reduces step count by a factor of 4 and speeds up both data collection and training dramatically.
2. **Frame Maxing:** Takes the element-wise maximum over the last two skipped frames to eliminate the standard Atari "flickering" issue where sprites are drawn only on alternate frames.
3. **Warping to $64 \times 64$ Grayscale:** Downscaling to $64 \times 64$ is much faster to process than the standard $84 \times 84$, while retaining perfect legibility of the ball and paddles.
4. **Frame Stacking (k=4):** A single static Atari frame contains no velocity information. By stacking 4 consecutive frames, the resulting input $(4, 64, 64)$ allows a standard Feed-Forward CNN to immediately extract velocities and directions of motion.

---

## 4. Heuristic-Guided Data Collection (`collect_data.py`)

A purely random policy in Pong collects extremely poor training transitions: the agent's paddle rarely touches the ball, leading to episodes consisting entirely of the opponent scoring. The dynamics model trained on this data would never learn the physics of paddle-ball contact.

To solve this, we implemented an **epsilon-greedy heuristic-guided collection policy**:
* The script locates the Y-coordinates of the ball and the right paddle directly from the pixels of the preprocessed $(64, 64)$ frame (by looking at pixel intensities > 140 for the ball and > 100 for the paddle).
* It moves the paddle towards the ball's Y-coordinate, creating a robust baseline player.
* We inject **exploration noise ($\epsilon = 0.2$)** so the agent makes intentional errors. This fills the transition buffer with standard play, successful hits, and diverse misses, providing the JEPA with a highly rich and balanced training dataset.

---

## 5. Downstream DQN Policy on Frozen Representations

To prove that the JEPA representation is rich in game semantics, we freeze the pre-trained `Encoder` and train a Deep Q-Network (DQN) purely on the $D$-dimensional latent vector $z_t$.

Because $z_t$ is low-dimensional ($256$) and already organizes physical game states, the DQN policy does not need to learn image convolution or feature extraction. It is a simple, lightweight 3-layer MLP that trains in minutes on a standard CPU. If the agent successfully learns to play Pong, it conclusively demonstrates that the JEPA latent space has captured the fundamental coordinates, velocities, and rules of the environment.

---

## 6. Advanced Evaluation and Visualizations (`visualize.py`)

To inspect and verify the model's inner representations, we implement two diagnostic tools:
1. **Latent Space t-SNE / PCA:** Projects the high-dimensional latent space ($256$-dim) onto a $2$D scatter plot. By color-coding each point according to the ball's actual Y-coordinate (extracted via pixels), we can visually inspect if the JEPA encoder has naturally organized the latent space based on the geometry of the physical game.
2. **Open-Loop Predictor Drift Analysis:** Simulates a sequence of steps entirely in latent space using the Predictor:
   $$\hat{z}_{t+k} = \text{Predictor}(\hat{z}_{t+k-1}, a_{t+k-1})$$
   We plot the mean squared error (MSE) between the predicted representations and the true encoder representations over time. This metric measures the "imagination drift" of the World Model, verifying the dynamics predictor's quality.
