import cv2
import numpy as np
import ale_py
import gymnasium as gym
from gymnasium import spaces


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        info = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env, width=64, height=64):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize frame
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))

class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = []
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.k
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        return np.concatenate(self.frames, axis=0)

class PongRewardShaping(gym.Wrapper):
    """
    Custom wrapper that shapes the rewards in Atari Pong by awarding a dense reward
    of +0.2 whenever the player (right paddle) successfully hits/returns the ball.
    This helps the reinforcement learning agent learn the physics of contact very quickly
    rather than relying on extremely sparse game scoring events.
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_ball_x = None
        self.prev_ball_y = None
        self.ball_moving_right = True
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_ball_x = None
        self.prev_ball_y = None
        self.ball_moving_right = True
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract current frame (last channel of frame stack)
        frame = obs[-1]
        
        # 1. Locate ball (cols 8 to 55, y cols 10 to 60)
        ball_y_idxs, ball_x_idxs = np.where(frame[10:60, 8:55] > 140)
        if len(ball_x_idxs) > 0:
            ball_x = ball_x_idxs.mean() + 8
            ball_y = ball_y_idxs.mean() + 10
            
            # Track direction changes
            if self.prev_ball_x is not None:
                if ball_x > self.prev_ball_x:
                    self.ball_moving_right = True
                elif ball_x < self.prev_ball_x:
                    # Reversal of direction: ball is now moving left.
                    # Was it moving right and close to the player paddle?
                    if self.ball_moving_right and self.prev_ball_x >= 51:
                        # Locate player paddle (cols 56 to 59, y cols 10 to 60)
                        paddle_y_idxs, _ = np.where(frame[10:60, 56:59] > 100)
                        if len(paddle_y_idxs) > 0:
                            paddle_y = paddle_y_idxs.mean() + 10
                            # Check vertical overlap
                            if abs(ball_y - paddle_y) <= 6:
                                reward += 0.2
                                print(f"--> [Reward Shaping] Successfully hit the ball! Awarded +0.2 (ball_x: {ball_x:.1f}, paddle_y: {paddle_y:.1f})")
                                
                    self.ball_moving_right = False
                    
            self.prev_ball_x = ball_x
            self.prev_ball_y = ball_y
        else:
            self.prev_ball_x = None
            self.prev_ball_y = None
            
        return obs, reward, terminated, truncated, info

class RestrictActions(gym.Wrapper):
    """
    Wrapper to restrict discrete actions in Pong from 6 to 3 core actions:
    0: NOOP (Stay still, maps to raw Pong action 0)
    1: UP (moves paddle UP, maps to raw Pong action 2)
    2: DOWN (moves paddle DOWN, maps to raw Pong action 3)
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)
        self.action_mapping = {
            0: 0,  # NOOP
            1: 2,  # UP
            2: 3   # DOWN
        }

    def step(self, action):
        raw_action = self.action_mapping[int(action)]
        return self.env.step(raw_action)

def make_pong(env_id="PongNoFrameskip-v4", render_mode=None, width=64, height=64, k=4, reward_shaping=False):
    env = gym.make(env_id, render_mode=render_mode)
    # Apply standard Atari wrappers
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame(env, width=width, height=height)
    env = ImageToPyTorch(env)
    env = FrameStack(env, k=k)
    env = RestrictActions(env)
    if reward_shaping:
        env = PongRewardShaping(env)
    return env


