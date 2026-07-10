# gym -> gymnasium shim so or_gym doesn't blow up on import
# (or_gym still imports the old "gym" package under the hood)
import sys
import gymnasium

sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces

import numpy as np
from collections.abc import Iterable
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TRPO
from or_gym.utils import assign_env_config


class NewsvendorEnv(gymnasium.Env):
    def __init__(self, *args, **kwargs):
        # defaults - can all be overri dden via kwargs, see assign_env_config below
        self.lead_time = 5
        self.max_inventory = 4000
        self.max_order_quantity = 2000
        self.step_limit = 40
        self.p_max = 100
        self.h_max = 5
        self.k_max = 10
        self.mu_max = 200
        self.gamma = 1
        assign_env_config(self, kwargs)

        # obs = [price, cost, holding cost, stockout cost, mean demand] + pipeline inventory
        self.obs_dim = self.lead_time + 5
        self.observation_space = gymnasium.spaces.Box(
            low=np.zeros(self.obs_dim, dtype=np.float32),
            high=np.array(
                [self.p_max, self.p_max, self.h_max, self.k_max, self.mu_max] +
                [self.max_order_quantity] * self.lead_time),
            dtype=np.float32)

        self.action_space = gymnasium.spaces.Box(
            low=np.zeros(1), high=np.array([self.max_order_quantity]),
            dtype=np.float32)

        self.reset()

    def _STEP(self, action):
        done = False

        # SB3 passes actions in as an ndarray, we just want the float
        action_val = action.item() if isinstance(action, np.ndarray) else action

        # clamp order qty so we don't blow past max_inventory
        order_qty = max(0, min(action_val, self.max_inventory - self.state[5:].sum()))
        demand = np.random.poisson(self.mu)
        inventory = self.state[5:]

        if self.lead_time == 0:
            inv_on_hand = order_qty
        else:
            inv_on_hand = inventory[0]  # oldest batch in the pipeline is what's available now

        sales = min(inv_on_hand, demand) * self.price
        excess_inventory = max(0, inv_on_hand - demand)
        short_inventory = max(0, demand - inv_on_hand)

        # note: purchase_cost scales with excess_inventory here, not order_qty alone -
        # keeping as-is to match the original formulation, just flagging it looks off
        purchase_cost = excess_inventory * self.cost * order_qty * self.gamma ** self.lead_time
        holding_cost = excess_inventory * self.h
        lost_sales_penalty = short_inventory * self.k
        reward = sales - purchase_cost - holding_cost - lost_sales_penalty

        # shift the pipeline down one slot and drop the new order in at the end
        new_inventory = np.zeros(self.lead_time)
        new_inventory[:-1] += inventory[1:]
        new_inventory[-1] += order_qty

        self.state = np.hstack([self.state[:5], new_inventory], dtype=np.float32)
        self.step_count += 1

        if self.step_count >= self.step_limit:
            done = True

        if isinstance(reward, Iterable):
            reward = sum(reward)

        # gymnasium wants 5 values back (obs, reward, terminated, truncated, info)
        return self.state, float(reward), done, False, {}

    def _RESET(self, seed=None, options=None):
        super().reset(seed=seed)

        # roll a fresh random "market" each episode - price/cost/etc all change
        self.price = max(1, np.random.rand() * self.p_max)
        self.cost = max(1, np.random.rand() * self.price)
        self.h = np.random.rand() * min(self.cost, self.h_max)
        self.k = np.random.rand() * self.k_max
        self.mu = np.random.rand() * self.mu_max

        self.state = np.zeros(self.obs_dim, dtype=np.float32)
        self.state[:5] = np.array([self.price, self.cost, self.h, self.k, self.mu])
        self.step_count = 0

        return self.state, {}

    def reset(self, seed=None, options=None):
        return self._RESET(seed=seed, options=options)

    def step(self, action):
        return self._STEP(action)


if __name__ == "__main__":
    env = NewsvendorEnv()
    check_env(env, warn=False)

    # VecNormalize keeps obs/reward on a sane scale - without it PPO/TRPO
    # tend to struggle since price and demand are on totally different ranges
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # bumped network size to 256x256 and dropped lr to 1e-4, default settings
    # were kind of unstable on this env
    ppo_model = PPO(
        "MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=1e-4,
        gamma=0.999,
        batch_size=128,
        policy_kwargs=dict(net_arch=[256, 256])
    )

    trpo_model = TRPO(
        "MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=1e-4,
        gamma=0.999,
        batch_size=128,
        policy_kwargs=dict(net_arch=[256, 256])
    )

    # swap this to switch between PPO/TRPO
    # active_model = ppo_model
    active_model = trpo_model

    print("Training model...")
    active_model.learn(total_timesteps=500000)

    print("\nEvaluating model...")
    # turn normalization off for eval so the reward we print is actual dollars,
    # not the normalized value
    vec_env.training = False
    vec_env.norm_reward = False

    mean_reward, std_reward = evaluate_policy(active_model, vec_env, n_eval_episodes=5)
    print(f"Mean Evaluation Reward: {mean_reward:.2f} +/- {std_reward:.2f}")