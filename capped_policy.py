"""
capped_policy.py
==================
A custom PPO policy that caps the standard deviation of the action
distribution per-dimension, instead of SB3's default behavior where
std = exp(log_std) is free to grow without any upper bound (ent_coef
only slows how fast it grows, it never puts a ceiling on it).

WHY THIS EXISTS:
Direct comparison against a working reference PPO-for-TORCS
implementation showed the most likely root cause of this project's
long-standing "steering flips between extremes every step" and
"accel and brake both pressed hard at once" problems: in that
reference implementation, the per-dimension std is computed as
    std = cap * sigmoid(raw_param)
with caps of 0.2 (steer), 0.2 (accel), 0.05 (brake) -- against action
ranges of [-1,1], [0,1], [0,1] respectively, meaning the noise can
never exceed 10-20% of the action's range. In this project's logs,
the measured `std` metric was repeatedly 1.6-1.8 -- LARGER than the
[-1,1] steering range itself, meaning the policy's output was close
to indistinguishable from pure random sampling, every single step.
That single architectural difference plausibly explains several
previously-diagnosed symptoms at once (steering chaos, simultaneous
accel/brake, std never coming down despite lowering ent_coef) rather
than these being three independent problems each needing their own
reward-side patch.

HOW IT WORKS:
- CappedDiagGaussianDistribution subclasses SB3's DiagGaussianDistribution,
  overriding only proba_distribution() to compute
  action_std = caps * sigmoid(log_std) instead of action_std = exp(log_std).
  The underlying learnable parameter (still called `log_std` for
  compatibility with SB3's existing code paths) is now interpreted as
  a pre-sigmoid logit, not a literal log-std -- the name is a holdover,
  not a literal description, after this change.
- CappedActorCriticPolicy subclasses SB3's ActorCriticPolicy, overriding
  only _build() to swap in the capped distribution (with this
  project's specific per-action caps) before delegating to the
  parent's _build() for everything else (network architecture, value
  head, optimizer) -- unchanged.

VERIFIED: this module was smoke-tested in isolation against a dummy
Box(29,)-observation / Box([-1,0,0],[1,1,1])-action environment
(matching this project's actual spaces) to confirm it builds, and that
sampled actions' std empirically stays under the configured caps even
after many random gradient steps -- see the bottom of this file for
that test, intended to be run standalone (not part of the training
pipeline).
"""

import torch as th
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

import config


class CappedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Same as SB3's DiagGaussianDistribution, except the standard
    deviation is capped per-dimension via sigmoid * fixed_cap instead
    of being computed as exp(log_std) (which has no upper bound).
    """

    def __init__(self, action_dim: int, caps):
        super().__init__(action_dim)
        # Stored as a plain tensor (Distribution is not an nn.Module,
        # so no register_buffer) -- moved to the right device on each
        # call in proba_distribution(), matching mean_actions' device.
        self.caps = th.as_tensor(caps, dtype=th.float32)
        assert self.caps.shape == (action_dim,), (
            f"caps must have shape ({action_dim},), got {self.caps.shape}"
        )

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor):
        """
        Overridden: action_std = caps * sigmoid(log_std), instead of
        the parent class's action_std = exp(log_std). `log_std` is the
        same underlying nn.Parameter SB3 creates and trains via
        proba_distribution_net() (inherited unchanged) -- only its
        INTERPRETATION changes here, from "literal log std" to
        "pre-sigmoid logit".
        """
        caps = self.caps.to(mean_actions.device)
        action_std = th.ones_like(mean_actions) * caps * th.sigmoid(log_std)
        self.distribution = th.distributions.Normal(mean_actions, action_std)
        return self


class CappedActorCriticPolicy(ActorCriticPolicy):
    """
    Drop-in replacement for SB3's default continuous-action policy,
    with the standard deviation capped per-action-dimension (see
    CappedDiagGaussianDistribution above). Use via:

        model = PPO(CappedActorCriticPolicy, env, ...)

    Caps are read from config.POLICY_PARAMS["sigma_caps"], in the same
    order as the action space: [steer, accel, brake].
    """

    def _build(self, lr_schedule) -> None:
        # Swap in the capped distribution BEFORE calling the parent's
        # _build(), which checks isinstance(self.action_dist,
        # DiagGaussianDistribution) -- still True here since
        # CappedDiagGaussianDistribution is a subclass, so the parent
        # logic correctly creates action_net/log_std via the
        # (inherited, unchanged) proba_distribution_net().
        action_dim = self.action_dist.action_dim if hasattr(self.action_dist, "action_dim") else len(config.POLICY_PARAMS["sigma_caps"])
        self.action_dist = CappedDiagGaussianDistribution(
            action_dim, caps=config.POLICY_PARAMS["sigma_caps"]
        )
        super()._build(lr_schedule)


if __name__ == "__main__":
    # ============================================================
    # Standalone smoke test -- NOT part of the training pipeline.
    # Run directly (`python capped_policy.py`) to verify the policy
    # builds correctly and that sampled actions' empirical std stays
    # under the configured caps, using a dummy environment with the
    # same observation/action space shapes as the real TORCS env.
    # ============================================================
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    class DummyTorcsLikeEnv(gym.Env):
        """Minimal stand-in: same obs/action shapes as the real env, random dynamics."""

        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(config.STATE_DIM,), dtype=np.float32)
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self._step_count = 0

        def reset(self, *, seed=None, options=None):
            self._step_count = 0
            return self.observation_space.sample(), {}

        def step(self, action):
            self._step_count += 1
            obs = self.observation_space.sample()
            reward = float(np.random.randn())
            terminated = self._step_count >= 50
            return obs, reward, terminated, False, {}

    print("Building env...")
    env = DummyVecEnv([lambda: DummyTorcsLikeEnv()])

    print("Building PPO model with CappedActorCriticPolicy...")
    model = PPO(CappedActorCriticPolicy, env, n_steps=64, batch_size=32, verbose=0)

    print(f"action_dist type: {type(model.policy.action_dist)}")
    assert isinstance(model.policy.action_dist, CappedDiagGaussianDistribution), "Wrong distribution class!"

    print("Running a few rollouts + updates...")
    model.learn(total_timesteps=256)

    print("Sampling actions to empirically check std stays under caps...")
    obs = env.reset()
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    with th.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        actions = th.stack([distribution.distribution.rsample() for _ in range(2000)])
    empirical_std = actions.std(dim=0).squeeze().numpy()
    caps = np.array(config.POLICY_PARAMS["sigma_caps"])
    print(f"Configured caps:  {caps}")
    print(f"Empirical std:    {empirical_std}")
    assert np.all(empirical_std <= caps + 1e-3), (
        f"Empirical std {empirical_std} exceeded caps {caps}!"
    )
    print("PASSED: empirical std stayed within configured caps after training.")