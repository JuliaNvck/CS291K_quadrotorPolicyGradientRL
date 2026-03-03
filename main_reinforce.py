import argparse

import numpy as np

from quadrotor import Quadrotor
import a4_student as student


def main(H=200, N=1000, learning_rate=1e-2, policy_sigma=0.05, iters=1000, seed=0, render=False):
    """Top-level function to run Policy Gradient and return success/failure data.

    Args:
        H (int): Episode length (horizon).
        N (int): Number of environments in batch.
        learning_rate (float): Learning rate for gradient ascent.
        policy_sigma (float): Action noise standard deviation.
        iters (int): Number of iterations (1 iteration = 1 batch gradient update).
        seed (int): PRNG seed for reproducibility.
        render (bool): If true, shows graphics window for greedy rollouts.

    Returns:
        episode_returns (array(iters)): For each iteration, average episode
            return across the batch environment.
    """
    env = Quadrotor(n_envs=N, seed=seed)
    policy = student.GaussianLinearPolicy(S=18, A=4, sigma=policy_sigma, seed=seed)

    returns_history = []

    for iteration in range(iters):
        print_info = iteration % 25 == 0

        s, a, r = student.rollout(env, policy, H, render=(print_info and render))

        mean_return = np.mean(np.sum(r, axis=0))
        if print_info:
            returns_history.append(mean_return)
            print(f"Iteration {iteration}: Mean Return = {mean_return:.2f}")

        student.policygrad_step(policy, s, a, r, learning_rate)

    return np.array(returns_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Policy Gradient in Quadrotor.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    main(seed=args.seed, render=args.render)
