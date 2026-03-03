import numpy as np

"""Assignment-wide notation:

S: dimensionality of state vector.
A: dimensionality of action vector.
N: number of environments in batch.
H: horizon (episode length).
"""


class GaussianLinearPolicy:
    def __init__(self, S, A, sigma, seed):
        """Initializes the linear policy parameters and the RNG.

        The Action distribution is Gaussian with mean Ws + b and covariance
        sigma^2 I, where the policy paramter W (resp. b) is a matrix (resp.
        vector) of appropriate dimension.

        Initialize each entry of W with a Gaussian distribution of mean 0 and
        standard deviation 0.01 independently. Initialize b a vector where all
        entries are 0.5 -- this gives hover thrust as the "default" action.

        The seed should be used to initialize a numpy.random.Generator and
        store it for future sample_action() calls, giving repeatable results.

        Args:
            S (int): State dimension.
            A (int): Action dimension.
            std (float): Elementwise action standard deviation.
            seed (int): PRNG seed.
        """
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, 0.01, (A, S))
        self.b = np.full(A, 0.5)

    def sample_action(self, states):
        """Samples actions from the policy.

        See __init__ docstring for the desired action distribution.
        Gaussian noise enables the agent to explore different actions 
        rather than always taking the same deterministic action for a given state.

        Args:
            states (array(N, S)): Batch of states.

        Returns:
            actions (array(N, A)): Batch of actions from policy.
        """
        N = len(states)
        # Compute mean actions W @ s + b for each state
        mean_actions = self.W @ states.T + self.b[:, np.newaxis]  # (A, N)
        
        # Sample Gaussian noise with standard deviation sigma
        noise = self.rng.normal(0, self.sigma, mean_actions.shape)  #(A, N)
        
        # Add noise to mean
        actions = mean_actions + noise
        return actions.T

    def log_prob_gradient(self, states, actions):
        """Computes the log-probability gradients for the given actions.

        See handout for details.

        Args:
            states (array(H, N, S)): States.
            actions (array(H, N, A)): Actions.

        Returns:
            grad_W (array(H, N, A, S)): Log-probability gradients w.r.t. W for
                each state-action pair in the input.
            grad_b (array(H, N, A)): Log-probability gradients w.r.t. b for
                each state-action pair in the input.
        """
        # TODO: Implement.
        H, N, _ = states.shape
        A, S = self.W.shape
        return np.zeros((H, N, A, S)), np.zeros((H, N, A))


def rollout(env, policy, H, render=False):
    """Deploys the Quadrotor policy for one episode and collects trajectories.

    Use env.reset(randomize=True) at the beginning of the episode. Call
    policy.sample_action and env.step repeatedly.

    Args:
        env (Quadrotor): Batch Quadrotor environment.
        policy (GaussianLinearPolicy): Policy.
        H (int): Episode length.

    Returns:
        states (array(H, N, S)): State trajectory batch.
        actions (array(H, N, A)): Action trajectory batch.
        rewards (array(H, N)): Reward trajectory batch.
    """
    # TODO: Implement.
    N = env.n_envs
    A, S = policy.W.shape
    return np.zeros((H, N, S)), np.zeros((H, N, A)), np.zeros((H, N))


def advantage_estimate(rewards):
    """Computes our advantage function estimate.

    See handout for details.

    Args:
        rewards (array(H, N)): Reward trajectory batch.

    Returns:
        advantages (array(H, N)): Advantage estimates.
    """
    # TODO: Implement.
    return np.zeros_like(rewards)


def policygrad_step(policy, states, actions, rewards, learning_rate):
    """Computes a policy gradient update using the given trajectory batch.

    Accumulate gradients using the batch Markovian REINFORCE, as discussed in
    the handout. Use your `advantage_estimate` and `policy.log_prob_gradient`
    as subroutines.

    Args:
        policy (GaussianLinearPolicy): Policy.
        states (array(H, N, S)): State trajectory batch.
        actions (array(H, N, A)): Action trajectory batch.
        rewards (array(H, N)): Reward trajectory batch.

    Returns:
        Nothing. Modifies policy.{W, b} in place.
    """
    # TODO: Implement.
    pass
