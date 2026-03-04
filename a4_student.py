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
        mean_actions = states @ self.W.T + self.b #(N, A)
        
        # Sample Gaussian noise with standard deviation sigma
        actions = self.rng.normal(loc=mean_actions, scale=self.sigma)
        
        return actions

    def log_prob_gradient(self, states, actions):
        """Computes the log-probability gradients for the given actions.

        See handout for details.

        Args:
            states (array(H, N, S)): States.
            actions (array(H, N, A)): Actions.
        
            Substitute mean and covariance into density func: p(a|s) = 1/(2pisigma^2)^{A/2}} exp(-1/2sigma^2} (a - Ws - b)^T (a - Ws - b))
            Apply properties of the logarithm: log pi(a|s) = -A/2 log(2pi*sigma^2) - 1/2sigma^2 * (a - Ws - b)^T (a - Ws - b)
            Apply chain rule: e = a - (Ws + b)
            d log pi(a|s) / db = 1/sigma^2 * (a - Ws - b) = 1/sigma^2 * e
            d log pi(a|s) / dW = 1/sigma^2 * (a - Ws - b) s^T = 1/sigma^2 * e s^T
            ( We now think of a as a constant, and ask: how would we adjust θ to increase the probability of this particular a? )

            compute the gradients for an entire batch of N length-H trajectories at once. 
            Your output will be a 4-dimensional array for ∇W and a 3-dimensional array for ∇b
        Returns:
            grad_W (array(H, N, A, S)): Log-probability gradients w.r.t. W for
                each state-action pair in the input.
            grad_b (array(H, N, A)): Log-probability gradients w.r.t. b for
                each state-action pair in the input.
        """
        H, N, _ = states.shape
        A, S = self.W.shape
        # Compute mean actions: Ws + b
        mean_actions = states @ self.W.T + self.b #(H, N, A)
        # Compute error term: e = a - (Ws + b)
        e = actions - mean_actions #(H, N, A)
        # Compute gradients
        grad_b = e / (self.sigma ** 2) #(H, N, A)
        # (H, N, A, S) = (H, N, A)[:, :, :, np.newaxis] * (H, N, S)[:, :, np.newaxis, :]
        grad_W = (e[:, :, :, np.newaxis] * states[:, :, np.newaxis, :]) / (self.sigma ** 2) #(H, N, A, S)
        return grad_W, grad_b


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
    N = env.n_envs
    A, S = policy.W.shape
    states = np.zeros((H, N, S))
    actions = np.zeros((H, N, A))
    rewards = np.zeros((H, N))
    current_states = env.reset(randomize=True)
    for t in range(H):
        # Record the current states
        states[t] = current_states
        # Sample actions for the entire batch
        actions[t] = policy.sample_action(current_states)
        # Step the environment forward
        next_states, step_rewards = env.step(actions[t])
        rewards[t] = step_rewards

        if render:
            env.render()
        # Update the states for the next iteration
        current_states = next_states
    
    return states, actions, rewards


def advantage_estimate(rewards):
    """Computes our advantage function estimate.

    See handout for details.

    Args:
        rewards (array(H, N)): Reward trajectory batch.

    Returns:
        advantages (array(H, N)): Advantage estimates.
    """
    R = np.zeros_like(rewards)
    # Rnh = sum of rewards from time step h to the end of the episode for trajectory n
    # Compute Rnh using dynamic programming: Rnh = rnh + Rn(h+1)
    R[-1] = rewards[-1] # Rn(H-1) = rn(H-1)
    for h in range(len(rewards) - 2, -1, -1):
        R[h] = rewards[h] + R[h + 1] # Rnh = rnh + Rn(h+1)

    # Calculate the mean across the batch (axis=1) and keep dimensions for broadcasting
    baseline = np.mean(R, axis=1, keepdims=True)
    A = R - baseline # A = R - baseline: difference between the episode return for simulation n starting from step h vs. the mean return from step h across the batch.
     
    return A


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
   # advantages (Shape: H, N)
    advantages = advantage_estimate(rewards)
    # log-probability gradients (Shapes: (H, N, A, S) and (H, N, A))
    grad_W, grad_b = policy.log_prob_gradient(states, actions)
    # 3. Reshape advantages for broadcasting
    # For W: (H, N) -> (H, N, 1, 1)
    adv_for_W = advantages[..., None, None]
    # For b: (H, N) -> (H, N, 1)
    adv_for_b = advantages[..., None]

    # Multiply and compute the mean across time (axis 0) and batch (axis 1)
    W_update = np.mean(grad_W * adv_for_W, axis=(0, 1)) # (A, S)
    b_update = np.mean(grad_b * adv_for_b, axis=(0, 1)) # (A,)

    # Gradient ascent update
    policy.W += learning_rate * W_update
    policy.b += learning_rate * b_update
