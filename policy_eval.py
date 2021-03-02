import numpy as np
from gridworld import GridworldEnv


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    nV = np.zeros(env.nS)

    delta = 1
    while delta > theta:
        for s in range(len(V)):
            v = 0
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += policy[s][a] * prob * (reward + discount_factor * V[next_state])
            nV[s] = v

        delta = np.linalg.norm(nV - V, np.inf)
        V = np.copy(nV)

    return np.array(V)


if __name__ == '__main__':
    env = GridworldEnv()

    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = policy_eval(random_policy, env)

    expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)