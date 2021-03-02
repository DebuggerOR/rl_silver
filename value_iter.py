import numpy as np
from gridworld import GridworldEnv
import pprint


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    V = np.zeros(env.nS)
    nV = np.zeros(env.nS)

    delta = 1
    while delta > theta:
        for s in range(len(V)):
            values = []
            for a in range(env.nA):
                v=0
                for prob, next_state, reward, done in env.P[s][a]:
                    v += prob * (reward + discount_factor * V[next_state])
                values.append(v)
            nV[s] = max(values)

        delta = np.linalg.norm(nV - V, np.inf)
        V = np.copy(nV)

    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        policy[s][np.argmax(V[s])] = 1

    # Implement!
    return policy, V


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=2)
    env = GridworldEnv()

    policy, v = value_iteration(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")

    # Test the value function
    expected_v = np.array([0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
