import numpy as np
import pprint
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
        print('eval iter ..')
        for s in range(len(V)):
            v = 0
            for a in range(env.nA):
                prob, next_state, reward, _ = env.P[s][a][0]
                v += policy[s][a] * prob * (reward + discount_factor * V[next_state])
            nV[s] = v

        delta = np.linalg.norm(nV - V, np.inf)
        print(delta)
        V = np.copy(nV)

    return np.array(V)


def policy_improvement(env, policy_eval=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while not policy_stable:
        V = policy_eval(policy, env, discount_factor)
        policy_stable = True

        for s in range(env.nS):
            prev_a = np.argmax(policy[s])
            cur_a_vals = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    cur_a_vals[a] += prob * (reward + discount_factor * V[next_state])

            cur_a = np.argmax(cur_a_vals)

            if prev_a != cur_a:
                policy_stable = False

    return policy, V


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=2)
    env = GridworldEnv()

    policy, v = policy_improvement(env)

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