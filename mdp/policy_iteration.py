import random

# Performs policy iteration on a set of states and transitions to learn the optimal policy
def policy_iteration(states, actions, transitions, rewards, gamma=0.9, iterations=10):
    policy = {s: random.choice(actions) for s in states}  # Initialize policy randomly
    V = {s: 0 for s in states}  # Initialize all state values to 0

    for _ in range(iterations):
        # Policy Evaluation: estimate value of current policy
        for s in states:
            a = policy[s]
            next_states = transitions.get((s, a), [s])  # Possible next states for (state, action)
            V[s] = sum(rewards.get((s, a), 0) + gamma * V.get(ns, 0) for ns in next_states) / len(next_states)

        # Policy Improvement: update policy based on new state values
        for s in states:
            best_a = max(actions, key=lambda a: sum(
                rewards.get((s, a), 0) + gamma * V.get(ns, 0)
                for ns in transitions.get((s, a), [s])
            ))
            policy[s] = best_a  # Update action for state if a better one is found

    return policy  # Return the improved policy