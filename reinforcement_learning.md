# Reinforcement Learning
Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward. RL problems are usually formulated using the framework of Markov Decision Processes, where transitions between states depend only on the current state and action.
- Environment: the conditions in which the agent takes actions in. It governs how the action changes states and provides rewards based on result or a desired goal for training the agent.
- State: the current conditions of the agent within the environment.
- Action: a decision made by the agent in a given state.
- Policy: the mapping of states to actions for an agent. This describes the actions that will be taken given a certain state.
- Reward: the desirability of a given state-action pair

## Bellman equations
The Bellman equations describe the value function of a Markov Decision Process as the immediate reward plus discounted future values. There are two equations: state-value function V(s) and action-value function Q(s, a).

### State-value function
The state value function describes the expected reward of taking any action from given state under a policy plus the reward of future states, discounted by a factor gamma.

[equation]

### Action-value function
The action-value function describes the immediate reward of taken an action at a given state under a policy, plus the value of future actions in future states, discounted by a factor gamma.

[equation]

## Q-learning
Q-learning is a model-free RL algorithm used to find an optimal action-selection policy for a Markov Decision Process. It learns the action-value function Q(s,a) that will govern a certain policy. 

Usually Q-learning is employed when the environment is highly constrained and the number of state-action pairs is small enough to track in a table, the Q-table. The Q-table is randomly initialized, and a policy is chosen (such as epsilon-greedy). Then, during an episode, the policy chooses actions from a beginning state until it reaches a terminal state or number of actions. The environment will provide rewards based on the actions and the Q-table is updated using the update rule:

Q(s, a) = Q(s, a) + alpha * [r + gamma * max Q(s', a') - Q(s, a)]

Here, alpha is the learning rate, gamma is the discount factor, s is the current state, a is the action taken, r is the reward received, s' is the next state, and a' is the action that has the highest Q-value in the next state.

This process is repeated until the Q values converge or a stop criteria is met. Now, the optimal policy is the one that chooses the action with the highest Q value for each state.

Q-learning is model-free because it does not model the environment's dynamics. Instead, it approximates the action-value function by updating the Q values.

### Policies
In Q-learning, policies describe how the state-action space is explored when trying to learn the Q values.

**Epsilon-greedy** - This is the most commonly used policy. With a probability of 1-epsilon, the agent chooses the action with the highest estimated Q-value. With a probability of epsilon, the agent chooses an action randomly. This allows the agent to explore the environment and find better policies while also exploiting its current knowledge to get rewards. Sometimes, the epsilon is set to a high value earlier in training to explore more then decays over time to exploit more.

**Boltzmann Exploration** - Similar to epsilon-greedy, but extends it to always choose an action probabilistically based on their Q-values. The probability of choosing an action is proportional to the exponential of its Q-value, scaled by a temperature parameter. High temperatures cause the actions to be chosen almost uniformly randomly, while low temperatures cause the actions with the highest Q-values to be chosen most often.

**UCB (Upper Confidence Bound) Policy** - UCB calculates a deterministic confidence interval of the expected Q value based on the number of times the actions was selected and the number of total actions taken. The new Q value is then the expected value plus the confidence interval.

**Thomson Sampling** - 

### Deep Q-networks
To handle large state spaces that cannot be tracked with a Q-table, we can use neural networks to estimate the Q-function. The input would be the state and the output is the Q values for all actions. The weights are updated by minimizing the difference between the predicted Q values and the target value based on the Bellman equation.

(GPT) One of the main challenges when using DQNs is that standard Q-learning updates can be highly correlated, which can lead to instabilities in learning. To deal with this, DQNs typically use two techniques: Experience Replay and Fixed Q-Targets.

**Experience Replay** - Instead of learning from each experience as it occurs, experiences (state, action, reward, next state) are stored in a replay memory. During learning, experiences are sampled randomly from this replay memory. This breaks the correlation between consecutive learning updates.

**Fixed Q-Targets** - Instead of updating the Q-values based on the current Q-values (which are changing constantly), the Q-values are updated based on some fixed old Q-values. This is achieved by keeping a separate network, called the target network, that is a copy of the original network but is updated less frequently.