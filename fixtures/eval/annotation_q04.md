# Query 4: Markov decision process MDP reward

**Domain:** reinforcement_learning
**Query ID:** q_rl_002
**Candidates:** 44
**Grading:** 0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant

---

## [1/44] 220512944

| Field | Value |
|-------|-------|
| **Pages** | 8-8 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.835 |
| **Found In** | vector, hybrid, citations |
| **Chunk ID** | `58905d40-379f-47e9-a622-8d024e323a3b` |
| **YOUR GRADE** | ____ |

**Full Text (1148 chars):**

```
2.3.1 Stationary MDP
A stationary Markov decision process (MDP) is a tuple ( X , A , p, r, γ ) where X is a state space, A is an action space, p : X ×A → ∆ X is a transition kernel, r : X ×A → R is a reward function and γ ∈ (0 , 1) is a discount factor. Using action a when the current state is x leads to a new state distributed according to p ( ·| x, a ) ∈ ∆ X and produces a reward r ( x, a ) . The reward could be stochastic but to simplify the presentation, we consider that r is a deterministic function of the state and the action. A stationary policy π : X → ∆ A , x ↦→ π ( ·| x ) provides a distribution over actions for each state. The goal of the MDP is to find a policy π ∗ which maximizes the total return defined as the expected (discounted) sum of future rewards:
subject to:
with
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
where m 0 is an initial distribution whose choice does not influence the set of optimal policies.
Assuming the model is fully known to the agent, the problem can be solved using for instance dynamic programming. The state-action value function associated to a stationary policy π is defined as:
```

---

## [2/44] Barber D., Cemgil A.T., Chiappa S. (eds.) - Bayesian Time Series Models-Cambridge University Press

| Field | Value |
|-------|-------|
| **Pages** | 404-405 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.819 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `1c85d0e1-e7b4-412e-b142-3b6a8012290c` |
| **YOUR GRADE** | ____ |

**Full Text (1141 chars):**

```
18.2 Markov decision processes and likelihood maximisation
A Markov decision process ([15]) is a stochastic process on the random variables of state st , action at , and reward rt , as defined by the
:
```
initial state distribution P ( s 0 = s ) ; transition probability P ( st + 1 = s 0 j at = a ; st = s ) ; reward probability P ( rt = r j at = a ; st = s ) ; policy P ( at = a j st = s ; ) = : as
```
We assume the process to be stationary (none of these quantities explicitly depends on time) and call the expectation R ( a ; s ) = E f r j a ; s g = P r r P ( r j a ; s ) the reward function. In
Figure 18.1 Dynamic Bayesian network for an MDP. The x states denote the state variables, a the actions and r the rewards.
model-based reinforcement learning the transition and reward probabilities are estimated from experience (see, e.g., [1]). In Section 18.6.1 we discuss follow-up work that extends our framework to the model-free case. The random variables st and at can be discrete or continuous whereas the reward rt is a real number. Figure 18.1 displays the dynamic Bayesian network for an infinite-horizon Markov decision process.
```

---

## [3/44] 220512944

| Field | Value |
|-------|-------|
| **Pages** | 9-9 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.814 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `a26c33fe-f9d9-4051-9411-a76b4093ccc3` |
| **YOUR GRADE** | ____ |

**Full Text (968 chars):**

```
2.3.2 Finite Horizon MDP
One can also consider problems set with a finite time horizon. A fi nite-horizon Markov decision process (MDP) is a tuple ( X , A , p, r, N T ) where X is a state space, A is an action space, N T is a time horizon, p : { 0 , . . . , N T -1 } × X × A → P ( X ) is a transition kernel, and r : { 0 , . . . , N T } × X × A → R is a reward function. At time n , using action a when the current state is x leads to a new state distributed according to p n ( ·| x, a ) ∈ ∆ X and produces a reward r n ( x, a ) ∈ R . A policy π : { 0 , . . . , N T -1 } × X → P ( A ) , ( n, x ) ↦→ π n ( ·| x ) provides a distribution over actions for each state at time n . The goal of the MDP is to find a policy π ∗ which maximizes the total return defined as the expected (discounted) sum of future rewards:
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
where m 0 is an initial distribution whose choice does not influence the set of optimal policies.
```

---

## [4/44] Real-world humanoid locomotion with reinforcement learning

| Field | Value |
|-------|-------|
| **Pages** | 16-16 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.808 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `f7e78eae-de11-46b2-a45f-0f182f019945` |
| **YOUR GRADE** | ____ |

**Full Text (827 chars):**

```
Policy learning
Problem formulation. We formulate the control problem as a Markov Decision Process (MDP), which provides a mathematical framework for modeling discrete-time decision-making processes. The MDP comprises the following elements: a state space S , an action space A , a transition function P ( s t +1 | s t , a t ) that determines the probability of transitioning from state s t to s t +1 after taking action a t at time step t , and a scalar reward function R ( s t +1 | s t , a t ) , which assigns a scalar value to each state-action-state transition, serving as feedback to the agent on the quality of its actions. Our approach to solving the MDP problem is through Reinforcement Learning (RL), which aims to find an optimal policy that maximizes the expected cumulative reward over a finite or infinite horizon.
```

---

## [5/44] Uday Kamath Kevin Keenan Garrett Somers Sarah Sorenson - Large Language Models: A Deep Dive-Springer 2024

| Field | Value |
|-------|-------|
| **Pages** | 215-215 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.807 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `df5c2fb7-bc91-4485-95a4-d5cbff0708bd` |
| **YOUR GRADE** | ____ |

**Full Text (965 chars):**

```
Why is Reinforcement Learning used for LLM alignment
The Markov decision process (MDP) is a foundational mathematical framework for RL, as it models situations within a discrete-time, stochastic control process(Puterman, 1990).
In an MDP, as shown in Fig. 5.3, a decision-making entity, an agent, engages with its surrounding environment through a series of chronological interactions. The agent obtains a representation of the environmental state at every discrete time interval. Utilizing this representation, the agent proceeds to choose an appropriate action. Subsequently, the environment transitions to a new state, and the agent receives a reward for the consequences of the prior action. During this procedure, the agentɿs primary objective is to maximize the cumulative rewards obtained from executing actions in specific states.
Fig. 5.3: Markov Decision Process for Reinforcement Learning
There are several critical terms for understanding this approach.
```

---

## [6/44] Uday Kamath Kevin Keenan Garrett Somers Sarah Sorenson - Large Language Models: A Deep Dive-Springer 2024

| Field | Value |
|-------|-------|
| **Pages** | 476-477 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.805 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `f7a87106-2943-4515-b0ea-a09eeaafd6a3` |
| **YOUR GRADE** | ____ |

**Full Text (1399 chars):**

```
B.1 Markov Decision Process
The Markov Decision Process (MDP) is a foundational mathematical framework for RL, as it models situations within a discrete-time, stochastic control process.
In an MDP, as shown in Fig. B.1, a decision-making entity, an agent, engages with its surrounding environment through a series of chronological interactions. The agent obtains a representation of the environmental state at every discrete time interval. By utilizing this representation, the agent proceeds to choose an appropriate action. Subsequently, the environment transitions to a new state, and the agent receives a reward for the consequences of the prior action. During this procedure, the agentɿs primary objective is to maximize the cumulative rewards obtained from executing actions in specific states.
- State : A state represents the current situation or environment in an RL problem. A set of states denoted by ( S ) .
- Action : An action is a decision made by the agent that affects the state of the environment. Represented by A t , with the set of actions denoted by ( A ) . At each time step t , the agent receives some representation of the environmentɿs state S t . Based on this state, the agent selects an action A t . This gives us the state-action pair ( S t , A t ) . The next increment is t + 1 , and the environment is transitioned to
Fig. B.1: Detailed Markov Decision Process for RL
```

---

## [7/44] Speech and Language Processing [draft]

| Field | Value |
|-------|-------|
| **Pages** | 889-889 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.801 |
| **Found In** | hybrid |
| **Chunk ID** | `d25e564d-5d92-4e55-afd5-b4e9bd28fbbb` |
| **YOUR GRADE** | ____ |

**Full Text (872 chars):**

```
24.5.4 Generating Dialogue Acts: Confirmation and Rejection
A Markov decision process or MDP is characterized by a set of states S an agent can be in, a set of actions A the agent can take, and a reward r(a,s) that the agent receives for taking an action in a state. Given these factors, we can compute a policy π which specifies which action a the agent should take when in a given state s , so as to receive the best reward. To understand each of these components, we'll need to look at a tutorial example in which the state space is extremely reduced. Thus we'll return to the simple frame-and-slot world, looking at a pedagogical MDP implementation taken from Levin et al. (2000). Their tutorial example is a 'Day-and-Month' dialogue system, whose goal is to get correct values of day and month for a two-slot frame via the shortest possible interaction with the user.
```

---

## [8/44] rl_specialization_notebook_guide

| Field | Value |
|-------|-------|
| **Pages** | 27-28 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.790 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `519988d9-11fc-4a3f-9393-ab02471950ae` |
| **YOUR GRADE** | ____ |

**Full Text (1148 chars):**

```
2.3 Formal MDP Definition
Markov Decision Process (MDP) : A tuple ( S , A , p, r, γ ) where S is the state space, A is the action space, p is the transition dynamics, r is the reward function, and γ is the discount factor
An MDP is defined by five components:
1. S : finite set of states (plus terminal state s terminal for episodic tasks)
2. A ( s ) : finite set of actions available in state s
3. p ( s ′ , r | s, a ) : dynamics function-probability of transitioning to s ′ with reward r given state s and action a
4. r ( s, a ) = E [ R t +1 | S t = s, A t = a ] : expected reward function
5. γ ∈ [0 , 1] : discount factor (controls how much the agent values future vs immediate reward)
The dynamics function p captures everything about how the environment works:
<!-- formula-not-decoded -->
Pattern Markov violation test: if knowing the last k > 1 states improves prediction, the single-state representation is not Markov. Fix: stack k frames or add derived features (velocity, acceleration).
Interview OpenAI MLE: 'Write down the five components of an MDP.' Forgetting γ or conflating p with r is an immediate red flag.
From p , we can derive:
```

---

## [9/44] rl_specialization_notebook_guide

| Field | Value |
|-------|-------|
| **Pages** | 28-28 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.789 |
| **Found In** | vector |
| **Chunk ID** | `f2c8947a-ad47-49e5-939b-febab42a7568` |
| **YOUR GRADE** | ____ |

**Full Text (278 chars):**

```
2.3 Formal MDP Definition
- State-transition probabilities : p ( s ′ | s, a ) = ∑ p ( s ′ , r | s, a
- Expected reward : r ( s, a ) = ∑ r ∑ ′ p ( s ′ , r | s, a
- State-action-next-state reward : r ( s, a, s ′ ) = ∑ r r · p ( s ,r | s,a ) p ( s ′ | s,a )
```
r ) r s ) ′
```
```
```

---

## [10/44] rl_specialization_notebook_guide

| Field | Value |
|-------|-------|
| **Pages** | 25-25 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.786 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `8a0aaf2a-e64b-448a-bdb9-3620ebafe30c` |
| **YOUR GRADE** | ____ |

**Full Text (486 chars):**

```
Where We Are
In C1M1 we studied the bandit setting-a single state, choosing among actions with unknown reward distributions. Now we add states and time . The agent moves through an environment, observing states and collecting rewards over many steps. The Markov Decision Process (MDP) is the mathematical framework that makes this precise. Everything in reinforcement learning-value functions, Bellman equations, policy optimization-is built on top of the MDP formalism introduced here.
```

---

## [11/44] Grokking Deep Reinforcement Learning

| Field | Value |
|-------|-------|
| **Pages** | 87-87 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.784 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `f7388a86-e04a-4d57-be0a-a10ec3647498` |
| **YOUR GRADE** | ____ |

**Full Text (1245 chars):**

```
Summary
Okay. I know this chapter is heavy on new terms, but that's its intent. The best summary for this chapter is on the previous page, more specifically, the definition of an MDP. Take another look at the last two equations and try to remember what each letter means. Once you do so, you can be assured that you got what's necessary out of this chapter to proceed.
At the highest level, a reinforcement learning problem is about the interactions between an agent and the environment in which the agent exists. A large variety of issues can be modeled under this setting. The Markov decision process is a mathematical framework for representing complex decision-making problems under uncertainty.
Markov decision processes (MDPs) are composed of a set of system states, a set of per-state actions, a transition function, a reward signal, a horizon, a discount factor, and an initial state distribution. States describe the configuration of the environment. Actions allow agents to interact with the environment. The transition function tells how the environment evolves and reacts to the agent's actions. The reward signal encodes the goal to be achieved by the agent. The horizon and discount factor add a notion of time to the interactions.
```

---

## [12/44] Grokking_Deep_Reinforcement_Learning (11)

| Field | Value |
|-------|-------|
| **Pages** | 87-87 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.784 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `7b65741c-6550-4fc9-9f1c-04dc7bdcba58` |
| **YOUR GRADE** | ____ |

**Full Text (1245 chars):**

```
Summary
Okay. I know this chapter is heavy on new terms, but that's its intent. The best summary for this chapter is on the previous page, more specifically, the definition of an MDP. Take another look at the last two equations and try to remember what each letter means. Once you do so, you can be assured that you got what's necessary out of this chapter to proceed.
At the highest level, a reinforcement learning problem is about the interactions between an agent and the environment in which the agent exists. A large variety of issues can be modeled under this setting. The Markov decision process is a mathematical framework for representing complex decision-making problems under uncertainty.
Markov decision processes (MDPs) are composed of a set of system states, a set of per-state actions, a transition function, a reward signal, a horizon, a discount factor, and an initial state distribution. States describe the configuration of the environment. Actions allow agents to interact with the environment. The transition function tells how the environment evolves and reacts to the agent's actions. The reward signal encodes the goal to be achieved by the agent. The horizon and discount factor add a notion of time to the interactions.
```

---

## [13/44] pml2

| Field | Value |
|-------|-------|
| **Pages** | 1147-1147 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.783 |
| **Found In** | vector |
| **Chunk ID** | `fbe1a3c5-69b5-429c-9459-21b4d606eaba` |
| **YOUR GRADE** | ____ |

**Full Text (777 chars):**

```
36.4.4 The optimal solution
' approach of [DMKM22].)
<!-- formula-not-decoded -->
The observed reward at each step is then predicted to be
<!-- formula-not-decoded -->
We see that this is a special form of a (controlled) Markov decision process (Section 36.5) known as a belief-state MDP .
In the special case of context-free bandits with a finite number of arms, the optimal policy of this belief state MDP can be computed using dynamic programming (c.f., Section 36.6); the result can be represented as a table of action probabilities, π t ( a 1 , . . . , a K ) , for each step; this is known as the Gittins index [Git89]. However, computing the optimal policy for general contextual bandits is intractable [PT87], so we have to resort to approximations, as we discuss below.
```

---

## [14/44] Causal AI

| Field | Value |
|-------|-------|
| **Pages** | 461-461 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.782 |
| **Found In** | fts, vector, hybrid, citations |
| **Chunk ID** | `6f5023a7-d637-4811-87ca-f50f6ca61dfe` |
| **YOUR GRADE** | ____ |

**Full Text (987 chars):**

```
12.5.1 Connecting causality and Markov decision processes
RL typically casts a decision process as a Markov decision process (MDP). A canonical toy example of an MDP is a grid world, illustrated in figure 12.27.
Figure 12.27 presents a 3  4 grid world. An agent can act within this grid world with a fixed set of actions, moving up, down, left, and right. The agent wants to execute a set of  actions  that  deliver  it  to  the  upper-right  corner  {0,  3}, where it gains a reward of 100. The agent wants to avoid the middle-right square {1, 3}, where it has a reward of -100 (a loss of 100). Position {1, 1} contains an obstacle the agent cannot traverse.
We can think of it as a game. When the game starts, the agent 'spawns' randomly in one of the squares, except for
Figure 12.27 A simple grid world
{0, 3}, {1, 3}, and {1, 1}. When the agent moves into a goal square, the game ends. To win, the agent must navigate around the obstacle in {1, 1}, avoid {1, 3}, and reach {0, 3}.
```

---

## [15/44] Causal Artificial Intelligence: AI for Real-World Causal Understanding and Reasoning

| Field | Value |
|-------|-------|
| **Pages** | 461-461 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.782 |
| **Found In** | fts, vector, hybrid, citations |
| **Chunk ID** | `0282b70e-aecf-45b2-95ad-72c7e269deb7` |
| **YOUR GRADE** | ____ |

**Full Text (987 chars):**

```
12.5.1 Connecting causality and Markov decision processes
RL typically casts a decision process as a Markov decision process (MDP). A canonical toy example of an MDP is a grid world, illustrated in figure 12.27.
Figure 12.27 presents a 3  4 grid world. An agent can act within this grid world with a fixed set of actions, moving up, down, left, and right. The agent wants to execute a set of  actions  that  deliver  it  to  the  upper-right  corner  {0,  3}, where it gains a reward of 100. The agent wants to avoid the middle-right square {1, 3}, where it has a reward of -100 (a loss of 100). Position {1, 1} contains an obstacle the agent cannot traverse.
We can think of it as a game. When the game starts, the agent 'spawns' randomly in one of the squares, except for
Figure 12.27 A simple grid world
{0, 3}, {1, 3}, and {1, 1}. When the agent moves into a goal square, the game ends. To win, the agent must navigate around the obstacle in {1, 1}, avoid {1, 3}, and reach {0, 3}.
```

---

## [16/44] Causal Ai

| Field | Value |
|-------|-------|
| **Pages** | 461-461 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.782 |
| **Found In** | fts, vector, hybrid, citations |
| **Chunk ID** | `842103b0-3a27-4830-bc15-c03feaf31951` |
| **YOUR GRADE** | ____ |

**Full Text (987 chars):**

```
12.5.1 Connecting causality and Markov decision processes
RL typically casts a decision process as a Markov decision process (MDP). A canonical toy example of an MDP is a grid world, illustrated in figure 12.27.
Figure 12.27 presents a 3  4 grid world. An agent can act within this grid world with a fixed set of actions, moving up, down, left, and right. The agent wants to execute a set of  actions  that  deliver  it  to  the  upper-right  corner  {0,  3}, where it gains a reward of 100. The agent wants to avoid the middle-right square {1, 3}, where it has a reward of -100 (a loss of 100). Position {1, 1} contains an obstacle the agent cannot traverse.
We can think of it as a game. When the game starts, the agent 'spawns' randomly in one of the squares, except for
Figure 12.27 A simple grid world
{0, 3}, {1, 3}, and {1, 1}. When the agent moves into a goal square, the game ends. To win, the agent must navigate around the obstacle in {1, 1}, avoid {1, 3}, and reach {0, 3}.
```

---

## [17/44] Reinforcement Learning: An Introduction

| Field | Value |
|-------|-------|
| **Pages** | 90-90 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.782 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `1a091af5-ad53-4173-bc09-a4207b8026fe` |
| **YOUR GRADE** | ____ |

**Full Text (1254 chars):**

```
3.8 Summary
Let us summarize the elements of the reinforcement learning problem that we have presented in this chapter. Reinforcement learning is about learning from interaction how to behave in order to achieve a goal. The reinforcement learning agent and its environment interact over a sequence of discrete time steps. The specification of their interface defines a particular task: the actions are the choices made by the agent; the states are the basis for making the choices; and the rewards are the basis for evaluating the choices. Everything inside the agent is known and controllable. Its environment, on the other hand, is incompletely controllable and may or may not be completely known. A policy is a stochastic rule by which the agent selects actions as a function of states. The agent's objective is to maximize the amount of reward it receives over time.
When the reinforcement learning setup described above is formulated with well defined transition probabilities it constitutes a Markov decision process (MDP). A fi nite MDP is an MDP with finite state, action, and (as we formulate it here) reward sets. Much of the current theory of reinforcement learning is restricted to finite MDPs, but the methods and ideas apply more generally.
```

---

## [18/44] rl_specialization_notebook_guide

| Field | Value |
|-------|-------|
| **Pages** | 342-342 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.782 |
| **Found In** | vector |
| **Chunk ID** | `1ea07670-c3e8-4073-a837-2b697bc63e61` |
| **YOUR GRADE** | ____ |

**Full Text (594 chars):**

```
Time target: 5 minutes.
Consider a 3-state MDP: { s 1 , s 2 , s 3 } where s 3 is terminal. Actions: { L, R } . Transitions are deterministic. γ = 0 . 9 .
s 1, Action = L. s 1, Reward = +2. s 1, Next State = s 1 (self-loop). s 1, Action = R. s 1, Reward = +1. s 1, Next State = s 2. s 2, Action = L. s 2, Reward = +3. s 2, Next State = s 1. s 2, Action = R. s 2, Reward = +10. s 2, Next State = s 3
Starting from v 0 ( s 1 ) = 0 , v 0 ( s 2 ) = 0 , v 0 ( s 3 ) = 0 , trace one complete sweep of value iteration to find v 1 ( s 1 ) and v 1 ( s 2 ) . Then state the greedy policy after this sweep.
```

---

## [19/44] RLbook2018

| Field | Value |
|-------|-------|
| **Pages** | 90-90 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.781 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `f1c5efb0-9b2c-456b-bce1-719fd1e8e341` |
| **YOUR GRADE** | ____ |

**Full Text (1261 chars):**

```
3.8 Summary
Let us summarize the elements of the reinforcement learning problem that we have presented in this chapter. Reinforcement learning is about learning from interaction how to behave in order to achieve a goal. The reinforcement learning agent and its environment interact over a sequence of discrete time steps. The specification of their interface defines a particular task: the actions are the choices made by the agent; the states are the basis for making the choices; and the rewards are the basis for evaluating the choices. Everything inside the agent is completely known and controllable by the agent; everything outside is incompletely controllable but may or may not be completely known. A policy is a stochastic rule by which the agent selects actions as a function of states. The agent's objective is to maximize the amount of reward it receives over time.
When the reinforcement learning setup described above is formulated with well defined transition probabilities it constitutes a Markov decision process (MDP). A fi nite MDP is an MDP with finite state, action, and (as we formulate it here) reward sets. Much of the current theory of reinforcement learning is restricted to finite MDPs, but the methods and ideas apply more generally.
```

---

## [20/44] Reinforcement Learning An Introduction 2nd ed

| Field | Value |
|-------|-------|
| **Pages** | 90-90 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.781 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `f16efa21-c1c9-457f-be5c-c8a55d9f4042` |
| **YOUR GRADE** | ____ |

**Full Text (1261 chars):**

```
3.8 Summary
Let us summarize the elements of the reinforcement learning problem that we have presented in this chapter. Reinforcement learning is about learning from interaction how to behave in order to achieve a goal. The reinforcement learning agent and its environment interact over a sequence of discrete time steps. The specification of their interface defines a particular task: the actions are the choices made by the agent; the states are the basis for making the choices; and the rewards are the basis for evaluating the choices. Everything inside the agent is completely known and controllable by the agent; everything outside is incompletely controllable but may or may not be completely known. A policy is a stochastic rule by which the agent selects actions as a function of states. The agent's objective is to maximize the amount of reward it receives over time.
When the reinforcement learning setup described above is formulated with well defined transition probabilities it constitutes a Markov decision process (MDP). A fi nite MDP is an MDP with finite state, action, and (as we formulate it here) reward sets. Much of the current theory of reinforcement learning is restricted to finite MDPs, but the methods and ideas apply more generally.
```

---

## [21/44] Reinforcement Learning An Introduction 2nd ed

| Field | Value |
|-------|-------|
| **Pages** | 90-90 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.781 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `078c5db8-7c2f-4df9-9753-b1bebfbf4c60` |
| **YOUR GRADE** | ____ |

**Full Text (1261 chars):**

```
3.8 Summary
Let us summarize the elements of the reinforcement learning problem that we have presented in this chapter. Reinforcement learning is about learning from interaction how to behave in order to achieve a goal. The reinforcement learning agent and its environment interact over a sequence of discrete time steps. The specification of their interface defines a particular task: the actions are the choices made by the agent; the states are the basis for making the choices; and the rewards are the basis for evaluating the choices. Everything inside the agent is completely known and controllable by the agent; everything outside is incompletely controllable but may or may not be completely known. A policy is a stochastic rule by which the agent selects actions as a function of states. The agent's objective is to maximize the amount of reward it receives over time.
When the reinforcement learning setup described above is formulated with well defined transition probabilities it constitutes a Markov decision process (MDP). A fi nite MDP is an MDP with finite state, action, and (as we formulate it here) reward sets. Much of the current theory of reinforcement learning is restricted to finite MDPs, but the methods and ideas apply more generally.
```

---

## [22/44] pml2

| Field | Value |
|-------|-------|
| **Pages** | 1152-1153 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.779 |
| **Found In** | vector |
| **Chunk ID** | `69e3c2db-56f1-41fe-ac87-c61cef43a050` |
| **YOUR GRADE** | ____ |

**Full Text (939 chars):**

```
25 36.5.1 Basics
s t , a t ) , and r t ∼ p R ( s t , a t , s t +1 ) . Hence, under policy π , the probability of generating a trajectory τ of length T can be written explicitly as
39
40
41
42
<!-- formula-not-decoded -->
43 44 It is useful to define the reward function from the reward model p R , as the average immediate reward of taking action a in state s , with the next state marginalized:
45
46
47
<!-- formula-not-decoded -->
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
Figure 36.9: Illustration of a partially observable Markov decision process (POMDP) with hidden environment state s t which generates the observation x t , controlled by an agent with internal belief state b t which generates the action a t . The reward r t depends on s t and a t . Nodes in this graph represent random variables (circles) and decision variables (squares).
```

---

## [23/44] Deep Reinforcement Learning in Action

| Field | Value |
|-------|-------|
| **Pages** | 69-69 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.778 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `5fae1ef9-8e49-4245-a1aa-ca0c8cf921f9` |
| **YOUR GRADE** | ____ |

**Full Text (1341 chars):**

```
2.6 The Markov property
In our contextual bandit problem, our neural network led us to choose the best action given a state without reference to any other prior states. We just gave it the current state,  and  it  produced  the  expected  rewards  for  each  possible  action.  This  is  an important property in reinforcement learning called the Markov property . A game (or any other control task) that exhibits the Markov property is said to be a Markov decision process (MDP). With an MDP, the current state alone contains enough information to choose optimal actions to maximize future rewards. Modeling a control task as an MDP is a key concept in reinforcement learning.
The MDP model simplifies an RL problem dramatically, as we do not need to take into account all previous states or actions-we don't need to have memory, we just need to analyze the present situation. Hence, we always attempt to model a problem as  (at  least  approximately)  a  Markov  decision  processes.  The  card  game  Blackjack (also known as 21) is an MDP because we can play the game successfully just by knowing our current state (what cards we have, and the dealer's one face-up card).
To test your understanding of the Markov property, consider each control problem or decision task in the following list and see if it has the Markov property or not:
```

---

## [24/44] Deep_Reinforcement_Learning_in_Action (5)

| Field | Value |
|-------|-------|
| **Pages** | 69-69 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.778 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `18059b6b-5883-415d-a7b0-64594d964f24` |
| **YOUR GRADE** | ____ |

**Full Text (1341 chars):**

```
2.6 The Markov property
In our contextual bandit problem, our neural network led us to choose the best action given a state without reference to any other prior states. We just gave it the current state,  and  it  produced  the  expected  rewards  for  each  possible  action.  This  is  an important property in reinforcement learning called the Markov property . A game (or any other control task) that exhibits the Markov property is said to be a Markov decision process (MDP). With an MDP, the current state alone contains enough information to choose optimal actions to maximize future rewards. Modeling a control task as an MDP is a key concept in reinforcement learning.
The MDP model simplifies an RL problem dramatically, as we do not need to take into account all previous states or actions-we don't need to have memory, we just need to analyze the present situation. Hence, we always attempt to model a problem as  (at  least  approximately)  a  Markov  decision  processes.  The  card  game  Blackjack (also known as 21) is an MDP because we can play the game successfully just by knowing our current state (what cards we have, and the dealer's one face-up card).
To test your understanding of the Markov property, consider each control problem or decision task in the following list and see if it has the Markov property or not:
```

---

## [25/44] Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow

| Field | Value |
|-------|-------|
| **Pages** | 656-656 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.778 |
| **Found In** | fts, citations |
| **Chunk ID** | `6dfd228d-7b01-4638-8afe-075a429d80e4` |
| **YOUR GRADE** | ____ |

**Full Text (677 chars):**

```
Markov Decision Processes
Markov decision processes were first  described  in  the  1950s  by  Richard  Bellman. 12 They resemble Markov chains but with a twist: at each step, an agent can choose one of  several  possible  actions,  and  the  transition  probabilities  depend  on  the  chosen action. Moreover, some state transitions return some reward (positive or negative), and the agent's goal is to find a policy that will maximize reward over time.
For example, the MDP represented in Figure 18-8 has three states (represented by circles) and up to three possible discrete actions at each step (represented by diamonds).
Figure 18-8. Example of a Markov decision process
```

---

## [26/44] Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow

| Field | Value |
|-------|-------|
| **Pages** | 659-659 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.777 |
| **Found In** | vector |
| **Chunk ID** | `1649e15f-0096-4ad3-9908-66c9bdca4f39` |
| **YOUR GRADE** | ____ |

**Full Text (653 chars):**

```
Markov Decision Processes
```
>>> np.argmax(Q_values, axis=1) # optimal action for each state array([0, 0, 1])
```
This gives us the optimal policy for this MDP , when using a discount factor of 0.90: in state s 0 choose  action a 0 ;  in  state s 1 choose  action a 0 (i.e.,  stay  put);  and  in  state s 2 choose action a 1 (the only possible action). Interestingly, if we increase the discount factor to 0.95, the optimal policy changes: in state s 1 the best action becomes a 2 (go through the fire!). This makes sense because the more you value future rewards, the more you are willing to put up with some pain now for the promise of future bliss.
```

---

## [27/44] Derong Liu, Qinglai Wei, Ding Wang, Xiong Yang, Hongliang Li (auth.) - Adaptive Dynamic Programming with Applications in Optimal Control-Springer International Publish

| Field | Value |
|-------|-------|
| **Pages** | 50-50 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.777 |
| **Found In** | vector |
| **Chunk ID** | `81bb7bf0-5603-45ad-9c93-59ee79d19f61` |
| **YOUR GRADE** | ____ |

**Full Text (1645 chars):**

```
1.4 Related Books
and high-quality solutions to problems that involve making decisions in the presence of uncertainty. The book integrates the disciplines of Markov design processes, mathematical programming, simulation, and statistics, to demonstrate how to successfully model and solve a wide range of real-life problems using the idea of approximate dynamic programming (ADP). It starts with a simple introduction using a discrete representation of states. The background of dynamic programming and Markov decision processes is given, and meanwhile the phenomenon of the curse of dimensionality is discussed. A detailed description on how to model a dynamic program and some important algorithmic strategies are presented next. The most important dimensions of ADP, i.e., modeling real applications, the interface with stochastic approximation methods, techniques for approximating general value functions, and a more in-depth presentation of ADP algorithms for finite- and infinite-horizon applications are provided, respectively. Several specific problems, including information acquisition and resource allocation, and algorithms that arise in this setting are introduced in the third part. The well-known exploration versus exploitation problem is proposed to discuss how to visit a state. These applications bring out the richness of ADP techniques. In summary, it models complex, high-dimensional problems in a natural and practical way; introduces and emphasizes the power of estimating a value function around the post-decision state; and presents a thorough discussion of recursive estimation. It is shown in this book that ADP is an
```

---

## [28/44] Barber D., Cemgil A.T., Chiappa S. (eds.) - Bayesian Time Series Models-Cambridge University Press

| Field | Value |
|-------|-------|
| **Pages** | 407-408 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.772 |
| **Found In** | vector |
| **Chunk ID** | `781a30f2-6c48-4832-b4d8-bb057eb450c1` |
| **YOUR GRADE** | ____ |

**Full Text (1182 chars):**

```
18.2 Markov decision processes and likelihood maximisation
Proof. Let H be some horizon for which we later take the limit to 1 . We can rewrite the value function of the original MDP as
<!-- formula-not-decoded -->
In the second line we pulled the summation over T to the front. Note that the second and third line are really di erent: the product is taken to the limit T instead of H since we eliminated the variables aT + 1: H ; sT + 1: H with the summation. The last expression has already the form of a mixture model, where T is the mixture variable, T is the mixture weight ( P ( T ) = T (1 ) the normalized geometric prior), and the last term is the expected reward in the final time slice of a fi nite-time MDP of length T (since the expectation is taken over a 0: T ; s 0: T j ).
The likelihood in our mixture model can be written as
<!-- formula-not-decoded -->
In Appendix 18.A we remark on the following points in some more detail:
- (i) the interpretation of the mixture with death probabilities of the agent,
- (ii) the di erence between the models w.r.t. the correlation between rewards,
- (iii) approaches to consider exponentiated rewards as observation likelihoods.
```

---

## [29/44] Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects

| Field | Value |
|-------|-------|
| **Pages** | 3-3 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.769 |
| **Found In** | fts |
| **Chunk ID** | `b1370aba-7725-457f-ab7b-752f4b9be8ef` |
| **YOUR GRADE** | ____ |

**Full Text (948 chars):**

```
3.2 Q-Learning
Q-learning (Watkins and Dayan, 1992) is a reinforcement learning (RL) algorithm that allows agent to explore environment and simultaneously compute maximum reward paths.
An RL problem is formalized as an Markov Decision Process (MDP). A MDP is defined by a four tuple ( S , A , T, R ) , where S is the state space, A is the action space, T : S×A → S is the system dynamics and R : S → R is the reward yielded on a execution of an action. The objective of a typical RL problem is to maximize the expected cumulative reward over time, called the returns R t = ∑ T t ′ = t r t ′ .
Q-learning works by maintaining an action-value function Q : S × A → R which is defined as the expected return
Q π ( s t , a t ) = E π [ R t ] from a given state-action pair. The Qlearning algorithm works by updating the Q -function using the Bellman equation for every transition from s to s ′ on action a yielding reward r ,
<!-- formula-not-decoded -->
```

---

## [30/44] pml2

| Field | Value |
|-------|-------|
| **Pages** | 31-31 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.769 |
| **Found In** | vector |
| **Chunk ID** | `49f6f9b7-016c-4a12-bf8a-5bfc99a2b23c` |
| **YOUR GRADE** | ____ |

**Full Text (932 chars):**

```
28.1 Introduction 919 28.2 Overview of Part V 920
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 36.2.1 Example: oil wildcatter 1099 36.2.2 Information arcs 1100 36.2.3 Value of information 1101 36.2.4 Computing the optimal policy 1102 36.3 A/B testing 1102 36.3.1 A Bayesian approach 1103 36.3.2 Example 1106 36.4 Contextual bandits 1107 36.4.1 Types of bandit 1107 36.4.2 Applications 1109 36.4.3 Exploration-exploitation tradeoff 1109 36.4.4 The optimal solution 1109 36.4.5 Upper confidence bounds (UCB) 1111 36.4.6 Thompson sampling 1113 36.4.7 Regret 1114 36.5 Markov decision problems 1115 36.5.1 Basics 1116 36.5.2 Partially observed MDPs 1117 36.5.3 Episodes and returns 1118 36.5.4 Value functions 1118 36.5.5 Optimal value functions and policies 1119 36.6 Planning in an MDP 1120 36.6.1 Value iteration 1121 36.6.2 Policy iteration 1122
```

---

## [31/44] pml2

| Field | Value |
|-------|-------|
| **Pages** | 1165-1165 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.767 |
| **Found In** | vector |
| **Chunk ID** | `93fd8cc6-d61c-41b9-9dc1-fe89177c0735` |
| **YOUR GRADE** | ____ |

**Full Text (943 chars):**

```
37.1.5.4 Optimal solution using Bayes-adaptive MDPs
The Bayes optimal solution to the exploration-exploitation tradeoff can be computed by formulating the problem as a special kind of POMDP known as a Bayes-adaptive MDP or BAMDP [Duf02]. This extends the Gittins index approach in Section 36.4.4 to the MDP setting.
In particular, a BAMDP has a belief state space, B , representing uncertainty about the reward model p R ( r | s, a, s ′ ) and transition model p T ( s ′ | s, a ) . The transition model on this augmented MDP can be written as follows:
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
where E b t [ T ( s t +1 | s t , a t )] is the posterior predictive distribution over next states, and p ( R,T | h t +1 ) is the new belief state given h t +1 = ( s 1: t +1 , a 1: t +1 , r 1: t +1 ) , which can be computed using Bayes rule. Similarly, the reward function for the augmented MDP is given by
<!-- formula-not-decoded -->
```

---

## [32/44] book2

| Field | Value |
|-------|-------|
| **Pages** | 1183-1184 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.767 |
| **Found In** | vector |
| **Chunk ID** | `5038d174-6481-447b-91cd-1bc73b009693` |
| **YOUR GRADE** | ____ |

**Full Text (942 chars):**

```
35.1.5.4 Optimal solution using Bayes-adaptive MDPs
The Bayes optimal solution to the exploration-exploitation tradeoff can be computed by formulating the problem as a special kind of POMDP known as a Bayes-adaptive MDP or BAMDP [Duf02]. This extends the Gittins index approach in Section 34.4.4 to the MDP setting.
In particular, a BAMDP has a belief state space, B , representing uncertainty about the reward model p R ( r | s, a, s ′ ) and transition model p ( s ′ | s, a ) . The transition model on this augmented MDP can be written as follows:
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
where E b t [ T ( s t +1 | s t , a t )] is the posterior predictive distribution over next states, and p ( R,T | h t +1 ) is the new belief state given h t +1 = ( s 1: t +1 , a 1: t +1 , r 1: t +1 ) , which can be computed using Bayes' rule.
Similarly, the reward function for the augmented MDP is given by
<!-- formula-not-decoded -->
```

---

## [33/44] pml2

| Field | Value |
|-------|-------|
| **Pages** | 1153-1154 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.766 |
| **Found In** | vector |
| **Chunk ID** | `fd8c97f6-b404-4e11-ba36-2c4395e37603` |
| **YOUR GRADE** | ____ |

**Full Text (1095 chars):**

```
36.5.2 Partially observed MDPs
An important generalization of the MDP framework relaxes the assumption that the agent sees the hidden world state s t directly; instead we assume it only sees a potentially noisy observation generated from the hidden state, x t ∼ p ( ·| s t , a t ) . The resulting model is called a partially observable Markov decision process or POMDP (pronounced 'pom-dee-pee'). Now the agent's policy is a mapping from all the available data to actions, a t ∼ π ( D 1: t -1 , x t ) , D t = ( x t , a t , r t ) . See Figure 36.9 for an illustration. MDPs are a special case where x t = s t .
In general, POMDPs are much harder to solve than MDPs. A common approximation is to use the last several observed inputs, say x t -h : t for history of size h , as a proxy for the hidden state, and
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
<!-- formula-not-decoded -->
24 25 26 G t is sometimes called the reward-to-go . For episodic tasks that terminate at time T , we define G t = 0 for t ≥ T . Clearly, the return satisfies the following recursive relationship:
27
```

---

## [34/44] rl_specialization_notebook_guide

| Field | Value |
|-------|-------|
| **Pages** | 186-186 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.764 |
| **Found In** | vector |
| **Chunk ID** | `74a9bc08-27e8-4d60-9373-dbc883ebbd4f` |
| **YOUR GRADE** | ____ |

**Full Text (602 chars):**

```
Problem 12.2 [TD(0) vs TD(0.9) vs MC on a Delayed-Reward Chain]
Consider a chain MDP with 10 states { S 1 , . . . , S 10 } . Theagent always moves right: S i → S i +1 . Rewards are 0 for all transitions except the last: R 10 = +1 (upon entering terminal state). γ = 1 . All initial value estimates are ˆ v ( S i ) = 0 .
- (a) After one episode, which states have their value estimates updated by TD(0)? By TD(0.9)? By MC?
- (b) Compute the update to ˆ v ( S 9 ) and ˆ v ( S 1 ) for each method (use α = 0 . 1 ).
- (c) How many episodes does TD(0) need before ˆ v ( S 1 ) > 0 ? How does TD(0.9) compare?
```

---

## [35/44] Grokking Deep Reinforcement Learning

| Field | Value |
|-------|-------|
| **Pages** | 250-252 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.763 |
| **Found In** | vector |
| **Chunk ID** | `7cbb50d9-8831-4d72-8fba-678fd74639d8` |
| **YOUR GRADE** | ____ |

**Full Text (1108 chars):**

```
boil iT Down
axis=1))}[s] return Q, V, pi, Q_track, pi_track (21) Notice we are now using a max_trajectory_depth variable, but are still planning. (22) We still check for the Q-function to have any difference, so it's worth our compute. (23) Select the action either on-policy or off-policy (using the greedy policy). (24) If we haven't experienced the transition, planning would be a mess, so break out. (25) Otherwise, we get the probabilities of next_state and sample the model accordingly. (26) Then, get the reward as prescribed by the reward-signal model. (27) And continue updating the Q-function as if with real experience. (28) Notice here we update the state variable right before we loop and continue the on-policy planning steps. (29) Outside the planning loop, we restore the state, and continue real interaction steps. (30) Everything else as usual
```
In  chapter  2,  we  developed  the  MDP  for  an  environment  called  frozen  lake  (FL).  As  you remember,  FL  is  a  simple  grid-world  (GW)  environment.  It  has  discrete  state  and  action spaces, with 16 states and four actions.
```

---

## [36/44] Grokking_Deep_Reinforcement_Learning (11)

| Field | Value |
|-------|-------|
| **Pages** | 250-252 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.763 |
| **Found In** | vector |
| **Chunk ID** | `500aca8f-72a4-4dd1-9bcf-758ce1af7d74` |
| **YOUR GRADE** | ____ |

**Full Text (1108 chars):**

```
boil iT Down
axis=1))}[s] return Q, V, pi, Q_track, pi_track (21) Notice we are now using a max_trajectory_depth variable, but are still planning. (22) We still check for the Q-function to have any difference, so it's worth our compute. (23) Select the action either on-policy or off-policy (using the greedy policy). (24) If we haven't experienced the transition, planning would be a mess, so break out. (25) Otherwise, we get the probabilities of next_state and sample the model accordingly. (26) Then, get the reward as prescribed by the reward-signal model. (27) And continue updating the Q-function as if with real experience. (28) Notice here we update the state variable right before we loop and continue the on-policy planning steps. (29) Outside the planning loop, we restore the state, and continue real interaction steps. (30) Everything else as usual
```
In  chapter  2,  we  developed  the  MDP  for  an  environment  called  frozen  lake  (FL).  As  you remember,  FL  is  a  simple  grid-world  (GW)  environment.  It  has  discrete  state  and  action spaces, with 16 states and four actions.
```

---

## [37/44] rl_specialization_notebook_guide

| Field | Value |
|-------|-------|
| **Pages** | 37-37 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.762 |
| **Found In** | vector |
| **Chunk ID** | `7e67ec17-3702-4b5a-8e7f-1f9da17e7022` |
| **YOUR GRADE** | ____ |

**Full Text (707 chars):**

```
Self-Check: C1M2
1. Draw the agent-environment interaction loop and label all signals with correct time indices.
2. State the Markov property in one sentence. Give an example of a state representation that violates it and how to fix it.
3. Write down the five components of an MDP.
4. Derive the recursive return identity G t = R t +1 + γ G t +1 from the definition of G t .
5. What is the maximum possible return for a continuing task with constant reward r = 1 and γ = 0 . 95 ?
6. Why must γ < 1 for continuing tasks but not for episodic tasks?
7. What is an absorbing state and why does Sutton & Barto introduce it?
8. Given rewards [3 , 1 , -2 , 4] and γ = 0 . 5 , compute G 0 using the backward method.
```

---

## [38/44] Barber D., Cemgil A.T., Chiappa S. (eds.) - Bayesian Time Series Models-Cambridge University Press

| Field | Value |
|-------|-------|
| **Pages** | 419-419 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.762 |
| **Found In** | vector |
| **Chunk ID** | `0b943194-7c69-46a0-90a4-31ddc20aa1a5` |
| **YOUR GRADE** | ____ |

**Full Text (603 chars):**

```
18.5 Application to POMDPs
Astationary, partially observable Markov decision process (POMDP, see e.g. [14]) is given by four time-independent probability functions,
the initial world state distribution, 1 = P ( s 0 = s ) ;. the world state transitions, 1 = P ( s t + 1 = s 0 j a t = a ; s t = s ) ;. the observation probabilities, 1 = P ( y t = y j s t = s ) ;. the reward probabilities, 1 = P ( r t = r j a t = a ; s t = s ) :
These functions are considered known. We assume the world states, actions and observations ( st ; yt ; at ) are discrete random variables while the reward rt is a real number.
```

---

## [39/44] Grokking Deep Reinforcement Learning

| Field | Value |
|-------|-------|
| **Pages** | 83-83 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.746 |
| **Found In** | fts |
| **Chunk ID** | `19743b26-7d70-4f7d-a02b-7e2d0711af46` |
| **YOUR GRADE** | ____ |

**Full Text (1249 chars):**

```
Extensions to MDPs
There are many extensions to the MDP framework, as we've discussed. They allow us to target slightly different types of RL problems. The following list isn't comprehensive, but it should give you an idea of how large the field is. Know that the acronym MDPs is often used to refer to all types of MDPs. We're currently looking only at the tip of the iceberg:
- Partially observable Markov decision process (POMDP): When the agent cannot fully observe the environment state
- Factored Markov decision process (FMDP): Allows the representation of the transition and reward function more compactly so that we can represent large MDPs
- Continuous [Time|Action|State] Markov decision process: When either time, action, state or any combination of them are continuous
- Relational Markov decision process (RMDP): Allows the combination of probabilistic and relational knowledge
- Semi-Markov decision process (SMDP): Allows the inclusion of abstract actions that can take multiple time steps to complete
- Multi-agent Markov decision process (MMDP): Allows the inclusion of multiple agents in the same environment
- Decentralized Markov decision process (Dec-MDP): Allows for multiple agents to collaborate and maximize a common reward
```

---

## [40/44] Grokking_Deep_Reinforcement_Learning (11)

| Field | Value |
|-------|-------|
| **Pages** | 83-83 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.746 |
| **Found In** | fts |
| **Chunk ID** | `9364a27a-13e3-4edd-84c5-c976927b22f5` |
| **YOUR GRADE** | ____ |

**Full Text (1249 chars):**

```
Extensions to MDPs
There are many extensions to the MDP framework, as we've discussed. They allow us to target slightly different types of RL problems. The following list isn't comprehensive, but it should give you an idea of how large the field is. Know that the acronym MDPs is often used to refer to all types of MDPs. We're currently looking only at the tip of the iceberg:
- Partially observable Markov decision process (POMDP): When the agent cannot fully observe the environment state
- Factored Markov decision process (FMDP): Allows the representation of the transition and reward function more compactly so that we can represent large MDPs
- Continuous [Time|Action|State] Markov decision process: When either time, action, state or any combination of them are continuous
- Relational Markov decision process (RMDP): Allows the combination of probabilistic and relational knowledge
- Semi-Markov decision process (SMDP): Allows the inclusion of abstract actions that can take multiple time steps to complete
- Multi-agent Markov decision process (MMDP): Allows the inclusion of multiple agents in the same environment
- Decentralized Markov decision process (Dec-MDP): Allows for multiple agents to collaborate and maximize a common reward
```

---

## [41/44] Reinforcement Learning: An Introduction

| Field | Value |
|-------|-------|
| **Pages** | 543-543 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.743 |
| **Found In** | fts |
| **Chunk ID** | `083c5b23-2325-42a0-9f81-596b5f40dfb4` |
| **YOUR GRADE** | ____ |

**Full Text (972 chars):**

```
Index
Law of Effect, 15-16 , 45, 343, 358-361, 417
learning automata, 18
Least Mean Square (LMS) algorithm, 279, 301
Least-Squares TD (LSTD), 228-229
linear function approx., 204-209 , 266-269
linear programming, 87, 90
local and global optima, 200
Markov decision process (MDP), 2, 14, 47-71
Markov property, 49 , 115, 465-468
Markov reward process (MRP), 125
maximization bias, 134-136
maximum-likelihood estimate, 128
MC,
see
Monte Carlo methods
Mean Square
Bellman Error, BE, 268
Projected Bellman Error, PBE, 269
Return Error, RE, 275
TD Error, TDE, 270
Value Error, VE, 199-200
memory-based function approx., 230-232
Michie, Donald, 17 , 71, 117
Minsky, Marvin, 16 , 17, 20, 89
model of the environment, 7, 159
model-based and model-free methods, 7, 159 in animal learning, 363-368
model-based reinforcement learning, 159-193 in neuroscience, 407-409
Monte Carlo methods, 91-117
first- and every-visit MC, 92
first-visit MC control, 101
first-visit MC prediction, 92
```

---

## [42/44] Reinforcement Learning An Introduction 2nd ed

| Field | Value |
|-------|-------|
| **Pages** | 543-543 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.740 |
| **Found In** | fts |
| **Chunk ID** | `e7746ca4-6252-4943-a63d-3951aee82e07` |
| **YOUR GRADE** | ____ |

**Full Text (973 chars):**

```
Index
Law of Effect, 15-16 , 45, 342, 358-361, 417
learning automata, 18
Least Mean Square (LMS) algorithm, 279, 301
Least-Squares TD (LSTD), 228-229
linear function approx., 204-209 , 266-269
linear programming, 87, 90
local and global optima, 200
Markov decision process (MDP), 2, 14, 47-71
Markov property, 49 , 115, 465-468
Markov reward process (MRP), 125
maximization bias, 134-136
maximum-likelihood estimate, 128
MC, see Monte Carlo methods
Mean Squared
Bellman Error, BE, 268
Projected Bellman Error, PBE, 269
Return Error, RE, 275
TD Error, TDE, 270
Value Error, VE, 199-200
memory-based function approx., 230-232
Michie, Donald, 17 , 71, 116
Minsky, Marvin, 16 , 17, 20, 89
model of the environment, 7, 159
model-based and model-free methods, 7, 159 in animal learning, 363-368
model-based reinforcement learning, 159-193 in neuroscience, 407-409
Monte Carlo methods, 91-117
first- and every-visit MC, 92
first-visit MC control, 101
first-visit MC prediction, 92
```

---

## [43/44] Reinforcement Learning An Introduction 2nd ed

| Field | Value |
|-------|-------|
| **Pages** | 543-543 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.740 |
| **Found In** | fts |
| **Chunk ID** | `f9b88889-0836-40db-add4-4c12f8188b7b` |
| **YOUR GRADE** | ____ |

**Full Text (973 chars):**

```
Index
Law of Effect, 15-16 , 45, 342, 358-361, 417
learning automata, 18
Least Mean Square (LMS) algorithm, 279, 301
Least-Squares TD (LSTD), 228-229
linear function approx., 204-209 , 266-269
linear programming, 87, 90
local and global optima, 200
Markov decision process (MDP), 2, 14, 47-71
Markov property, 49 , 115, 465-468
Markov reward process (MRP), 125
maximization bias, 134-136
maximum-likelihood estimate, 128
MC, see Monte Carlo methods
Mean Squared
Bellman Error, BE, 268
Projected Bellman Error, PBE, 269
Return Error, RE, 275
TD Error, TDE, 270
Value Error, VE, 199-200
memory-based function approx., 230-232
Michie, Donald, 17 , 71, 116
Minsky, Marvin, 16 , 17, 20, 89
model of the environment, 7, 159
model-based and model-free methods, 7, 159 in animal learning, 363-368
model-based reinforcement learning, 159-193 in neuroscience, 407-409
Monte Carlo methods, 91-117
first- and every-visit MC, 92
first-visit MC control, 101
first-visit MC prediction, 92
```

---

## [44/44] RLbook2018

| Field | Value |
|-------|-------|
| **Pages** | 543-543 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.739 |
| **Found In** | fts |
| **Chunk ID** | `d880509f-6ccf-4ae2-b432-27b08f3dc16e` |
| **YOUR GRADE** | ____ |

**Full Text (928 chars):**

```
Index
learning automata, 18
Least Mean Square (LMS) algorithm, 279, 301
Least-Squares TD (LSTD), 228-229
linear function approx., 204-209 , 266-269
linear programming, 87, 90
local and global optima, 200
Markov decision process (MDP), 2, 14, 47-71
Markov property, 49 , 115, 465-468
Markov reward process (MRP), 125
maximization bias, 134-136
maximum-likelihood estimate, 128
MC, see Monte Carlo methods
Mean Squared
Bellman Error, BE, 268
Projected Bellman Error, PBE, 269
Return Error, RE, 275
TD Error, TDE, 270
Value Error, VE, 199-200
memory-based function approx., 230-232
Michie, Donald, 17 , 71, 116
Minsky, Marvin, 16 , 17, 20, 89
model of the environment, 7, 159
model-based and model-free methods, 7, 159 in animal learning, 363-368
model-based reinforcement learning, 159-193 in neuroscience, 407-409
Monte Carlo methods, 91-117
first- and every-visit MC, 92
first-visit MC control, 101
first-visit MC prediction, 92
```

---
