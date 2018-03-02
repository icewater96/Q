
# Deep $Q$-learning

In this notebook, we'll build a neural network that can learn to play games through reinforcement learning. More specifically, we'll use $Q$-learning to train an agent to play a game called [Cart-Pole](https://gym.openai.com/envs/CartPole-v0). In this game, a freely swinging pole is attached to a cart. The cart can move to the left and right, and the goal is to keep the pole upright as long as possible.

![Cart-Pole](assets/cart-pole.jpg)

We can simulate this game using [OpenAI Gym](https://github.com/openai/gym). First, let's check out how OpenAI Gym works. Then, we'll get into training an agent to play the Cart-Pole game.


```python
import gym
import numpy as np

# Create the Cart-Pole game environment
env = gym.make('CartPole-v1')

# Number of possible actions
print('Number of possible actions:', env.action_space.n)
print(env.observation_space)
```

    Number of possible actions: 2
    Box(4,)


We interact with the simulation through `env`.  You can see how many actions are possible from `env.action_space.n`, and to get a random action you can use `env.action_space.sample()`.  Passing in an action as an integer to `env.step` will generate the next step in the simulation.  This is general to all Gym games. 

In the Cart-Pole game, there are two possible actions, moving the cart left or right. So there are two actions we can take, encoded as 0 and 1.

Run the code below to interact with the environment.


```python
actions = [] # actions that the agent selects
rewards = [] # obtained rewards
state = env.reset()

print(state)
while True:
    action = env.action_space.sample()  # choose a random action
    state, reward, done, _ = env.step(action) 
    rewards.append(reward)
    actions.append(action)
    if done:
        break
```

    [ 0.04642058 -0.01328033 -0.01678839  0.00454185]


We can look at the actions and rewards:


```python
print('Actions:', actions)
print('Rewards:', rewards)
```

    Actions: [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    Rewards: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


The game resets after the pole has fallen past a certain angle. For each step while the game is running, it returns a reward of 1.0. The longer the game runs, the more reward we get. Then, our network's goal is to maximize the reward by keeping the pole vertical. It will do this by moving the cart to the left and the right.

## $Q$-Network

To keep track of the action values, we'll use a neural network that accepts a state $s$ as input.  The output will be $Q$-values for each available action $a$ (i.e., the output is **all** action values $Q(s,a)$ _corresponding to the input state $s$_).

<img src="assets/q-network.png" width=550px>

For this Cart-Pole game, the state has four values: the position and velocity of the cart, and the position and velocity of the pole.  Thus, the neural network has **four inputs**, one for each value in the state, and **two outputs**, one for each possible action. 

As explored in the lesson, to get the training target, we'll first use the context provided by the state $s$ to choose an action $a$, then simulate the game using that action. This will get us the next state, $s'$, and the reward $r$. With that, we can calculate $\hat{Q}(s,a) = r + \gamma \max_{a'}{Q(s', a')}$.  Then we update the weights by minimizing $(\hat{Q}(s,a) - Q(s,a))^2$. 

Below is one implementation of the $Q$-network. It uses two fully connected layers with ReLU activations. Two seems to be good enough, three might be better. Feel free to try it out.


```python
import tensorflow as tf

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, 
                 action_size=2, hidden_size=10, 
                 name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)
            
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            
            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, 
                                                            activation_fn=None)
            
            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
```

## Experience replay

Reinforcement learning algorithms can have stability issues due to correlations between states. To reduce correlations when training, we can store the agent's experiences and later draw a random mini-batch of those experiences to train on. 

Here, we'll create a `Memory` object that will store our experiences, our transitions $<s, a, r, s'>$. This memory will have a maximum capacity, so we can keep newer experiences in memory while getting rid of older experiences. Then, we'll sample a random mini-batch of transitions $<s, a, r, s'>$ and train on those.

Below, I've implemented a `Memory` object. If you're unfamiliar with `deque`, this is a double-ended queue. You can think of it like a tube open on both sides. You can put objects in either side of the tube. But if it's full, adding anything more will push an object out the other side. This is a great data structure to use for the memory buffer.


```python
from collections import deque

class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]
```

## $Q$-Learning training algorithm

We will use the below algorithm to train the network.  For this game, the goal is to keep the pole upright for 195 frames. So we can start a new episode once meeting that goal. The game ends if the pole tilts over too far, or if the cart moves too far the left or right. When a game ends, we'll start a new episode. Now, to train the agent:

* Initialize the memory $D$
* Initialize the action-value network $Q$ with random weights
* **For** episode $\leftarrow 1$ **to** $M$ **do**
  * Observe $s_0$
  * **For** $t \leftarrow 0$ **to** $T-1$ **do**
     * With probability $\epsilon$ select a random action $a_t$, otherwise select $a_t = \mathrm{argmax}_a Q(s_t,a)$
     * Execute action $a_t$ in simulator and observe reward $r_{t+1}$ and new state $s_{t+1}$
     * Store transition $<s_t, a_t, r_{t+1}, s_{t+1}>$ in memory $D$
     * Sample random mini-batch from $D$: $<s_j, a_j, r_j, s'_j>$
     * Set $\hat{Q}_j = r_j$ if the episode ends at $j+1$, otherwise set $\hat{Q}_j = r_j + \gamma \max_{a'}{Q(s'_j, a')}$
     * Make a gradient descent step with loss $(\hat{Q}_j - Q(s_j, a_j))^2$
  * **endfor**
* **endfor**

You are welcome (and encouraged!) to take the time to extend this code to implement some of the improvements that we discussed in the lesson, to include fixed $Q$ targets, double DQNs, prioritized replay, and/or dueling networks.

## Hyperparameters

One of the more difficult aspects of reinforcement learning is the large number of hyperparameters. Not only are we tuning the network, but we're tuning the simulation.


```python
train_episodes = 1000          # max number of episodes to learn from
max_steps = 200                # max steps in an episode
gamma = 0.99                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Network parameters
hidden_size = 64               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 20                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory
```


```python
tf.reset_default_graph()
mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate)
```

## Populate the experience memory

Here we re-initialize the simulation and pre-populate the memory. The agent is taking random actions and storing the transitions in memory. This will help the agent with exploring the game.


```python
# Initialize the simulation
env.reset()
# Take one random step to get the pole and cart moving
state, reward, done, _ = env.step(env.action_space.sample())

memory = Memory(max_size=memory_size)

# Make a bunch of random actions and store the experiences
for ii in range(pretrain_length):

    # Make a random action
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        
        # Start new episode
        env.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = env.step(env.action_space.sample())
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        state = next_state
```

## Training

Below we'll train our agent.


```python
# Now train with experiences
saver = tf.train.Saver()
rewards_list = []
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    
    step = 0
    for ep in range(1, train_episodes):
        total_reward = 0
        t = 0
        while t < max_steps:
            step += 1
            # Uncomment this next line to watch the training
            # env.render() 
            
            # Explore or Exploit
            explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step) 
            if explore_p > np.random.rand():
                # Make a random action
                action = env.action_space.sample()
            else:
                # Get action from Q-network
                feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                Qs = sess.run(mainQN.output, feed_dict=feed)
                action = np.argmax(Qs)
            
            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)
    
            total_reward += reward
            
            if done:
                # the episode ends so no next state
                next_state = np.zeros(state.shape)
                t = max_steps
                
                print('Episode: {}'.format(ep),
                      'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_p))
                rewards_list.append((ep, total_reward))
                
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                
                # Start new episode
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                state = next_state
                t += 1
            
            # Sample mini-batch from memory
            batch = memory.sample(batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])
            
            # Train network
            target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})
            
            # Set target_Qs to 0 for states where episode ends
            episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            target_Qs[episode_ends] = (0, 0)
            
            targets = rewards + gamma * np.max(target_Qs, axis=1)

            loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                feed_dict={mainQN.inputs_: states,
                                           mainQN.targetQs_: targets,
                                           mainQN.actions_: actions})
        
    saver.save(sess, "checkpoints/cartpole.ckpt")
```

    Episode: 1 Total reward: 9.0 Training loss: 1.0825 Explore P: 0.9991
    Episode: 2 Total reward: 10.0 Training loss: 1.0775 Explore P: 0.9981
    Episode: 3 Total reward: 31.0 Training loss: 1.0766 Explore P: 0.9951
    Episode: 4 Total reward: 27.0 Training loss: 1.1361 Explore P: 0.9924
    Episode: 5 Total reward: 36.0 Training loss: 1.0239 Explore P: 0.9889
    Episode: 6 Total reward: 22.0 Training loss: 1.1038 Explore P: 0.9867
    Episode: 7 Total reward: 33.0 Training loss: 1.1452 Explore P: 0.9835
    Episode: 8 Total reward: 9.0 Training loss: 1.0819 Explore P: 0.9826
    Episode: 9 Total reward: 19.0 Training loss: 1.0111 Explore P: 0.9808
    Episode: 10 Total reward: 13.0 Training loss: 1.2561 Explore P: 0.9795
    Episode: 11 Total reward: 18.0 Training loss: 1.1711 Explore P: 0.9778
    Episode: 12 Total reward: 25.0 Training loss: 1.1796 Explore P: 0.9754
    Episode: 13 Total reward: 31.0 Training loss: 1.3363 Explore P: 0.9724
    Episode: 14 Total reward: 11.0 Training loss: 1.1527 Explore P: 0.9713
    Episode: 15 Total reward: 12.0 Training loss: 1.3378 Explore P: 0.9702
    Episode: 16 Total reward: 36.0 Training loss: 1.1104 Explore P: 0.9667
    Episode: 17 Total reward: 29.0 Training loss: 2.2639 Explore P: 0.9639
    Episode: 18 Total reward: 18.0 Training loss: 1.4429 Explore P: 0.9622
    Episode: 19 Total reward: 33.0 Training loss: 1.5609 Explore P: 0.9591
    Episode: 20 Total reward: 20.0 Training loss: 1.8611 Explore P: 0.9572
    Episode: 21 Total reward: 41.0 Training loss: 3.0262 Explore P: 0.9533
    Episode: 22 Total reward: 34.0 Training loss: 2.1985 Explore P: 0.9501
    Episode: 23 Total reward: 30.0 Training loss: 1.8990 Explore P: 0.9473
    Episode: 24 Total reward: 12.0 Training loss: 1.9643 Explore P: 0.9462
    Episode: 25 Total reward: 16.0 Training loss: 4.4530 Explore P: 0.9447
    Episode: 26 Total reward: 14.0 Training loss: 3.6058 Explore P: 0.9434
    Episode: 27 Total reward: 28.0 Training loss: 7.8610 Explore P: 0.9408
    Episode: 28 Total reward: 15.0 Training loss: 6.0131 Explore P: 0.9394
    Episode: 29 Total reward: 8.0 Training loss: 5.5756 Explore P: 0.9386
    Episode: 30 Total reward: 26.0 Training loss: 8.8285 Explore P: 0.9362
    Episode: 31 Total reward: 8.0 Training loss: 9.5709 Explore P: 0.9355
    Episode: 32 Total reward: 19.0 Training loss: 9.5291 Explore P: 0.9337
    Episode: 33 Total reward: 21.0 Training loss: 14.2260 Explore P: 0.9318
    Episode: 34 Total reward: 21.0 Training loss: 8.8670 Explore P: 0.9298
    Episode: 35 Total reward: 41.0 Training loss: 8.6609 Explore P: 0.9261
    Episode: 36 Total reward: 24.0 Training loss: 8.0929 Explore P: 0.9239
    Episode: 37 Total reward: 19.0 Training loss: 2.4267 Explore P: 0.9222
    Episode: 38 Total reward: 11.0 Training loss: 7.5095 Explore P: 0.9211
    Episode: 39 Total reward: 18.0 Training loss: 15.9525 Explore P: 0.9195
    Episode: 40 Total reward: 22.0 Training loss: 4.0328 Explore P: 0.9175
    Episode: 41 Total reward: 18.0 Training loss: 15.8352 Explore P: 0.9159
    Episode: 42 Total reward: 23.0 Training loss: 7.9242 Explore P: 0.9138
    Episode: 43 Total reward: 23.0 Training loss: 18.2658 Explore P: 0.9117
    Episode: 44 Total reward: 14.0 Training loss: 3.3089 Explore P: 0.9105
    Episode: 45 Total reward: 32.0 Training loss: 16.8924 Explore P: 0.9076
    Episode: 46 Total reward: 30.0 Training loss: 28.1182 Explore P: 0.9049
    Episode: 47 Total reward: 12.0 Training loss: 4.2626 Explore P: 0.9038
    Episode: 48 Total reward: 14.0 Training loss: 3.6822 Explore P: 0.9026
    Episode: 49 Total reward: 22.0 Training loss: 34.2644 Explore P: 0.9006
    Episode: 50 Total reward: 22.0 Training loss: 41.7989 Explore P: 0.8987
    Episode: 51 Total reward: 18.0 Training loss: 34.6232 Explore P: 0.8971
    Episode: 52 Total reward: 24.0 Training loss: 19.9647 Explore P: 0.8949
    Episode: 53 Total reward: 20.0 Training loss: 16.0754 Explore P: 0.8932
    Episode: 54 Total reward: 14.0 Training loss: 4.2704 Explore P: 0.8919
    Episode: 55 Total reward: 17.0 Training loss: 5.4576 Explore P: 0.8904
    Episode: 56 Total reward: 10.0 Training loss: 29.2175 Explore P: 0.8895
    Episode: 57 Total reward: 13.0 Training loss: 4.3809 Explore P: 0.8884
    Episode: 58 Total reward: 14.0 Training loss: 28.3774 Explore P: 0.8872
    Episode: 59 Total reward: 19.0 Training loss: 42.9497 Explore P: 0.8855
    Episode: 60 Total reward: 42.0 Training loss: 6.0260 Explore P: 0.8818
    Episode: 61 Total reward: 15.0 Training loss: 4.6972 Explore P: 0.8805
    Episode: 62 Total reward: 13.0 Training loss: 31.3180 Explore P: 0.8794
    Episode: 63 Total reward: 11.0 Training loss: 5.7885 Explore P: 0.8784
    Episode: 64 Total reward: 13.0 Training loss: 5.4414 Explore P: 0.8773
    Episode: 65 Total reward: 12.0 Training loss: 5.9849 Explore P: 0.8763
    Episode: 66 Total reward: 13.0 Training loss: 6.4230 Explore P: 0.8752
    Episode: 67 Total reward: 16.0 Training loss: 20.2633 Explore P: 0.8738
    Episode: 68 Total reward: 15.0 Training loss: 5.9629 Explore P: 0.8725
    Episode: 69 Total reward: 12.0 Training loss: 69.8975 Explore P: 0.8714
    Episode: 70 Total reward: 11.0 Training loss: 67.5708 Explore P: 0.8705
    Episode: 71 Total reward: 8.0 Training loss: 21.1127 Explore P: 0.8698
    Episode: 72 Total reward: 20.0 Training loss: 55.6538 Explore P: 0.8681
    Episode: 73 Total reward: 28.0 Training loss: 30.5856 Explore P: 0.8657
    Episode: 74 Total reward: 13.0 Training loss: 6.4053 Explore P: 0.8646
    Episode: 75 Total reward: 21.0 Training loss: 5.4403 Explore P: 0.8628
    Episode: 76 Total reward: 13.0 Training loss: 39.6189 Explore P: 0.8617
    Episode: 77 Total reward: 13.0 Training loss: 44.2036 Explore P: 0.8606
    Episode: 78 Total reward: 20.0 Training loss: 85.0165 Explore P: 0.8589
    Episode: 79 Total reward: 10.0 Training loss: 57.6595 Explore P: 0.8580
    Episode: 80 Total reward: 26.0 Training loss: 39.5855 Explore P: 0.8558
    Episode: 81 Total reward: 22.0 Training loss: 113.8912 Explore P: 0.8540
    Episode: 82 Total reward: 15.0 Training loss: 4.4298 Explore P: 0.8527
    Episode: 83 Total reward: 32.0 Training loss: 5.6116 Explore P: 0.8500
    Episode: 84 Total reward: 16.0 Training loss: 64.5395 Explore P: 0.8487
    Episode: 85 Total reward: 20.0 Training loss: 3.5666 Explore P: 0.8470
    Episode: 86 Total reward: 11.0 Training loss: 43.1074 Explore P: 0.8461
    Episode: 87 Total reward: 23.0 Training loss: 130.5460 Explore P: 0.8441
    Episode: 88 Total reward: 16.0 Training loss: 33.4682 Explore P: 0.8428
    Episode: 89 Total reward: 11.0 Training loss: 4.4879 Explore P: 0.8419
    Episode: 90 Total reward: 22.0 Training loss: 85.4544 Explore P: 0.8401
    Episode: 91 Total reward: 12.0 Training loss: 26.2933 Explore P: 0.8391
    Episode: 92 Total reward: 26.0 Training loss: 5.0719 Explore P: 0.8369
    Episode: 93 Total reward: 12.0 Training loss: 4.8914 Explore P: 0.8359
    Episode: 94 Total reward: 10.0 Training loss: 4.6052 Explore P: 0.8351
    Episode: 95 Total reward: 20.0 Training loss: 33.9332 Explore P: 0.8335
    Episode: 96 Total reward: 9.0 Training loss: 34.5079 Explore P: 0.8327
    Episode: 97 Total reward: 9.0 Training loss: 24.1026 Explore P: 0.8320
    Episode: 98 Total reward: 28.0 Training loss: 26.3950 Explore P: 0.8297
    Episode: 99 Total reward: 15.0 Training loss: 78.5830 Explore P: 0.8284
    Episode: 100 Total reward: 14.0 Training loss: 84.8797 Explore P: 0.8273
    Episode: 101 Total reward: 15.0 Training loss: 33.6920 Explore P: 0.8261
    Episode: 102 Total reward: 18.0 Training loss: 62.2002 Explore P: 0.8246
    Episode: 103 Total reward: 14.0 Training loss: 127.5215 Explore P: 0.8235
    Episode: 104 Total reward: 17.0 Training loss: 109.2452 Explore P: 0.8221
    Episode: 105 Total reward: 21.0 Training loss: 26.8692 Explore P: 0.8204
    Episode: 106 Total reward: 17.0 Training loss: 131.5253 Explore P: 0.8190
    Episode: 107 Total reward: 19.0 Training loss: 41.8519 Explore P: 0.8175
    Episode: 108 Total reward: 31.0 Training loss: 106.2567 Explore P: 0.8150
    Episode: 109 Total reward: 18.0 Training loss: 68.4353 Explore P: 0.8135
    Episode: 110 Total reward: 19.0 Training loss: 38.1549 Explore P: 0.8120
    Episode: 111 Total reward: 46.0 Training loss: 31.2670 Explore P: 0.8083
    Episode: 112 Total reward: 31.0 Training loss: 45.2015 Explore P: 0.8058
    Episode: 113 Total reward: 36.0 Training loss: 44.3483 Explore P: 0.8030
    Episode: 114 Total reward: 10.0 Training loss: 35.8294 Explore P: 0.8022
    Episode: 115 Total reward: 14.0 Training loss: 40.5618 Explore P: 0.8011
    Episode: 116 Total reward: 12.0 Training loss: 37.0887 Explore P: 0.8001
    Episode: 117 Total reward: 21.0 Training loss: 41.4468 Explore P: 0.7985
    Episode: 118 Total reward: 25.0 Training loss: 37.7234 Explore P: 0.7965
    Episode: 119 Total reward: 22.0 Training loss: 56.9449 Explore P: 0.7948
    Episode: 120 Total reward: 16.0 Training loss: 33.1359 Explore P: 0.7935
    Episode: 121 Total reward: 25.0 Training loss: 58.5125 Explore P: 0.7916
    Episode: 122 Total reward: 18.0 Training loss: 30.4934 Explore P: 0.7902
    Episode: 123 Total reward: 16.0 Training loss: 1.7473 Explore P: 0.7889
    Episode: 124 Total reward: 13.0 Training loss: 32.3452 Explore P: 0.7879
    Episode: 125 Total reward: 12.0 Training loss: 55.7532 Explore P: 0.7870
    Episode: 126 Total reward: 12.0 Training loss: 67.8670 Explore P: 0.7860
    Episode: 127 Total reward: 14.0 Training loss: 29.4000 Explore P: 0.7850
    Episode: 128 Total reward: 19.0 Training loss: 1.7736 Explore P: 0.7835
    Episode: 129 Total reward: 10.0 Training loss: 1.5513 Explore P: 0.7827
    Episode: 130 Total reward: 23.0 Training loss: 80.6940 Explore P: 0.7809
    Episode: 131 Total reward: 16.0 Training loss: 27.0503 Explore P: 0.7797
    Episode: 132 Total reward: 11.0 Training loss: 123.3956 Explore P: 0.7789
    Episode: 133 Total reward: 13.0 Training loss: 29.9610 Explore P: 0.7779
    Episode: 134 Total reward: 23.0 Training loss: 50.0328 Explore P: 0.7761
    Episode: 135 Total reward: 10.0 Training loss: 1.5021 Explore P: 0.7753
    Episode: 136 Total reward: 16.0 Training loss: 23.7013 Explore P: 0.7741
    Episode: 137 Total reward: 13.0 Training loss: 1.7582 Explore P: 0.7731
    Episode: 138 Total reward: 22.0 Training loss: 80.3443 Explore P: 0.7714
    Episode: 139 Total reward: 13.0 Training loss: 24.3109 Explore P: 0.7704
    Episode: 140 Total reward: 9.0 Training loss: 23.2779 Explore P: 0.7698
    Episode: 141 Total reward: 13.0 Training loss: 24.8544 Explore P: 0.7688
    Episode: 142 Total reward: 9.0 Training loss: 29.2628 Explore P: 0.7681
    Episode: 143 Total reward: 9.0 Training loss: 0.9820 Explore P: 0.7674
    Episode: 144 Total reward: 9.0 Training loss: 21.5511 Explore P: 0.7667
    Episode: 145 Total reward: 14.0 Training loss: 49.7597 Explore P: 0.7657
    Episode: 146 Total reward: 18.0 Training loss: 1.1823 Explore P: 0.7643
    Episode: 147 Total reward: 14.0 Training loss: 1.6976 Explore P: 0.7633
    Episode: 148 Total reward: 33.0 Training loss: 23.5492 Explore P: 0.7608
    Episode: 149 Total reward: 17.0 Training loss: 1.6865 Explore P: 0.7595
    Episode: 150 Total reward: 13.0 Training loss: 1.7389 Explore P: 0.7585
    Episode: 151 Total reward: 14.0 Training loss: 55.3316 Explore P: 0.7575
    Episode: 152 Total reward: 28.0 Training loss: 1.4038 Explore P: 0.7554
    Episode: 153 Total reward: 21.0 Training loss: 0.9270 Explore P: 0.7538
    Episode: 154 Total reward: 12.0 Training loss: 23.9442 Explore P: 0.7529
    Episode: 155 Total reward: 18.0 Training loss: 1.6394 Explore P: 0.7516
    Episode: 156 Total reward: 23.0 Training loss: 26.1559 Explore P: 0.7499
    Episode: 157 Total reward: 8.0 Training loss: 45.7674 Explore P: 0.7493
    Episode: 158 Total reward: 14.0 Training loss: 1.2427 Explore P: 0.7483
    Episode: 159 Total reward: 14.0 Training loss: 1.2040 Explore P: 0.7472
    Episode: 160 Total reward: 12.0 Training loss: 1.3726 Explore P: 0.7463
    Episode: 161 Total reward: 15.0 Training loss: 51.5983 Explore P: 0.7452
    Episode: 162 Total reward: 9.0 Training loss: 41.3277 Explore P: 0.7446
    Episode: 163 Total reward: 23.0 Training loss: 18.3459 Explore P: 0.7429
    Episode: 164 Total reward: 17.0 Training loss: 19.2118 Explore P: 0.7417
    Episode: 165 Total reward: 21.0 Training loss: 40.7625 Explore P: 0.7401
    Episode: 166 Total reward: 10.0 Training loss: 39.3648 Explore P: 0.7394
    Episode: 167 Total reward: 21.0 Training loss: 35.1799 Explore P: 0.7379
    Episode: 168 Total reward: 9.0 Training loss: 1.2009 Explore P: 0.7372
    Episode: 169 Total reward: 9.0 Training loss: 1.2047 Explore P: 0.7365
    Episode: 170 Total reward: 13.0 Training loss: 1.0428 Explore P: 0.7356
    Episode: 171 Total reward: 26.0 Training loss: 50.3442 Explore P: 0.7337
    Episode: 172 Total reward: 40.0 Training loss: 34.3836 Explore P: 0.7308
    Episode: 173 Total reward: 14.0 Training loss: 61.3237 Explore P: 0.7298
    Episode: 174 Total reward: 13.0 Training loss: 1.0110 Explore P: 0.7289
    Episode: 175 Total reward: 20.0 Training loss: 1.3509 Explore P: 0.7275
    Episode: 176 Total reward: 12.0 Training loss: 16.7319 Explore P: 0.7266
    Episode: 177 Total reward: 30.0 Training loss: 30.7135 Explore P: 0.7244
    Episode: 178 Total reward: 9.0 Training loss: 1.1425 Explore P: 0.7238
    Episode: 179 Total reward: 18.0 Training loss: 1.2462 Explore P: 0.7225
    Episode: 180 Total reward: 19.0 Training loss: 42.8355 Explore P: 0.7212
    Episode: 181 Total reward: 16.0 Training loss: 30.0460 Explore P: 0.7200
    Episode: 182 Total reward: 19.0 Training loss: 37.9650 Explore P: 0.7187
    Episode: 183 Total reward: 20.0 Training loss: 34.7443 Explore P: 0.7173
    Episode: 184 Total reward: 20.0 Training loss: 43.6439 Explore P: 0.7159
    Episode: 185 Total reward: 24.0 Training loss: 19.8665 Explore P: 0.7142
    Episode: 186 Total reward: 10.0 Training loss: 1.5224 Explore P: 0.7135
    Episode: 187 Total reward: 9.0 Training loss: 1.0640 Explore P: 0.7128
    Episode: 188 Total reward: 23.0 Training loss: 31.0138 Explore P: 0.7112
    Episode: 189 Total reward: 23.0 Training loss: 44.2581 Explore P: 0.7096
    Episode: 190 Total reward: 18.0 Training loss: 1.3653 Explore P: 0.7083
    Episode: 191 Total reward: 15.0 Training loss: 31.1222 Explore P: 0.7073
    Episode: 192 Total reward: 11.0 Training loss: 1.6334 Explore P: 0.7065
    Episode: 193 Total reward: 35.0 Training loss: 1.6860 Explore P: 0.7041
    Episode: 194 Total reward: 15.0 Training loss: 30.6354 Explore P: 0.7031
    Episode: 195 Total reward: 14.0 Training loss: 31.9213 Explore P: 0.7021
    Episode: 196 Total reward: 22.0 Training loss: 1.5689 Explore P: 0.7006
    Episode: 197 Total reward: 16.0 Training loss: 18.5825 Explore P: 0.6995
    Episode: 198 Total reward: 25.0 Training loss: 1.4916 Explore P: 0.6977
    Episode: 199 Total reward: 20.0 Training loss: 17.4179 Explore P: 0.6964
    Episode: 200 Total reward: 10.0 Training loss: 20.1747 Explore P: 0.6957
    Episode: 201 Total reward: 18.0 Training loss: 54.9824 Explore P: 0.6944
    Episode: 202 Total reward: 12.0 Training loss: 1.1253 Explore P: 0.6936
    Episode: 203 Total reward: 17.0 Training loss: 29.4571 Explore P: 0.6925
    Episode: 204 Total reward: 23.0 Training loss: 1.4722 Explore P: 0.6909
    Episode: 205 Total reward: 18.0 Training loss: 21.4582 Explore P: 0.6897
    Episode: 206 Total reward: 21.0 Training loss: 29.9465 Explore P: 0.6882
    Episode: 207 Total reward: 12.0 Training loss: 17.1324 Explore P: 0.6874
    Episode: 208 Total reward: 16.0 Training loss: 16.3658 Explore P: 0.6863
    Episode: 209 Total reward: 13.0 Training loss: 31.5155 Explore P: 0.6855
    Episode: 210 Total reward: 20.0 Training loss: 1.4743 Explore P: 0.6841
    Episode: 211 Total reward: 24.0 Training loss: 1.6565 Explore P: 0.6825
    Episode: 212 Total reward: 32.0 Training loss: 30.1917 Explore P: 0.6804
    Episode: 213 Total reward: 11.0 Training loss: 16.3663 Explore P: 0.6796
    Episode: 214 Total reward: 21.0 Training loss: 1.4708 Explore P: 0.6782
    Episode: 215 Total reward: 9.0 Training loss: 15.6673 Explore P: 0.6776
    Episode: 216 Total reward: 10.0 Training loss: 52.2784 Explore P: 0.6769
    Episode: 217 Total reward: 17.0 Training loss: 1.0741 Explore P: 0.6758
    Episode: 218 Total reward: 17.0 Training loss: 15.8913 Explore P: 0.6747
    Episode: 219 Total reward: 27.0 Training loss: 53.1486 Explore P: 0.6729
    Episode: 220 Total reward: 10.0 Training loss: 32.8478 Explore P: 0.6722
    Episode: 221 Total reward: 12.0 Training loss: 16.7510 Explore P: 0.6714
    Episode: 222 Total reward: 12.0 Training loss: 15.1265 Explore P: 0.6706
    Episode: 223 Total reward: 10.0 Training loss: 15.3289 Explore P: 0.6700
    Episode: 224 Total reward: 10.0 Training loss: 18.5602 Explore P: 0.6693
    Episode: 225 Total reward: 10.0 Training loss: 38.1703 Explore P: 0.6687
    Episode: 226 Total reward: 14.0 Training loss: 67.6953 Explore P: 0.6677
    Episode: 227 Total reward: 12.0 Training loss: 34.8815 Explore P: 0.6669
    Episode: 228 Total reward: 19.0 Training loss: 1.4376 Explore P: 0.6657
    Episode: 229 Total reward: 12.0 Training loss: 30.6767 Explore P: 0.6649
    Episode: 230 Total reward: 16.0 Training loss: 1.3434 Explore P: 0.6639
    Episode: 231 Total reward: 59.0 Training loss: 24.0116 Explore P: 0.6600
    Episode: 232 Total reward: 12.0 Training loss: 31.9260 Explore P: 0.6592
    Episode: 233 Total reward: 15.0 Training loss: 47.1436 Explore P: 0.6583
    Episode: 234 Total reward: 11.0 Training loss: 30.2353 Explore P: 0.6576
    Episode: 235 Total reward: 22.0 Training loss: 1.0378 Explore P: 0.6561
    Episode: 236 Total reward: 12.0 Training loss: 38.4891 Explore P: 0.6554
    Episode: 237 Total reward: 37.0 Training loss: 52.6639 Explore P: 0.6530
    Episode: 238 Total reward: 14.0 Training loss: 0.9358 Explore P: 0.6521
    Episode: 239 Total reward: 19.0 Training loss: 32.4735 Explore P: 0.6509
    Episode: 240 Total reward: 37.0 Training loss: 25.4442 Explore P: 0.6485
    Episode: 241 Total reward: 12.0 Training loss: 1.0027 Explore P: 0.6477
    Episode: 242 Total reward: 16.0 Training loss: 24.3688 Explore P: 0.6467
    Episode: 243 Total reward: 11.0 Training loss: 15.7689 Explore P: 0.6460
    Episode: 244 Total reward: 16.0 Training loss: 0.9581 Explore P: 0.6450
    Episode: 245 Total reward: 11.0 Training loss: 0.9271 Explore P: 0.6443
    Episode: 246 Total reward: 19.0 Training loss: 34.4743 Explore P: 0.6431
    Episode: 247 Total reward: 10.0 Training loss: 35.1712 Explore P: 0.6425
    Episode: 248 Total reward: 101.0 Training loss: 14.8139 Explore P: 0.6361
    Episode: 249 Total reward: 71.0 Training loss: 16.8541 Explore P: 0.6317
    Episode: 250 Total reward: 17.0 Training loss: 16.9168 Explore P: 0.6306
    Episode: 251 Total reward: 13.0 Training loss: 28.1742 Explore P: 0.6298
    Episode: 252 Total reward: 22.0 Training loss: 23.8759 Explore P: 0.6284
    Episode: 253 Total reward: 32.0 Training loss: 12.6128 Explore P: 0.6265
    Episode: 254 Total reward: 16.0 Training loss: 0.9587 Explore P: 0.6255
    Episode: 255 Total reward: 48.0 Training loss: 64.2331 Explore P: 0.6225
    Episode: 256 Total reward: 26.0 Training loss: 1.1459 Explore P: 0.6209
    Episode: 257 Total reward: 23.0 Training loss: 0.9197 Explore P: 0.6195
    Episode: 258 Total reward: 31.0 Training loss: 1.0103 Explore P: 0.6177
    Episode: 259 Total reward: 20.0 Training loss: 35.4479 Explore P: 0.6164
    Episode: 260 Total reward: 16.0 Training loss: 32.7294 Explore P: 0.6155
    Episode: 261 Total reward: 29.0 Training loss: 14.2934 Explore P: 0.6137
    Episode: 262 Total reward: 22.0 Training loss: 13.2722 Explore P: 0.6124
    Episode: 263 Total reward: 14.0 Training loss: 15.7102 Explore P: 0.6115
    Episode: 264 Total reward: 11.0 Training loss: 22.7121 Explore P: 0.6109
    Episode: 265 Total reward: 11.0 Training loss: 1.5780 Explore P: 0.6102
    Episode: 266 Total reward: 17.0 Training loss: 20.8779 Explore P: 0.6092
    Episode: 267 Total reward: 15.0 Training loss: 25.5312 Explore P: 0.6083
    Episode: 268 Total reward: 14.0 Training loss: 17.5387 Explore P: 0.6075
    Episode: 269 Total reward: 41.0 Training loss: 31.5788 Explore P: 0.6050
    Episode: 270 Total reward: 18.0 Training loss: 15.9828 Explore P: 0.6040
    Episode: 271 Total reward: 25.0 Training loss: 19.1913 Explore P: 0.6025
    Episode: 272 Total reward: 22.0 Training loss: 0.8593 Explore P: 0.6012
    Episode: 273 Total reward: 54.0 Training loss: 17.4961 Explore P: 0.5980
    Episode: 274 Total reward: 41.0 Training loss: 1.0452 Explore P: 0.5956
    Episode: 275 Total reward: 42.0 Training loss: 17.6754 Explore P: 0.5931
    Episode: 276 Total reward: 14.0 Training loss: 32.0260 Explore P: 0.5923
    Episode: 277 Total reward: 48.0 Training loss: 56.8543 Explore P: 0.5895
    Episode: 278 Total reward: 14.0 Training loss: 1.1211 Explore P: 0.5887
    Episode: 279 Total reward: 33.0 Training loss: 25.2841 Explore P: 0.5868
    Episode: 280 Total reward: 30.0 Training loss: 0.9856 Explore P: 0.5851
    Episode: 281 Total reward: 32.0 Training loss: 16.6797 Explore P: 0.5832
    Episode: 282 Total reward: 15.0 Training loss: 1.1921 Explore P: 0.5824
    Episode: 283 Total reward: 12.0 Training loss: 1.2496 Explore P: 0.5817
    Episode: 284 Total reward: 35.0 Training loss: 0.9382 Explore P: 0.5797
    Episode: 285 Total reward: 17.0 Training loss: 17.6744 Explore P: 0.5787
    Episode: 286 Total reward: 17.0 Training loss: 32.9992 Explore P: 0.5778
    Episode: 287 Total reward: 17.0 Training loss: 14.0377 Explore P: 0.5768
    Episode: 288 Total reward: 23.0 Training loss: 33.0902 Explore P: 0.5755
    Episode: 289 Total reward: 180.0 Training loss: 0.9720 Explore P: 0.5654
    Episode: 290 Total reward: 62.0 Training loss: 0.9516 Explore P: 0.5620
    Episode: 291 Total reward: 97.0 Training loss: 42.9879 Explore P: 0.5566
    Episode: 292 Total reward: 100.0 Training loss: 1.1638 Explore P: 0.5512
    Episode: 293 Total reward: 28.0 Training loss: 12.5818 Explore P: 0.5497
    Episode: 294 Total reward: 17.0 Training loss: 15.7571 Explore P: 0.5488
    Episode: 295 Total reward: 24.0 Training loss: 0.8055 Explore P: 0.5475
    Episode: 296 Total reward: 41.0 Training loss: 13.3484 Explore P: 0.5453
    Episode: 297 Total reward: 21.0 Training loss: 1.0004 Explore P: 0.5442
    Episode: 298 Total reward: 49.0 Training loss: 0.9416 Explore P: 0.5416
    Episode: 299 Total reward: 30.0 Training loss: 15.6084 Explore P: 0.5400
    Episode: 300 Total reward: 43.0 Training loss: 0.8362 Explore P: 0.5377
    Episode: 301 Total reward: 93.0 Training loss: 12.4377 Explore P: 0.5328
    Episode: 302 Total reward: 60.0 Training loss: 16.3132 Explore P: 0.5297
    Episode: 303 Total reward: 23.0 Training loss: 18.7120 Explore P: 0.5285
    Episode: 304 Total reward: 51.0 Training loss: 1.0137 Explore P: 0.5258
    Episode: 305 Total reward: 30.0 Training loss: 32.7317 Explore P: 0.5243
    Episode: 306 Total reward: 25.0 Training loss: 29.2241 Explore P: 0.5230
    Episode: 307 Total reward: 25.0 Training loss: 18.5531 Explore P: 0.5217
    Episode: 308 Total reward: 50.0 Training loss: 22.8144 Explore P: 0.5192
    Episode: 309 Total reward: 27.0 Training loss: 22.1336 Explore P: 0.5178
    Episode: 310 Total reward: 21.0 Training loss: 1.1234 Explore P: 0.5167
    Episode: 311 Total reward: 32.0 Training loss: 0.9540 Explore P: 0.5151
    Episode: 312 Total reward: 41.0 Training loss: 13.5156 Explore P: 0.5131
    Episode: 313 Total reward: 40.0 Training loss: 19.2338 Explore P: 0.5110
    Episode: 314 Total reward: 30.0 Training loss: 17.1681 Explore P: 0.5095
    Episode: 315 Total reward: 22.0 Training loss: 0.8482 Explore P: 0.5085
    Episode: 316 Total reward: 29.0 Training loss: 14.4616 Explore P: 0.5070
    Episode: 317 Total reward: 51.0 Training loss: 18.8509 Explore P: 0.5045
    Episode: 318 Total reward: 93.0 Training loss: 53.0324 Explore P: 0.4999
    Episode: 319 Total reward: 103.0 Training loss: 0.9257 Explore P: 0.4949
    Episode: 320 Total reward: 31.0 Training loss: 0.9653 Explore P: 0.4934
    Episode: 321 Total reward: 42.0 Training loss: 16.9444 Explore P: 0.4914
    Episode: 322 Total reward: 18.0 Training loss: 0.9625 Explore P: 0.4905
    Episode: 323 Total reward: 78.0 Training loss: 15.3554 Explore P: 0.4868
    Episode: 324 Total reward: 25.0 Training loss: 19.4103 Explore P: 0.4856
    Episode: 325 Total reward: 99.0 Training loss: 16.1983 Explore P: 0.4809
    Episode: 326 Total reward: 56.0 Training loss: 16.5058 Explore P: 0.4783
    Episode: 327 Total reward: 72.0 Training loss: 15.1456 Explore P: 0.4749
    Episode: 328 Total reward: 20.0 Training loss: 0.8699 Explore P: 0.4740
    Episode: 329 Total reward: 39.0 Training loss: 31.6176 Explore P: 0.4722
    Episode: 330 Total reward: 52.0 Training loss: 31.3398 Explore P: 0.4698
    Episode: 331 Total reward: 43.0 Training loss: 21.0343 Explore P: 0.4678
    Episode: 332 Total reward: 30.0 Training loss: 1.1007 Explore P: 0.4664
    Episode: 333 Total reward: 51.0 Training loss: 31.7322 Explore P: 0.4641
    Episode: 334 Total reward: 38.0 Training loss: 0.8444 Explore P: 0.4624
    Episode: 335 Total reward: 72.0 Training loss: 0.8804 Explore P: 0.4591
    Episode: 336 Total reward: 108.0 Training loss: 21.4181 Explore P: 0.4543
    Episode: 337 Total reward: 104.0 Training loss: 1.3943 Explore P: 0.4497
    Episode: 338 Total reward: 54.0 Training loss: 47.3494 Explore P: 0.4473
    Episode: 339 Total reward: 21.0 Training loss: 21.1173 Explore P: 0.4464
    Episode: 340 Total reward: 23.0 Training loss: 49.6179 Explore P: 0.4454
    Episode: 341 Total reward: 27.0 Training loss: 1.8399 Explore P: 0.4442
    Episode: 342 Total reward: 36.0 Training loss: 1.0149 Explore P: 0.4427
    Episode: 343 Total reward: 15.0 Training loss: 1.2107 Explore P: 0.4420
    Episode: 344 Total reward: 24.0 Training loss: 21.8877 Explore P: 0.4410
    Episode: 345 Total reward: 96.0 Training loss: 48.9794 Explore P: 0.4369
    Episode: 346 Total reward: 54.0 Training loss: 0.9647 Explore P: 0.4346
    Episode: 347 Total reward: 90.0 Training loss: 0.7481 Explore P: 0.4308
    Episode: 348 Total reward: 116.0 Training loss: 39.5585 Explore P: 0.4259
    Episode: 349 Total reward: 182.0 Training loss: 1.1110 Explore P: 0.4184
    Episode: 350 Total reward: 93.0 Training loss: 1.1796 Explore P: 0.4146
    Episode: 351 Total reward: 66.0 Training loss: 0.8048 Explore P: 0.4120
    Episode: 352 Total reward: 45.0 Training loss: 59.9439 Explore P: 0.4102
    Episode: 353 Total reward: 52.0 Training loss: 45.7201 Explore P: 0.4081
    Episode: 354 Total reward: 33.0 Training loss: 25.7962 Explore P: 0.4068
    Episode: 355 Total reward: 47.0 Training loss: 0.8971 Explore P: 0.4049
    Episode: 356 Total reward: 39.0 Training loss: 39.7785 Explore P: 0.4034
    Episode: 357 Total reward: 47.0 Training loss: 22.0561 Explore P: 0.4015
    Episode: 358 Total reward: 45.0 Training loss: 1.2805 Explore P: 0.3998
    Episode: 359 Total reward: 62.0 Training loss: 0.9501 Explore P: 0.3974
    Episode: 360 Total reward: 68.0 Training loss: 18.9639 Explore P: 0.3948
    Episode: 361 Total reward: 122.0 Training loss: 82.6750 Explore P: 0.3901
    Episode: 362 Total reward: 86.0 Training loss: 55.4715 Explore P: 0.3868
    Episode: 363 Total reward: 45.0 Training loss: 32.0578 Explore P: 0.3851
    Episode: 364 Total reward: 53.0 Training loss: 91.1264 Explore P: 0.3832
    Episode: 365 Total reward: 41.0 Training loss: 26.4660 Explore P: 0.3816
    Episode: 366 Total reward: 58.0 Training loss: 0.6597 Explore P: 0.3795
    Episode: 367 Total reward: 47.0 Training loss: 59.8764 Explore P: 0.3778
    Episode: 368 Total reward: 24.0 Training loss: 1.4844 Explore P: 0.3769
    Episode: 369 Total reward: 22.0 Training loss: 59.1434 Explore P: 0.3761
    Episode: 370 Total reward: 51.0 Training loss: 1.1671 Explore P: 0.3742
    Episode: 371 Total reward: 29.0 Training loss: 1.2244 Explore P: 0.3731
    Episode: 372 Total reward: 41.0 Training loss: 16.1702 Explore P: 0.3717
    Episode: 373 Total reward: 25.0 Training loss: 0.8762 Explore P: 0.3708
    Episode: 374 Total reward: 45.0 Training loss: 25.1460 Explore P: 0.3691
    Episode: 375 Total reward: 53.0 Training loss: 74.4855 Explore P: 0.3672
    Episode: 376 Total reward: 28.0 Training loss: 0.9295 Explore P: 0.3662
    Episode: 377 Total reward: 41.0 Training loss: 35.3659 Explore P: 0.3648
    Episode: 378 Total reward: 61.0 Training loss: 1.1034 Explore P: 0.3626
    Episode: 379 Total reward: 42.0 Training loss: 27.6082 Explore P: 0.3611
    Episode: 380 Total reward: 79.0 Training loss: 36.3328 Explore P: 0.3584
    Episode: 381 Total reward: 33.0 Training loss: 1.4994 Explore P: 0.3572
    Episode: 382 Total reward: 71.0 Training loss: 1.1154 Explore P: 0.3548
    Episode: 383 Total reward: 31.0 Training loss: 1.3183 Explore P: 0.3537
    Episode: 384 Total reward: 58.0 Training loss: 1.0734 Explore P: 0.3517
    Episode: 385 Total reward: 23.0 Training loss: 32.3470 Explore P: 0.3509
    Episode: 386 Total reward: 61.0 Training loss: 1.5135 Explore P: 0.3489
    Episode: 387 Total reward: 49.0 Training loss: 1.1919 Explore P: 0.3472
    Episode: 388 Total reward: 94.0 Training loss: 0.7668 Explore P: 0.3441
    Episode: 389 Total reward: 28.0 Training loss: 1.1780 Explore P: 0.3431
    Episode: 390 Total reward: 27.0 Training loss: 32.4567 Explore P: 0.3422
    Episode: 391 Total reward: 50.0 Training loss: 0.9582 Explore P: 0.3406
    Episode: 392 Total reward: 46.0 Training loss: 1.6737 Explore P: 0.3390
    Episode: 393 Total reward: 72.0 Training loss: 28.3850 Explore P: 0.3367
    Episode: 394 Total reward: 55.0 Training loss: 1.1934 Explore P: 0.3349
    Episode: 395 Total reward: 24.0 Training loss: 22.5921 Explore P: 0.3341
    Episode: 396 Total reward: 35.0 Training loss: 36.3772 Explore P: 0.3330
    Episode: 397 Total reward: 26.0 Training loss: 0.7498 Explore P: 0.3321
    Episode: 398 Total reward: 31.0 Training loss: 1.9542 Explore P: 0.3311
    Episode: 399 Total reward: 12.0 Training loss: 1.0720 Explore P: 0.3308
    Episode: 400 Total reward: 34.0 Training loss: 39.7196 Explore P: 0.3297
    Episode: 401 Total reward: 34.0 Training loss: 30.9143 Explore P: 0.3286
    Episode: 402 Total reward: 35.0 Training loss: 1.7095 Explore P: 0.3275
    Episode: 403 Total reward: 33.0 Training loss: 1.5531 Explore P: 0.3264
    Episode: 404 Total reward: 37.0 Training loss: 1.3322 Explore P: 0.3253
    Episode: 405 Total reward: 56.0 Training loss: 62.6381 Explore P: 0.3235
    Episode: 406 Total reward: 94.0 Training loss: 32.7304 Explore P: 0.3206
    Episode: 407 Total reward: 82.0 Training loss: 1.4627 Explore P: 0.3180
    Episode: 408 Total reward: 117.0 Training loss: 1.1862 Explore P: 0.3144
    Episode: 409 Total reward: 33.0 Training loss: 65.8830 Explore P: 0.3134
    Episode: 410 Total reward: 64.0 Training loss: 1.3288 Explore P: 0.3115
    Episode: 411 Total reward: 38.0 Training loss: 25.4278 Explore P: 0.3104
    Episode: 412 Total reward: 44.0 Training loss: 1.1098 Explore P: 0.3090
    Episode: 413 Total reward: 106.0 Training loss: 37.5103 Explore P: 0.3059
    Episode: 414 Total reward: 30.0 Training loss: 41.6378 Explore P: 0.3050
    Episode: 415 Total reward: 35.0 Training loss: 30.4886 Explore P: 0.3040
    Episode: 416 Total reward: 55.0 Training loss: 41.5814 Explore P: 0.3024
    Episode: 417 Total reward: 54.0 Training loss: 39.0876 Explore P: 0.3008
    Episode: 418 Total reward: 89.0 Training loss: 0.6842 Explore P: 0.2982
    Episode: 419 Total reward: 98.0 Training loss: 44.0083 Explore P: 0.2954
    Episode: 420 Total reward: 40.0 Training loss: 1.1200 Explore P: 0.2943
    Episode: 421 Total reward: 41.0 Training loss: 45.3539 Explore P: 0.2931
    Episode: 422 Total reward: 42.0 Training loss: 44.9662 Explore P: 0.2919
    Episode: 423 Total reward: 39.0 Training loss: 42.7976 Explore P: 0.2908
    Episode: 424 Total reward: 66.0 Training loss: 74.2424 Explore P: 0.2890
    Episode: 425 Total reward: 58.0 Training loss: 77.8789 Explore P: 0.2874
    Episode: 426 Total reward: 83.0 Training loss: 35.7921 Explore P: 0.2851
    Episode: 427 Total reward: 36.0 Training loss: 51.6724 Explore P: 0.2841
    Episode: 428 Total reward: 25.0 Training loss: 1.2372 Explore P: 0.2834
    Episode: 429 Total reward: 26.0 Training loss: 1.5748 Explore P: 0.2827
    Episode: 430 Total reward: 49.0 Training loss: 57.4424 Explore P: 0.2813
    Episode: 431 Total reward: 30.0 Training loss: 27.4765 Explore P: 0.2805
    Episode: 432 Total reward: 28.0 Training loss: 17.9525 Explore P: 0.2798
    Episode: 433 Total reward: 62.0 Training loss: 1.8489 Explore P: 0.2781
    Episode: 434 Total reward: 36.0 Training loss: 44.2162 Explore P: 0.2771
    Episode: 435 Total reward: 32.0 Training loss: 0.9699 Explore P: 0.2763
    Episode: 436 Total reward: 48.0 Training loss: 1.6482 Explore P: 0.2750
    Episode: 437 Total reward: 44.0 Training loss: 78.6956 Explore P: 0.2739
    Episode: 438 Total reward: 32.0 Training loss: 1.3597 Explore P: 0.2730
    Episode: 439 Total reward: 24.0 Training loss: 49.2449 Explore P: 0.2724
    Episode: 440 Total reward: 23.0 Training loss: 1.0795 Explore P: 0.2718
    Episode: 441 Total reward: 28.0 Training loss: 1.3187 Explore P: 0.2710
    Episode: 442 Total reward: 23.0 Training loss: 74.8113 Explore P: 0.2704
    Episode: 443 Total reward: 34.0 Training loss: 67.8514 Explore P: 0.2696
    Episode: 444 Total reward: 45.0 Training loss: 89.3985 Explore P: 0.2684
    Episode: 445 Total reward: 28.0 Training loss: 1.4319 Explore P: 0.2677
    Episode: 446 Total reward: 43.0 Training loss: 1.3100 Explore P: 0.2666
    Episode: 447 Total reward: 41.0 Training loss: 138.6207 Explore P: 0.2655
    Episode: 448 Total reward: 96.0 Training loss: 102.5972 Explore P: 0.2631
    Episode: 449 Total reward: 49.0 Training loss: 2.0493 Explore P: 0.2618
    Episode: 450 Total reward: 37.0 Training loss: 46.0248 Explore P: 0.2609
    Episode: 451 Total reward: 40.0 Training loss: 1.0375 Explore P: 0.2599
    Episode: 452 Total reward: 31.0 Training loss: 1.0105 Explore P: 0.2591
    Episode: 453 Total reward: 77.0 Training loss: 66.2692 Explore P: 0.2572
    Episode: 454 Total reward: 52.0 Training loss: 1.2599 Explore P: 0.2559
    Episode: 455 Total reward: 31.0 Training loss: 46.5882 Explore P: 0.2552
    Episode: 456 Total reward: 29.0 Training loss: 1.6239 Explore P: 0.2545
    Episode: 457 Total reward: 33.0 Training loss: 0.5101 Explore P: 0.2537
    Episode: 458 Total reward: 56.0 Training loss: 44.5752 Explore P: 0.2523
    Episode: 459 Total reward: 23.0 Training loss: 0.7336 Explore P: 0.2518
    Episode: 460 Total reward: 46.0 Training loss: 44.4456 Explore P: 0.2506
    Episode: 461 Total reward: 96.0 Training loss: 0.8592 Explore P: 0.2483
    Episode: 462 Total reward: 64.0 Training loss: 45.8363 Explore P: 0.2468
    Episode: 463 Total reward: 71.0 Training loss: 1.5298 Explore P: 0.2451
    Episode: 464 Total reward: 100.0 Training loss: 0.7878 Explore P: 0.2428
    Episode: 465 Total reward: 76.0 Training loss: 1.0335 Explore P: 0.2410
    Episode: 466 Total reward: 61.0 Training loss: 0.7160 Explore P: 0.2396
    Episode: 467 Total reward: 43.0 Training loss: 0.5561 Explore P: 0.2387
    Episode: 468 Total reward: 40.0 Training loss: 1.1897 Explore P: 0.2377
    Episode: 469 Total reward: 27.0 Training loss: 38.4930 Explore P: 0.2371
    Episode: 470 Total reward: 45.0 Training loss: 1.0287 Explore P: 0.2361
    Episode: 471 Total reward: 43.0 Training loss: 0.5671 Explore P: 0.2351
    Episode: 472 Total reward: 46.0 Training loss: 0.8213 Explore P: 0.2341
    Episode: 473 Total reward: 96.0 Training loss: 0.8516 Explore P: 0.2320
    Episode: 474 Total reward: 41.0 Training loss: 1.3549 Explore P: 0.2311
    Episode: 475 Total reward: 65.0 Training loss: 0.8243 Explore P: 0.2296
    Episode: 476 Total reward: 130.0 Training loss: 0.6750 Explore P: 0.2268
    Episode: 477 Total reward: 73.0 Training loss: 0.9420 Explore P: 0.2252
    Episode: 478 Total reward: 61.0 Training loss: 87.7821 Explore P: 0.2239
    Episode: 479 Total reward: 84.0 Training loss: 50.1122 Explore P: 0.2221
    Episode: 480 Total reward: 43.0 Training loss: 1.1363 Explore P: 0.2212
    Episode: 481 Total reward: 46.0 Training loss: 84.6163 Explore P: 0.2202
    Episode: 482 Total reward: 47.0 Training loss: 0.8150 Explore P: 0.2192
    Episode: 483 Total reward: 36.0 Training loss: 60.4427 Explore P: 0.2185
    Episode: 484 Total reward: 43.0 Training loss: 93.7032 Explore P: 0.2176
    Episode: 485 Total reward: 65.0 Training loss: 1.1671 Explore P: 0.2163
    Episode: 486 Total reward: 95.0 Training loss: 48.9025 Explore P: 0.2143
    Episode: 487 Total reward: 39.0 Training loss: 0.9411 Explore P: 0.2135
    Episode: 488 Total reward: 36.0 Training loss: 99.3581 Explore P: 0.2128
    Episode: 489 Total reward: 35.0 Training loss: 0.8992 Explore P: 0.2121
    Episode: 490 Total reward: 41.0 Training loss: 0.6475 Explore P: 0.2112
    Episode: 491 Total reward: 34.0 Training loss: 245.6840 Explore P: 0.2106
    Episode: 492 Total reward: 42.0 Training loss: 47.5800 Explore P: 0.2097
    Episode: 493 Total reward: 26.0 Training loss: 33.8612 Explore P: 0.2092
    Episode: 494 Total reward: 52.0 Training loss: 48.4307 Explore P: 0.2082
    Episode: 495 Total reward: 34.0 Training loss: 89.7353 Explore P: 0.2075
    Episode: 496 Total reward: 26.0 Training loss: 1.1094 Explore P: 0.2070
    Episode: 497 Total reward: 32.0 Training loss: 1.6698 Explore P: 0.2064
    Episode: 498 Total reward: 25.0 Training loss: 49.7100 Explore P: 0.2059
    Episode: 499 Total reward: 16.0 Training loss: 0.8341 Explore P: 0.2055
    Episode: 500 Total reward: 45.0 Training loss: 46.6392 Explore P: 0.2047
    Episode: 501 Total reward: 30.0 Training loss: 0.6467 Explore P: 0.2041
    Episode: 502 Total reward: 22.0 Training loss: 0.7904 Explore P: 0.2037
    Episode: 503 Total reward: 26.0 Training loss: 46.6203 Explore P: 0.2032
    Episode: 504 Total reward: 22.0 Training loss: 0.8583 Explore P: 0.2027
    Episode: 505 Total reward: 34.0 Training loss: 1.0157 Explore P: 0.2021
    Episode: 506 Total reward: 57.0 Training loss: 50.8511 Explore P: 0.2010
    Episode: 507 Total reward: 24.0 Training loss: 51.9230 Explore P: 0.2005
    Episode: 508 Total reward: 48.0 Training loss: 1.5560 Explore P: 0.1996
    Episode: 509 Total reward: 23.0 Training loss: 1.1004 Explore P: 0.1992
    Episode: 510 Total reward: 41.0 Training loss: 1.7594 Explore P: 0.1984
    Episode: 511 Total reward: 17.0 Training loss: 0.9339 Explore P: 0.1981
    Episode: 512 Total reward: 40.0 Training loss: 0.7684 Explore P: 0.1973
    Episode: 513 Total reward: 83.0 Training loss: 1.3498 Explore P: 0.1958
    Episode: 514 Total reward: 46.0 Training loss: 68.0810 Explore P: 0.1949
    Episode: 515 Total reward: 32.0 Training loss: 47.3843 Explore P: 0.1943
    Episode: 516 Total reward: 31.0 Training loss: 1.7414 Explore P: 0.1938
    Episode: 517 Total reward: 69.0 Training loss: 1.3368 Explore P: 0.1925
    Episode: 518 Total reward: 34.0 Training loss: 1.4142 Explore P: 0.1919
    Episode: 519 Total reward: 71.0 Training loss: 107.8491 Explore P: 0.1906
    Episode: 520 Total reward: 36.0 Training loss: 0.6696 Explore P: 0.1900
    Episode: 521 Total reward: 45.0 Training loss: 37.5521 Explore P: 0.1891
    Episode: 522 Total reward: 34.0 Training loss: 42.3880 Explore P: 0.1885
    Episode: 523 Total reward: 104.0 Training loss: 1.3751 Explore P: 0.1867
    Episode: 524 Total reward: 22.0 Training loss: 1.3119 Explore P: 0.1863
    Episode: 525 Total reward: 50.0 Training loss: 0.9083 Explore P: 0.1854
    Episode: 526 Total reward: 106.0 Training loss: 0.7327 Explore P: 0.1836
    Episode: 527 Total reward: 43.0 Training loss: 0.9971 Explore P: 0.1828
    Episode: 528 Total reward: 40.0 Training loss: 37.0942 Explore P: 0.1821
    Episode: 529 Total reward: 106.0 Training loss: 1.0815 Explore P: 0.1803
    Episode: 530 Total reward: 46.0 Training loss: 0.9340 Explore P: 0.1795
    Episode: 531 Total reward: 28.0 Training loss: 32.7012 Explore P: 0.1791
    Episode: 532 Total reward: 105.0 Training loss: 76.9986 Explore P: 0.1773
    Episode: 533 Total reward: 101.0 Training loss: 1.7702 Explore P: 0.1756
    Episode: 534 Total reward: 42.0 Training loss: 1.3609 Explore P: 0.1749
    Episode: 535 Total reward: 45.0 Training loss: 1.2900 Explore P: 0.1742
    Episode: 536 Total reward: 94.0 Training loss: 1.1945 Explore P: 0.1727
    Episode: 537 Total reward: 76.0 Training loss: 1.0375 Explore P: 0.1714
    Episode: 538 Total reward: 47.0 Training loss: 1.6555 Explore P: 0.1707
    Episode: 539 Total reward: 82.0 Training loss: 1.3342 Explore P: 0.1694
    Episode: 540 Total reward: 53.0 Training loss: 87.0573 Explore P: 0.1685
    Episode: 541 Total reward: 170.0 Training loss: 37.6587 Explore P: 0.1658
    Episode: 542 Total reward: 81.0 Training loss: 0.8945 Explore P: 0.1646
    Episode: 543 Total reward: 102.0 Training loss: 1.3204 Explore P: 0.1630
    Episode: 544 Total reward: 71.0 Training loss: 1.0987 Explore P: 0.1619
    Episode: 545 Total reward: 72.0 Training loss: 150.3442 Explore P: 0.1608
    Episode: 546 Total reward: 157.0 Training loss: 1.5256 Explore P: 0.1585
    Episode: 547 Total reward: 111.0 Training loss: 1.5824 Explore P: 0.1568
    Episode: 548 Total reward: 94.0 Training loss: 1.1042 Explore P: 0.1555
    Episode: 549 Total reward: 88.0 Training loss: 33.2695 Explore P: 0.1542
    Episode: 550 Total reward: 153.0 Training loss: 1.1117 Explore P: 0.1520
    Episode: 551 Total reward: 146.0 Training loss: 32.0605 Explore P: 0.1500
    Episode: 552 Total reward: 52.0 Training loss: 60.5128 Explore P: 0.1492
    Episode: 553 Total reward: 74.0 Training loss: 1.0767 Explore P: 0.1482
    Episode: 554 Total reward: 89.0 Training loss: 1.0994 Explore P: 0.1470
    Episode: 555 Total reward: 49.0 Training loss: 1.3819 Explore P: 0.1463
    Episode: 556 Total reward: 172.0 Training loss: 31.7570 Explore P: 0.1440
    Episode: 557 Total reward: 71.0 Training loss: 0.7433 Explore P: 0.1430
    Episode: 558 Total reward: 71.0 Training loss: 97.4984 Explore P: 0.1421
    Episode: 560 Total reward: 23.0 Training loss: 0.7479 Explore P: 0.1392
    Episode: 561 Total reward: 98.0 Training loss: 1.1190 Explore P: 0.1379
    Episode: 563 Total reward: 56.0 Training loss: 1.0913 Explore P: 0.1347
    Episode: 564 Total reward: 95.0 Training loss: 0.8767 Explore P: 0.1335
    Episode: 566 Total reward: 21.0 Training loss: 1.4097 Explore P: 0.1308
    Episode: 568 Total reward: 61.0 Training loss: 1.0506 Explore P: 0.1277
    Episode: 570 Total reward: 108.0 Training loss: 1.2683 Explore P: 0.1241
    Episode: 572 Total reward: 111.0 Training loss: 1.1438 Explore P: 0.1206
    Episode: 573 Total reward: 90.0 Training loss: 0.9197 Explore P: 0.1196
    Episode: 576 Total reward: 99.0 Training loss: 1.1236 Explore P: 0.1143
    Episode: 579 Total reward: 13.0 Training loss: 1.0363 Explore P: 0.1101
    Episode: 580 Total reward: 65.0 Training loss: 0.8385 Explore P: 0.1094
    Episode: 581 Total reward: 119.0 Training loss: 76.4201 Explore P: 0.1083
    Episode: 583 Total reward: 135.0 Training loss: 0.8603 Explore P: 0.1050
    Episode: 585 Total reward: 132.0 Training loss: 1.4547 Explore P: 0.1019
    Episode: 587 Total reward: 138.0 Training loss: 1.2112 Explore P: 0.0989
    Episode: 588 Total reward: 63.0 Training loss: 0.5996 Explore P: 0.0983
    Episode: 589 Total reward: 96.0 Training loss: 1.3039 Explore P: 0.0975
    Episode: 590 Total reward: 53.0 Training loss: 61.8064 Explore P: 0.0970
    Episode: 591 Total reward: 54.0 Training loss: 1.0142 Explore P: 0.0965
    Episode: 592 Total reward: 86.0 Training loss: 0.8421 Explore P: 0.0958
    Episode: 593 Total reward: 55.0 Training loss: 0.8152 Explore P: 0.0953
    Episode: 594 Total reward: 87.0 Training loss: 51.6248 Explore P: 0.0946
    Episode: 595 Total reward: 42.0 Training loss: 1.2251 Explore P: 0.0942
    Episode: 596 Total reward: 25.0 Training loss: 1.4259 Explore P: 0.0940
    Episode: 597 Total reward: 42.0 Training loss: 0.7207 Explore P: 0.0937
    Episode: 598 Total reward: 73.0 Training loss: 1.0209 Explore P: 0.0931
    Episode: 599 Total reward: 74.0 Training loss: 0.7975 Explore P: 0.0924
    Episode: 600 Total reward: 59.0 Training loss: 0.8920 Explore P: 0.0920
    Episode: 601 Total reward: 31.0 Training loss: 0.5880 Explore P: 0.0917
    Episode: 602 Total reward: 53.0 Training loss: 1.1259 Explore P: 0.0913
    Episode: 603 Total reward: 52.0 Training loss: 104.8127 Explore P: 0.0909
    Episode: 604 Total reward: 25.0 Training loss: 1.4683 Explore P: 0.0906
    Episode: 605 Total reward: 32.0 Training loss: 0.7544 Explore P: 0.0904
    Episode: 606 Total reward: 45.0 Training loss: 1.2714 Explore P: 0.0900
    Episode: 607 Total reward: 31.0 Training loss: 0.7547 Explore P: 0.0898
    Episode: 608 Total reward: 28.0 Training loss: 83.1030 Explore P: 0.0896
    Episode: 609 Total reward: 45.0 Training loss: 1.0917 Explore P: 0.0892
    Episode: 610 Total reward: 19.0 Training loss: 1.3218 Explore P: 0.0891
    Episode: 611 Total reward: 22.0 Training loss: 1.4669 Explore P: 0.0889
    Episode: 612 Total reward: 15.0 Training loss: 1.2770 Explore P: 0.0888
    Episode: 613 Total reward: 25.0 Training loss: 1.0878 Explore P: 0.0886
    Episode: 614 Total reward: 35.0 Training loss: 0.9151 Explore P: 0.0883
    Episode: 615 Total reward: 39.0 Training loss: 0.8682 Explore P: 0.0880
    Episode: 616 Total reward: 51.0 Training loss: 1.2394 Explore P: 0.0876
    Episode: 617 Total reward: 25.0 Training loss: 1.0525 Explore P: 0.0874
    Episode: 618 Total reward: 33.0 Training loss: 1.1861 Explore P: 0.0871
    Episode: 619 Total reward: 27.0 Training loss: 68.7462 Explore P: 0.0869
    Episode: 620 Total reward: 25.0 Training loss: 1.6499 Explore P: 0.0867
    Episode: 621 Total reward: 31.0 Training loss: 1.4728 Explore P: 0.0865
    Episode: 622 Total reward: 20.0 Training loss: 1.3360 Explore P: 0.0863
    Episode: 623 Total reward: 38.0 Training loss: 1.4536 Explore P: 0.0861
    Episode: 624 Total reward: 51.0 Training loss: 0.9914 Explore P: 0.0857
    Episode: 625 Total reward: 28.0 Training loss: 1.1818 Explore P: 0.0855
    Episode: 626 Total reward: 34.0 Training loss: 0.7762 Explore P: 0.0852
    Episode: 627 Total reward: 36.0 Training loss: 1.4418 Explore P: 0.0849
    Episode: 628 Total reward: 29.0 Training loss: 1.1278 Explore P: 0.0847
    Episode: 629 Total reward: 41.0 Training loss: 0.6995 Explore P: 0.0844
    Episode: 630 Total reward: 37.0 Training loss: 1.1775 Explore P: 0.0841
    Episode: 631 Total reward: 31.0 Training loss: 0.8110 Explore P: 0.0839
    Episode: 632 Total reward: 26.0 Training loss: 0.9239 Explore P: 0.0837
    Episode: 633 Total reward: 43.0 Training loss: 0.6583 Explore P: 0.0834
    Episode: 634 Total reward: 34.0 Training loss: 1.2079 Explore P: 0.0831
    Episode: 635 Total reward: 35.0 Training loss: 0.7573 Explore P: 0.0829
    Episode: 636 Total reward: 60.0 Training loss: 104.6702 Explore P: 0.0825
    Episode: 637 Total reward: 57.0 Training loss: 92.2541 Explore P: 0.0820
    Episode: 638 Total reward: 57.0 Training loss: 1.0459 Explore P: 0.0816
    Episode: 639 Total reward: 48.0 Training loss: 62.7886 Explore P: 0.0813
    Episode: 640 Total reward: 68.0 Training loss: 0.6358 Explore P: 0.0808
    Episode: 641 Total reward: 34.0 Training loss: 99.8760 Explore P: 0.0806
    Episode: 642 Total reward: 40.0 Training loss: 67.0972 Explore P: 0.0803
    Episode: 643 Total reward: 69.0 Training loss: 1.5811 Explore P: 0.0798
    Episode: 644 Total reward: 58.0 Training loss: 91.5515 Explore P: 0.0794
    Episode: 645 Total reward: 46.0 Training loss: 1.0127 Explore P: 0.0791
    Episode: 646 Total reward: 43.0 Training loss: 0.9824 Explore P: 0.0788
    Episode: 647 Total reward: 73.0 Training loss: 1.0012 Explore P: 0.0783
    Episode: 648 Total reward: 85.0 Training loss: 73.7900 Explore P: 0.0777
    Episode: 649 Total reward: 51.0 Training loss: 68.1266 Explore P: 0.0774
    Episode: 650 Total reward: 39.0 Training loss: 1.0596 Explore P: 0.0771
    Episode: 651 Total reward: 62.0 Training loss: 1.0994 Explore P: 0.0767
    Episode: 652 Total reward: 77.0 Training loss: 0.9932 Explore P: 0.0762
    Episode: 653 Total reward: 74.0 Training loss: 0.8897 Explore P: 0.0757
    Episode: 655 Total reward: 22.0 Training loss: 0.8273 Explore P: 0.0742
    Episode: 656 Total reward: 86.0 Training loss: 1.5060 Explore P: 0.0737
    Episode: 659 Total reward: 99.0 Training loss: 0.8442 Explore P: 0.0706
    Episode: 662 Total reward: 7.0 Training loss: 0.3972 Explore P: 0.0682
    Episode: 665 Total reward: 37.0 Training loss: 0.4313 Explore P: 0.0657
    Episode: 667 Total reward: 155.0 Training loss: 0.6528 Explore P: 0.0637
    Episode: 669 Total reward: 184.0 Training loss: 0.8234 Explore P: 0.0617
    Episode: 671 Total reward: 90.0 Training loss: 0.5068 Explore P: 0.0602
    Episode: 673 Total reward: 115.0 Training loss: 15.8515 Explore P: 0.0587
    Episode: 675 Total reward: 58.0 Training loss: 0.5398 Explore P: 0.0574
    Episode: 677 Total reward: 1.0 Training loss: 60.1238 Explore P: 0.0565
    Episode: 679 Total reward: 26.0 Training loss: 0.3943 Explore P: 0.0555
    Episode: 681 Total reward: 19.0 Training loss: 30.9515 Explore P: 0.0545
    Episode: 683 Total reward: 41.0 Training loss: 0.5046 Explore P: 0.0534
    Episode: 685 Total reward: 17.0 Training loss: 0.7004 Explore P: 0.0525
    Episode: 686 Total reward: 190.0 Training loss: 0.6983 Explore P: 0.0517
    Episode: 687 Total reward: 20.0 Training loss: 0.5894 Explore P: 0.0516
    Episode: 688 Total reward: 18.0 Training loss: 0.6599 Explore P: 0.0515
    Episode: 689 Total reward: 13.0 Training loss: 19.6875 Explore P: 0.0515
    Episode: 690 Total reward: 17.0 Training loss: 1.0635 Explore P: 0.0514
    Episode: 691 Total reward: 163.0 Training loss: 0.6311 Explore P: 0.0507
    Episode: 692 Total reward: 169.0 Training loss: 0.6215 Explore P: 0.0501
    Episode: 693 Total reward: 11.0 Training loss: 1.2989 Explore P: 0.0500
    Episode: 694 Total reward: 16.0 Training loss: 0.9625 Explore P: 0.0499
    Episode: 695 Total reward: 14.0 Training loss: 1.0109 Explore P: 0.0499
    Episode: 696 Total reward: 11.0 Training loss: 0.8986 Explore P: 0.0498
    Episode: 697 Total reward: 14.0 Training loss: 1.3260 Explore P: 0.0498
    Episode: 698 Total reward: 10.0 Training loss: 0.6947 Explore P: 0.0497
    Episode: 699 Total reward: 12.0 Training loss: 0.8091 Explore P: 0.0497
    Episode: 700 Total reward: 10.0 Training loss: 0.4885 Explore P: 0.0497
    Episode: 701 Total reward: 14.0 Training loss: 0.9189 Explore P: 0.0496
    Episode: 702 Total reward: 14.0 Training loss: 1.0030 Explore P: 0.0496
    Episode: 703 Total reward: 15.0 Training loss: 1.0129 Explore P: 0.0495
    Episode: 704 Total reward: 14.0 Training loss: 0.7741 Explore P: 0.0494
    Episode: 705 Total reward: 14.0 Training loss: 18.7141 Explore P: 0.0494
    Episode: 706 Total reward: 12.0 Training loss: 0.7680 Explore P: 0.0493
    Episode: 707 Total reward: 12.0 Training loss: 0.9533 Explore P: 0.0493
    Episode: 708 Total reward: 15.0 Training loss: 1.3175 Explore P: 0.0492
    Episode: 709 Total reward: 12.0 Training loss: 0.8715 Explore P: 0.0492
    Episode: 710 Total reward: 10.0 Training loss: 0.7716 Explore P: 0.0491
    Episode: 711 Total reward: 14.0 Training loss: 1.5015 Explore P: 0.0491
    Episode: 712 Total reward: 10.0 Training loss: 0.8854 Explore P: 0.0490
    Episode: 713 Total reward: 12.0 Training loss: 0.9519 Explore P: 0.0490
    Episode: 714 Total reward: 10.0 Training loss: 2.0882 Explore P: 0.0490
    Episode: 715 Total reward: 10.0 Training loss: 1.3294 Explore P: 0.0489
    Episode: 716 Total reward: 10.0 Training loss: 1.3681 Explore P: 0.0489
    Episode: 717 Total reward: 12.0 Training loss: 2.1211 Explore P: 0.0488
    Episode: 718 Total reward: 11.0 Training loss: 1.1942 Explore P: 0.0488
    Episode: 719 Total reward: 9.0 Training loss: 2.2088 Explore P: 0.0488
    Episode: 720 Total reward: 9.0 Training loss: 2.3290 Explore P: 0.0487
    Episode: 721 Total reward: 10.0 Training loss: 518.4750 Explore P: 0.0487
    Episode: 722 Total reward: 13.0 Training loss: 71.6646 Explore P: 0.0486
    Episode: 723 Total reward: 13.0 Training loss: 1.7532 Explore P: 0.0486
    Episode: 724 Total reward: 10.0 Training loss: 1.7007 Explore P: 0.0485
    Episode: 725 Total reward: 13.0 Training loss: 0.8270 Explore P: 0.0485
    Episode: 726 Total reward: 13.0 Training loss: 0.8814 Explore P: 0.0484
    Episode: 727 Total reward: 12.0 Training loss: 39.9917 Explore P: 0.0484
    Episode: 728 Total reward: 10.0 Training loss: 0.9025 Explore P: 0.0484
    Episode: 729 Total reward: 12.0 Training loss: 1.3281 Explore P: 0.0483
    Episode: 730 Total reward: 17.0 Training loss: 1.2666 Explore P: 0.0483
    Episode: 731 Total reward: 14.0 Training loss: 26.1423 Explore P: 0.0482
    Episode: 732 Total reward: 12.0 Training loss: 1.1001 Explore P: 0.0482
    Episode: 733 Total reward: 12.0 Training loss: 495.0217 Explore P: 0.0481
    Episode: 734 Total reward: 11.0 Training loss: 26.0040 Explore P: 0.0481
    Episode: 735 Total reward: 11.0 Training loss: 20.6802 Explore P: 0.0480
    Episode: 736 Total reward: 15.0 Training loss: 0.6094 Explore P: 0.0480
    Episode: 737 Total reward: 19.0 Training loss: 0.8542 Explore P: 0.0479
    Episode: 738 Total reward: 18.0 Training loss: 1.1146 Explore P: 0.0478
    Episode: 739 Total reward: 18.0 Training loss: 438.9958 Explore P: 0.0478
    Episode: 740 Total reward: 15.0 Training loss: 1.7551 Explore P: 0.0477
    Episode: 741 Total reward: 18.0 Training loss: 0.9383 Explore P: 0.0476
    Episode: 744 Total reward: 99.0 Training loss: 309.8169 Explore P: 0.0458
    Episode: 745 Total reward: 74.0 Training loss: 30.6902 Explore P: 0.0455
    Episode: 746 Total reward: 62.0 Training loss: 1.6034 Explore P: 0.0453
    Episode: 748 Total reward: 20.0 Training loss: 4.7512 Explore P: 0.0445
    Episode: 749 Total reward: 132.0 Training loss: 1.8296 Explore P: 0.0441
    Episode: 750 Total reward: 23.0 Training loss: 2.8600 Explore P: 0.0440
    Episode: 751 Total reward: 100.0 Training loss: 3.0550 Explore P: 0.0437
    Episode: 752 Total reward: 17.0 Training loss: 2.8279 Explore P: 0.0436
    Episode: 753 Total reward: 18.0 Training loss: 4.0457 Explore P: 0.0436
    Episode: 754 Total reward: 90.0 Training loss: 2.8454 Explore P: 0.0433
    Episode: 755 Total reward: 24.0 Training loss: 2.6946 Explore P: 0.0432
    Episode: 756 Total reward: 25.0 Training loss: 2.9165 Explore P: 0.0431
    Episode: 757 Total reward: 18.0 Training loss: 4.0491 Explore P: 0.0430
    Episode: 758 Total reward: 19.0 Training loss: 3.2300 Explore P: 0.0430
    Episode: 759 Total reward: 17.0 Training loss: 3.9905 Explore P: 0.0429
    Episode: 760 Total reward: 19.0 Training loss: 2.7146 Explore P: 0.0429
    Episode: 761 Total reward: 15.0 Training loss: 2.6364 Explore P: 0.0428
    Episode: 762 Total reward: 18.0 Training loss: 3.6511 Explore P: 0.0428
    Episode: 763 Total reward: 17.0 Training loss: 2.7483 Explore P: 0.0427
    Episode: 764 Total reward: 15.0 Training loss: 2.5874 Explore P: 0.0426
    Episode: 765 Total reward: 15.0 Training loss: 3.7333 Explore P: 0.0426
    Episode: 766 Total reward: 19.0 Training loss: 1.7604 Explore P: 0.0425
    Episode: 767 Total reward: 95.0 Training loss: 4.6628 Explore P: 0.0422
    Episode: 768 Total reward: 85.0 Training loss: 438.5443 Explore P: 0.0420
    Episode: 770 Total reward: 109.0 Training loss: 4.9026 Explore P: 0.0410
    Episode: 773 Total reward: 99.0 Training loss: 5.5398 Explore P: 0.0395
    Episode: 776 Total reward: 30.0 Training loss: 8.8445 Explore P: 0.0382
    Episode: 777 Total reward: 11.0 Training loss: 297.4535 Explore P: 0.0382
    Episode: 778 Total reward: 14.0 Training loss: 3.9338 Explore P: 0.0382
    Episode: 779 Total reward: 13.0 Training loss: 10.2292 Explore P: 0.0381
    Episode: 780 Total reward: 10.0 Training loss: 4.6332 Explore P: 0.0381
    Episode: 781 Total reward: 13.0 Training loss: 4.6383 Explore P: 0.0381
    Episode: 782 Total reward: 9.0 Training loss: 3.7765 Explore P: 0.0380
    Episode: 783 Total reward: 14.0 Training loss: 6.9654 Explore P: 0.0380
    Episode: 784 Total reward: 13.0 Training loss: 9.3541 Explore P: 0.0380
    Episode: 785 Total reward: 14.0 Training loss: 6.8332 Explore P: 0.0379
    Episode: 786 Total reward: 13.0 Training loss: 7.2616 Explore P: 0.0379
    Episode: 787 Total reward: 13.0 Training loss: 6.1916 Explore P: 0.0378
    Episode: 788 Total reward: 13.0 Training loss: 5.6513 Explore P: 0.0378
    Episode: 789 Total reward: 13.0 Training loss: 3.7610 Explore P: 0.0378
    Episode: 790 Total reward: 13.0 Training loss: 6.3971 Explore P: 0.0377
    Episode: 791 Total reward: 12.0 Training loss: 6.7465 Explore P: 0.0377
    Episode: 792 Total reward: 11.0 Training loss: 6.3855 Explore P: 0.0377
    Episode: 793 Total reward: 9.0 Training loss: 3.6154 Explore P: 0.0377
    Episode: 794 Total reward: 9.0 Training loss: 60.1925 Explore P: 0.0376
    Episode: 795 Total reward: 11.0 Training loss: 5.6593 Explore P: 0.0376
    Episode: 796 Total reward: 9.0 Training loss: 4.0777 Explore P: 0.0376
    Episode: 797 Total reward: 9.0 Training loss: 170.5929 Explore P: 0.0375
    Episode: 798 Total reward: 8.0 Training loss: 8.3514 Explore P: 0.0375
    Episode: 799 Total reward: 10.0 Training loss: 6.5967 Explore P: 0.0375
    Episode: 800 Total reward: 9.0 Training loss: 9.1563 Explore P: 0.0375
    Episode: 801 Total reward: 12.0 Training loss: 9.7960 Explore P: 0.0374
    Episode: 802 Total reward: 8.0 Training loss: 5.8221 Explore P: 0.0374
    Episode: 803 Total reward: 12.0 Training loss: 6.4400 Explore P: 0.0374
    Episode: 804 Total reward: 8.0 Training loss: 4.7432 Explore P: 0.0374
    Episode: 805 Total reward: 10.0 Training loss: 4.9999 Explore P: 0.0373
    Episode: 806 Total reward: 10.0 Training loss: 4.4956 Explore P: 0.0373
    Episode: 807 Total reward: 10.0 Training loss: 6.6245 Explore P: 0.0373
    Episode: 808 Total reward: 13.0 Training loss: 228.3381 Explore P: 0.0372
    Episode: 809 Total reward: 9.0 Training loss: 154.8405 Explore P: 0.0372
    Episode: 810 Total reward: 10.0 Training loss: 615.5743 Explore P: 0.0372
    Episode: 811 Total reward: 14.0 Training loss: 6.2489 Explore P: 0.0372
    Episode: 812 Total reward: 8.0 Training loss: 6.8984 Explore P: 0.0371
    Episode: 813 Total reward: 12.0 Training loss: 6.7138 Explore P: 0.0371
    Episode: 814 Total reward: 10.0 Training loss: 6.0621 Explore P: 0.0371
    Episode: 815 Total reward: 12.0 Training loss: 4.3599 Explore P: 0.0370
    Episode: 816 Total reward: 14.0 Training loss: 7.5487 Explore P: 0.0370
    Episode: 817 Total reward: 16.0 Training loss: 119.7378 Explore P: 0.0370
    Episode: 820 Total reward: 99.0 Training loss: 10.2093 Explore P: 0.0356
    Episode: 823 Total reward: 99.0 Training loss: 6.0271 Explore P: 0.0344
    Episode: 825 Total reward: 129.0 Training loss: 46.9435 Explore P: 0.0336
    Episode: 827 Total reward: 100.0 Training loss: 3.0971 Explore P: 0.0329
    Episode: 829 Total reward: 137.0 Training loss: 7.0191 Explore P: 0.0322
    Episode: 831 Total reward: 146.0 Training loss: 8.4871 Explore P: 0.0314
    Episode: 833 Total reward: 191.0 Training loss: 253.0718 Explore P: 0.0306
    Episode: 835 Total reward: 134.0 Training loss: 8.7159 Explore P: 0.0299
    Episode: 837 Total reward: 177.0 Training loss: 99.3409 Explore P: 0.0292
    Episode: 839 Total reward: 110.0 Training loss: 7.3589 Explore P: 0.0286
    Episode: 841 Total reward: 95.0 Training loss: 6.6769 Explore P: 0.0280
    Episode: 843 Total reward: 124.0 Training loss: 50.2016 Explore P: 0.0275
    Episode: 845 Total reward: 128.0 Training loss: 5.6657 Explore P: 0.0269
    Episode: 847 Total reward: 124.0 Training loss: 4.4901 Explore P: 0.0264
    Episode: 849 Total reward: 126.0 Training loss: 6.6858 Explore P: 0.0258
    Episode: 851 Total reward: 102.0 Training loss: 4.6775 Explore P: 0.0254
    Episode: 853 Total reward: 159.0 Training loss: 6.3308 Explore P: 0.0248
    Episode: 856 Total reward: 10.0 Training loss: 4.4621 Explore P: 0.0242
    Episode: 859 Total reward: 5.0 Training loss: 1.8795 Explore P: 0.0237
    Episode: 861 Total reward: 184.0 Training loss: 2.0657 Explore P: 0.0232
    Episode: 863 Total reward: 116.0 Training loss: 1.6622 Explore P: 0.0227
    Episode: 865 Total reward: 84.0 Training loss: 6.6902 Explore P: 0.0224
    Episode: 867 Total reward: 53.0 Training loss: 5.1286 Explore P: 0.0221
    Episode: 869 Total reward: 49.0 Training loss: 171.2404 Explore P: 0.0218
    Episode: 871 Total reward: 33.0 Training loss: 0.8161 Explore P: 0.0215
    Episode: 873 Total reward: 69.0 Training loss: 2.0550 Explore P: 0.0212
    Episode: 875 Total reward: 45.0 Training loss: 2.6673 Explore P: 0.0209
    Episode: 877 Total reward: 74.0 Training loss: 1.1790 Explore P: 0.0206
    Episode: 879 Total reward: 33.0 Training loss: 411.7237 Explore P: 0.0204
    Episode: 881 Total reward: 41.0 Training loss: 1.0091 Explore P: 0.0201
    Episode: 883 Total reward: 24.0 Training loss: 2.7432 Explore P: 0.0199
    Episode: 885 Total reward: 30.0 Training loss: 0.4017 Explore P: 0.0197
    Episode: 887 Total reward: 75.0 Training loss: 0.4007 Explore P: 0.0194
    Episode: 889 Total reward: 102.0 Training loss: 393.4166 Explore P: 0.0191
    Episode: 891 Total reward: 87.0 Training loss: 1.3304 Explore P: 0.0189
    Episode: 893 Total reward: 117.0 Training loss: 0.5361 Explore P: 0.0186
    Episode: 895 Total reward: 74.0 Training loss: 0.2955 Explore P: 0.0184
    Episode: 897 Total reward: 117.0 Training loss: 0.5033 Explore P: 0.0181
    Episode: 899 Total reward: 115.0 Training loss: 0.8183 Explore P: 0.0179
    Episode: 901 Total reward: 107.0 Training loss: 167.2904 Explore P: 0.0176
    Episode: 903 Total reward: 109.0 Training loss: 0.5002 Explore P: 0.0174
    Episode: 905 Total reward: 125.0 Training loss: 0.4015 Explore P: 0.0172
    Episode: 907 Total reward: 117.0 Training loss: 0.2261 Explore P: 0.0169
    Episode: 909 Total reward: 66.0 Training loss: 0.1306 Explore P: 0.0168
    Episode: 911 Total reward: 77.0 Training loss: 0.9438 Explore P: 0.0166
    Episode: 913 Total reward: 92.0 Training loss: 0.8653 Explore P: 0.0164
    Episode: 915 Total reward: 122.0 Training loss: 0.0881 Explore P: 0.0162
    Episode: 917 Total reward: 89.0 Training loss: 0.1481 Explore P: 0.0160
    Episode: 919 Total reward: 104.0 Training loss: 0.1291 Explore P: 0.0158
    Episode: 921 Total reward: 83.0 Training loss: 0.2232 Explore P: 0.0157
    Episode: 923 Total reward: 111.0 Training loss: 0.3858 Explore P: 0.0155
    Episode: 925 Total reward: 92.0 Training loss: 0.1132 Explore P: 0.0153
    Episode: 927 Total reward: 141.0 Training loss: 0.3446 Explore P: 0.0152
    Episode: 929 Total reward: 141.0 Training loss: 54.1056 Explore P: 0.0150
    Episode: 931 Total reward: 107.0 Training loss: 0.1330 Explore P: 0.0148
    Episode: 933 Total reward: 102.0 Training loss: 1.3635 Explore P: 0.0147
    Episode: 935 Total reward: 81.0 Training loss: 0.1166 Explore P: 0.0146
    Episode: 937 Total reward: 131.0 Training loss: 0.1017 Explore P: 0.0144
    Episode: 939 Total reward: 114.0 Training loss: 0.2348 Explore P: 0.0143
    Episode: 941 Total reward: 140.0 Training loss: 58.3441 Explore P: 0.0141
    Episode: 943 Total reward: 133.0 Training loss: 0.0885 Explore P: 0.0140
    Episode: 945 Total reward: 183.0 Training loss: 0.1272 Explore P: 0.0138
    Episode: 947 Total reward: 181.0 Training loss: 0.0682 Explore P: 0.0137
    Episode: 950 Total reward: 33.0 Training loss: 0.1721 Explore P: 0.0135
    Episode: 953 Total reward: 60.0 Training loss: 0.1799 Explore P: 0.0134
    Episode: 955 Total reward: 183.0 Training loss: 0.0768 Explore P: 0.0133
    Episode: 958 Total reward: 99.0 Training loss: 0.1003 Explore P: 0.0131
    Episode: 961 Total reward: 82.0 Training loss: 0.1984 Explore P: 0.0130
    Episode: 964 Total reward: 76.0 Training loss: 0.1628 Explore P: 0.0128
    Episode: 967 Total reward: 99.0 Training loss: 0.0715 Explore P: 0.0127
    Episode: 970 Total reward: 99.0 Training loss: 0.0881 Explore P: 0.0125
    Episode: 973 Total reward: 99.0 Training loss: 0.0688 Explore P: 0.0124
    Episode: 976 Total reward: 99.0 Training loss: 81.9871 Explore P: 0.0123
    Episode: 979 Total reward: 99.0 Training loss: 0.1413 Explore P: 0.0122
    Episode: 982 Total reward: 99.0 Training loss: 0.1866 Explore P: 0.0121
    Episode: 985 Total reward: 99.0 Training loss: 0.0578 Explore P: 0.0120
    Episode: 988 Total reward: 99.0 Training loss: 0.1633 Explore P: 0.0119
    Episode: 991 Total reward: 99.0 Training loss: 0.1042 Explore P: 0.0118
    Episode: 994 Total reward: 99.0 Training loss: 0.0792 Explore P: 0.0117
    Episode: 997 Total reward: 99.0 Training loss: 0.0502 Explore P: 0.0116


## Visualizing training

Below we plot the total rewards for each episode. The rolling average is plotted in blue.


```python
%matplotlib inline
import matplotlib.pyplot as plt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 
```


```python
eps, rews = np.array(rewards_list).T
smoothed_rews = running_mean(rews, 10)
plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
plt.plot(eps, rews, color='grey', alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
```




    Text(0,0.5,'Total Reward')




![png](output_21_1.png)


## Playing Atari Games

So, Cart-Pole is a pretty simple game. However, the same model can be used to train an agent to play something much more complicated like Pong or Space Invaders. Instead of a state like we're using here though, you'd want to use convolutional layers to get the state from the screen images.

![Deep Q-Learning Atari](assets/atari-network.png)

I'll leave it as a challenge for you to use deep Q-learning to train an agent to play Atari games. Here's the original paper which will get you started: http://www.davidqiu.com:8888/research/nature14236.pdf.
