# ReinforcementLearningToy
A TensorFlow based DQN agent who moves in a small grid world

This repository was created in preparation of my MSc thesis, in order to put in practice the study of Deep Reinforcement Learning.

The task of the agent is to go within a 4x4 grid world from the top left corner to the bottom right, in the minimum number of steps possible. The agent gets a reward when it gets to the bottom right corner and it gets penalised for each step it does and in case it falls from the 4x4 grid.

Both the DQN agent and the environment are implemented using the `tf_agents` library.
