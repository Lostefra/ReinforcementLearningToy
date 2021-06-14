from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.utils.common import function

import matplotlib.pyplot as plt
import numpy as np

from environment import TestEnvironment
from agent import build_QNetwork, build_DqnAgent, build_replay_buffer, build_train_metrics, build_collect_driver, \
    driver_warm_up, train_agent, eval_policy

if __name__ == '__main__':
    env = TestEnvironment(discount=0.9)
    env.seed(42)
    tf_env = TFPyEnvironment(env)
    print("Environment: ok")

    q_net = build_QNetwork(tf_env)
    dqn_agent = build_DqnAgent(q_net, tf_env)
    dqn_agent.initialize()
    print("Agent: ok")

    replay_buffer = build_replay_buffer(dqn_agent, tf_env, 5000)
    replay_buffer_observer = replay_buffer.add_batch
    print("Buffer: ok")

    train_metrics = build_train_metrics()
    collect_driver = build_collect_driver(dqn_agent, tf_env, replay_buffer_observer, train_metrics, update_period=1)
    print("Starting warm up...")
    driver_warm_up(tf_env, replay_buffer)
    print("\nWarm up: ok")

    # Inspect replay_buffer content, further inspection: e.g. tf.print(replay_buffer.gather_all().action, summarize=-1)
    # print(replay_buffer.gather_all())

    dataset = replay_buffer.as_dataset(
        sample_batch_size=32,
        num_steps=2,  # 2 steps trajectory = 1 full transition, including the next step's observation
        num_parallel_calls=3,
        single_deterministic_pass=False).prefetch(3)

    # To speed up training, convert the main functions to TensorFlow functions
    collect_driver.run = function(collect_driver.run)
    dqn_agent.train = function(dqn_agent.train)

    print("Starting training...")
    history = train_agent(dqn_agent, tf_env, dataset, collect_driver, train_metrics, n_iterations=1800)
    print("\nTraining is over!")

    eval_policy(dqn_agent.policy, tf_env, num_episodes=1)

    # Plot loss history
    plt.plot(np.fromiter(history.keys(), dtype=float), np.fromiter(history.values(), dtype=float))
    plt.show()

    tf_env.close()
