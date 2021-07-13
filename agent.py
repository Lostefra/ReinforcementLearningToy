import numpy as np
import tensorflow as tf
import logging

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import categorical_q_network, actor_distribution_network
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.eval.metric_utils import log_metrics

from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.losses import Huber
from tf_agents.utils import common


# #################### DQN AGENT ##################### #
def build_QNetwork(tf_env):
    preprocessing_layer = Lambda(lambda obs: tf.cast(tf.expand_dims(obs, axis=3), np.float32))  # bs, w, h, c_in
    conv_layer_params = [(1, (2, 2), 1)]  # (C_out, filter_size, stride)
    fc_layer_params = [256]  # hidden_units
    q_net = QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        # conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params)
    return q_net


def build_DqnAgent(q_net, tf_env):
    train_step = tf.Variable(0)
    update_period = 1  # run a training step every 1 collect step
    optimizer = RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0,
                        epsilon=0.00001, centered=True)
    epsilon_fn = PolynomialDecay(
        initial_learning_rate=1.0,  # initial ε
        decay_steps=1000 // update_period,
        end_learning_rate=0.01)  # final ε
    agent = DqnAgent(tf_env.time_step_spec(),
                     tf_env.action_spec(),
                     q_network=q_net,
                     optimizer=optimizer,
                     target_update_period=50,
                     # td_errors_loss_fn=Huber(reduction="none"),
                     td_errors_loss_fn=common.element_wise_squared_loss,
                     gamma=0.9,  # discount factor
                     train_step_counter=train_step,
                     epsilon_greedy=lambda: epsilon_fn(train_step))
    return agent
# ################################################################ #


# #################### CATEGORICAL DQN AGENT ##################### #
def build_categoricalDqnAgent(tf_env):
    fc_layer_params = [256]  # hidden_units
    num_atoms = 51
    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        num_atoms=num_atoms,
        fc_layer_params=fc_layer_params)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    train_step_counter = tf.compat.v2.Variable(0)
    agent = categorical_dqn_agent.CategoricalDqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        min_q_value=-100,
        max_q_value=100,
        n_step_update=5,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=0.9,
        train_step_counter=train_step_counter)
    return agent
# ################################################################ #


# ####################### REINFORCE AGENT ######################## #
def build_ReinforceAgent(tf_env):
    fc_layer_params = [256]  # hidden_units
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=fc_layer_params)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
    train_step_counter = tf.compat.v2.Variable(0)
    agent = reinforce_agent.ReinforceAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    return agent
# ################################################################ #


def build_replay_buffer(agent, tf_env, max_length=5000):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,  # bs = 1
        max_length=max_length)
    return replay_buffer


# Custom observer that counts and displays the number of times it is called
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 1 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


def build_train_metrics():
    logging.getLogger().setLevel(logging.INFO)  # for log_metrics
    return [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]


def build_collect_driver(agent, tf_env, replay_buffer_observer, train_metrics, update_period=1):
    return DynamicStepDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer_observer] + train_metrics,
        num_steps=update_period)  # collect 1 step for each training iteration


def driver_warm_up(tf_env, replay_buffer):
    initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                            tf_env.action_spec())
    init_driver = DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch, ShowProgress(500)],
        num_steps=500)
    final_time_step, final_policy_state = init_driver.run()


def train_agent(agent, tf_env, dataset, collect_driver, train_metrics, n_iterations=10000):
    history = {}
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 100 == 0:
            print()
            log_metrics(train_metrics)
            history[iteration] = "{:.5f}".format(train_loss.loss.numpy())
    return history


def eval_policy(policy, tf_env, num_episodes=1):
    for _ in range(num_episodes):
        print("=" * 10, "NEW EPISODE", "=" * 10)
        time_step = tf_env.reset()
        tf_env.render("human")
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            tf_env.render("human")
