import gym
from tensorflow import convert_to_tensor, GradientTape
from tensorflow.keras.activations import softplus, relu
from tensorflow.keras.backend import expand_dims, clip
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow_probability.python.distributions import Normal

env = gym.make('MountainCarContinuous-v0')
state = env.reset()
state_input = Input(state.shape, name='state_input')
actor_dense1_layer = Dense(4, activation=relu, name='actor_dense1')
actor_dense1 = actor_dense1_layer(state_input)
actor_out_layer = Dense(2, activation=None, name='actor_nn_out')
actor_out = actor_out_layer(actor_dense1)
mean = actor_out[:, 0]
std = softplus(actor_out[:, 1])
dist = Normal(mean, std, name='dist')
action = dist.sample((), name='sample')
action_log_prob = dist.log_prob(action, name='log_prob')
action_log_prob = expand_dims(action_log_prob)
action = expand_dims(action)
action = clip(action, -1.0, 1.0)
critic_concat = Concatenate()([state_input, action])
critic_dense1_layer = Dense(units=8, activation=relu, name='critic_dense1')
critic_dense1 = critic_dense1_layer(critic_concat)
critic_out_layer = Dense(1, activation=None)
critic_out = critic_out_layer(critic_dense1)
actor_critic_model = Model(inputs=[state_input], outputs=[action, action_log_prob, critic_out])
critic_loss_op = MeanSquaredError(name='critic_loss')
actor_vars = actor_dense1_layer.trainable_variables + actor_out_layer.trainable_variables
critic_vars = critic_dense1_layer.trainable_variables + critic_out_layer.trainable_variables
GAMES = 100
LR = 1e-4
GAMMA = 0.99
for _ in range(GAMES):
    done = False
    state = env.reset()
    while not done:
        with GradientTape(persistent=True) as tape:
            env.render()
            action_out, action_log_prob_out, value = actor_critic_model(convert_to_tensor([state]))
            next_state, reward, done, _ = env.step(action_out[0])
            _, _, value_next = actor_critic_model(convert_to_tensor([next_state]))
            value_target = reward + (value_next * GAMMA)
            critic_loss = critic_loss_op(value_target, value)
        critic_grads = tape.gradient(critic_loss, critic_vars)
        actor_grads = tape.gradient(action_log_prob_out, actor_vars)
        for g, v in zip(critic_grads, critic_vars):
            v.assign_sub(g * LR)
        for g, v in zip(actor_grads, actor_vars):
            v.assign_add(g * value[0] * LR)
        state = next_state
