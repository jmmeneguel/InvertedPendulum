from typing import Callable
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from session import EPS, NUM_INPUTS, ACTIONS


class Model(keras.Model):
    def __init__(self,
                 num_hidden: int = 128,
                 num_inputs=NUM_INPUTS,
                 gamma: float = 0.99):
        super(Model, self).__init__()

        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        action = layers.Dense(len(ACTIONS), activation="softmax")(common)
        critic = layers.Dense(1)(common)

        self.model = keras.Model(inputs=inputs, outputs=[action, critic])

        self.loss_model = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.gamma = gamma


    def call(self, x):
        x = tf.convert_to_tensor(x)
        x = tf.expand_dims(x, 0)
        x = self.model(x)
        return x

    def predict(self, x):
        return self.call(x)

    def loss(self, value, ret):
        return self.loss_model(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))

    def checkpoint(self, filepath):
        self.model.save_weights(filepath)

    def apply_gradients(self, values):
        self.optimizer.apply_gradients(values)

    def train(self, game, max_steps_per_episode, running_reward):
        action_probs_history = []
        critic_value_history = []
        rewards_history = []
        episode_reward = 0

        state = game.reset()

        with tf.GradientTape() as tape:
            for i in range(1, max_steps_per_episode):
                action_probs, critic_value = self(state)
                critic_value_history.append(critic_value[0, 0])

                action = np.random.choice(
                    game.num_actions, p=np.squeeze(action_probs))
                action_probs_history.append(
                    tf.math.log(action_probs[0, action]))

                state, reward, done, _ = game.step(action)
                rewards_history.append(reward)
                episode_reward += reward

                if done:
                    break

            running_reward = 0.05 * episode_reward + \
                (1 - 0.05) * running_reward

            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + self.gamma * discounted_sum
                returns.insert(0, discounted_sum)

            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / \
                (np.std(returns) + EPS)
            returns = returns.tolist()

            history = zip(action_probs_history,
                          critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                critic_losses.append(
                    self.loss(value, ret)
                )

            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, self.trainable_variables)
            self.apply_gradients(
                zip(grads, self.trainable_variables))

            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

        return running_reward
