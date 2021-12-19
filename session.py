from time import sleep
import matplotlib.pyplot as plt
import numpy as np

# Smallest number such that 1.0 + eps != 1.0
EPS = np.finfo(np.float32).eps.item()

ACTIONS = [0, 1]
NUM_INPUTS = 4
ENV_NAME = "CartPole-v0"


class Session:
    def __init__(self,
                 model,
                 game,
                 max_steps_per_episode: int = 1000) -> None:
        self.model = model
        self.game = game
        self.max_steps_per_episode = max_steps_per_episode

    def run(self, reward_target):
        running_reward = 0
        episode_count = 0
        reward_history = []

        _, ax = plt.subplots()
        ax.set_xlabel("Episódio")
        ax.set_ylabel("Recompensa")
        ax.set_title("Recompensa por episódio")

        while True:
            running_reward = self.model.train(self.game,
                                              self.max_steps_per_episode,
                                              running_reward)
            episode_count += 1
            reward_history.append(running_reward)

            ax.plot(reward_history, "b")
            plt.pause(0.001)
            
            if running_reward >= reward_target:  
                print(f"Resolvido no episódio {episode_count}!")
                break

        plt.savefig("reward_history.png")
        return model
        

if __name__ == "__main__":
    from game import Game, RandomPlayer
    from model import Model

    model = Model()
    game = Game()

    n_iter = game.run_until_done(RandomPlayer(), sleep=0.1)
    print(f"Jogador aleatório conseguiu {n_iter} passos.")

    session = Session(model, game)
    trained_model = session.run(reward_target=195)
    trained_model.checkpoint("model.h5")

    input("Pressione ENTER para exibir resultado...")

    n_iter = game.run_until_done(trained_model, sleep=0.1)
    print(f"Jogador treinado conseguiu {n_iter} passos.")
