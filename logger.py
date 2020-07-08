import matplotlib.pyplot as plt
import keras as K

from typing import Dict


class Logger:

    def __init__(self, logdir_path: str):
        self.history = []

        self.loss_value = []
        self.mean_gain = []

        self.logdir_path = logdir_path

    def create_settings_model_file(self, model: K.Model):
        def print_to_file(string: str):
            path_to_file = self.logdir_path + '/model_summary.txt'
            with open(path_to_file, 'w+') as file:
                print(string, file=file)
        model.summary(print_fn=print_to_file)

    def create_settings_agent_file(self, agent):
        path_to_file = self.logdir_path + '/agent_summary.txt'
        with open(path_to_file, 'w+') as file:
            file.write(str(agent))

    def add_event(self, event: Dict[str, None]):

        self.loss_value.append(event['loss_value'])
        self.mean_gain.append(event['mean_gain'])
        self.history.append(event)

        self._save_log_in_txt(event)
        self._save_loss_value_plot()
        self._save_gain_plot()

    def get_the_best_experiment(self, n: int = 1):
        """ n is number of bests"""
        if n == 1:
            return [max(self.history, key=lambda x: x['mean_gain'])]
        else:
            bests = []
            for event in self.history:
                if len(bests) < n:
                    bests.append(event)
                elif event['mean_gain'] > min(bests, key=lambda x: x['mean_gain']):
                    sorted(bests, key=lambda x: x['mean_gain'])
                    bests = bests[1:]
                    bests.append(event)
            return bests

    def _save_log_in_txt(self, event: Dict[str, None]):
        file_path = self.logdir_path + '/result.txt'
        with open(file_path, 'a') as file:
            for key, item in event.items():
                file.write(key + ': ' + str(item) + ' ')
            file.write('\n')

    def _save_loss_value_plot(self):
        plt.figure(figsize=(12, 12), dpi=80,)
        fig, ax = plt.subplots()
        ax.plot(self.loss_value)
        ax.set(xlabel='epochs', ylabel='loss value',
               title='Loss Value')
        fig.savefig(self.logdir_path + '/loss_value.png')

    def _save_gain_plot(self):
        plt.figure(figsize=(12, 12), dpi=80,)
        fig, ax = plt.subplots()
        ax.plot(self.mean_gain)
        ax.set(xlabel='epochs', ylabel='mean gain',
               title='Gain')
        fig.savefig(self.logdir_path + '/gain.png')
