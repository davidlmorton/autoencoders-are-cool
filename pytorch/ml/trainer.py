from matplotlib import pyplot as plt
from ml.data_generator import DataGenerator
from ml.utils import Progbar
from torch.autograd import Variable

import math
import pandas as pd
import numpy as np


class Trainer:
    def __init__(self, *, model, training_data_loader, test_data_loader,
                 optimizer, loss_alpha=0.1):
        self.model = model

        self.training_data_loader = training_data_loader
        self.training_data_generator = DataGenerator(training_data_loader)

        self.test_data_loader = test_data_loader
        self.test_data_generator = DataGenerator(test_data_loader)

        self.train_to_test_ratio = (len(training_data_loader) //
                                    len(test_data_loader))

        self.optimizer = optimizer
        self.loss_alpha = loss_alpha

        self.reset_history()

    def reset_history(self):
        self.last_avg_training_loss = 0.0
        self.last_avg_test_loss = 0.0
        self.history_df = None

    def plot_history(self, title="Training History", figsize=(15, 5),
                    skip_first=200, fig=None):
        if fig is None:
            fig = plt.figure(figsize=figsize)

        history_df = self.history_df.loc[skip_first:]

        ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3, fig=fig)
        history_df.training_losses.plot(ax=ax, color='mediumseagreen',
                                             label='Training Loss')
        history_df.test_losses.plot(ax=ax, color='tomato',
                                    label='Test Loss')
        ax.set_title(title)
        ax.legend()

        ax = plt.subplot2grid((4, 1), (3, 0), fig=fig)
        history_df.learning_rates.plot(ax=ax, color='dodgerblue',
                                       label='Learning Rate')
        ax.legend()

        plt.tight_layout()
        return fig

    def multi_train(self, *, learning_rate, cycles=7,
            disable_progress_bar=False):
        for i in range(cycles):
            self.train(num_epochs=i+1, max_learning_rate=learning_rate,
                      disable_progress_bar=disable_progress_bar)

    def train(self, *, num_epochs, max_learning_rate=0.002,
            min_learning_rate=None, disable_progress_bar=False):
        if min_learning_rate is None:
            min_learning_rate = max_learning_rate / 50.0

        avg_training_loss = self.last_avg_training_loss
        avg_test_loss = self.last_avg_test_loss
        iterations_since_test = 0
        learning_rates = []
        training_losses = []
        test_losses = []

        num_iterations = math.ceil(num_epochs * len(self.training_data_loader))

        verbose = 0 if disable_progress_bar else 1
        progress_bar = Progbar(num_iterations,
                stateful_metrics=['loss', 'val_loss'], verbose=verbose)
        for i in range(num_iterations):
            new_learning_rate = self._update_learning_rate(
                min_learning_rate=min_learning_rate,
                max_learning_rate=max_learning_rate,
                progression=i/num_iterations)

            training_loss = self._do_training_step()

            avg_training_loss = ((self.loss_alpha * training_loss) +
                    (1.0 - self.loss_alpha) * avg_training_loss)
            self.last_avg_training_loss = avg_training_loss

            iterations_since_test += 1
            if iterations_since_test >= self.train_to_test_ratio:
                test_loss = self._do_test_step()
                avg_test_loss = ((self.loss_alpha * test_loss) +
                        (1.0 - self.loss_alpha) * avg_test_loss)
                self.last_avg_test_loss = avg_test_loss

                iterations_since_test = 0

            learning_rates.append(new_learning_rate)
            training_losses.append(avg_training_loss)
            test_losses.append(avg_test_loss)

            progress_bar.add(1,
                [('loss', avg_training_loss),
                 ('val_loss', avg_test_loss)])

        result = {'learning_rates': learning_rates,
                'training_losses': training_losses,
                'test_losses': test_losses}
        history_df = pd.DataFrame.from_dict(result)
        if self.history_df is None:
            self.history_df = history_df
        else:
            self.history_df = pd.concat([self.history_df, history_df],
                                       ignore_index=True)

        return result

    def _update_learning_rate(self, *, min_learning_rate,
            max_learning_rate, progression):
        new_learning_rate = (min_learning_rate +
                (max_learning_rate - min_learning_rate) *
                (1 + math.cos(math.pi * progression)) / 2)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate
        return new_learning_rate

    def _do_training_step(self):
        self.optimizer.zero_grad()

        data, labels = self.training_data_generator.next()
        step_dict = self._do_step(data, labels, train=True)
        loss = step_dict['total_loss']

        loss.backward()
        self.optimizer.step()

        return loss.data[0] / len(data)

    def _do_step(self, data, labels, train):
        run_dict = self._run_model(data, labels, train=train)
        losses = self.model.calculate_losses(**run_dict)
        return {**losses, **run_dict}

    def _run_model(self, data, labels, train):
        self.model.train(train)
        x_in = Variable(data).cuda()
        y_in = Variable(labels).cuda()
        in_dict = {'x_in': x_in, 'y_in': y_in}

        out_dict = self.model(**in_dict)
        return {**out_dict, **in_dict}

    def _do_test_step(self):
        data, labels = self.test_data_generator.next()
        step_dict = self._do_step(data, labels, train=False)
        self.model.train(True)
        return step_dict['total_loss'].data[0] / len(data)

    @property
    def num_trainable_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad,
                self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def test(self):
        loss = 0.0
        for data, labels in self.test_data_loader:
            step_dict = self._do_step(data, labels, train=False)
            loss += step_dict['total_loss'].data[0]
        return loss / len(self.test_data_loader.dataset)

    def plot_input_output_pairs(self, title='A Sampling of Autoencoder Results',
        num_cols=10, figsize=(15, 3.2)):
        data, labels = self.test_data_generator.next()
        out_dict = self._run_model(data, labels, train=True)
        x_out = out_dict['x_out']

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=20)

        for i in range(num_cols):
            input_image = data[i][0]
            output_image = x_out.view_as(data).data.cpu()[i][0]

            ax = fig.add_subplot(2, num_cols, i+1)
            ax.imshow(input_image, cmap='gray')
            if i == 0:
                ax.set_ylabel('Input')
            else:
                ax.axis('off')

            ax = fig.add_subplot(2, num_cols, num_cols+i+1)
            ax.imshow(output_image, cmap='gray')
            if i == 0:
                ax.set_ylabel('Output')
            else:
                ax.axis('off')
        return fig

    def plot_latent_space(self, title="Latent Representation", figsize=(8, 8)):
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=20)

        self.model.train(False)
        for data, labels in self.test_data_loader:
            x_in = Variable(data).cuda()
            y_in = Variable(labels).cuda()
            in_dict = {'x_in': x_in, 'y_in': y_in}

            x_latent = self.model.encode(**in_dict)

            x_latent_numpy = x_latent.cpu().data.numpy()
            plt.scatter(x=x_latent_numpy.T[0], y=x_latent_numpy.T[1],
                        c=labels.numpy(), alpha=0.4)
        plt.colorbar()
        return fig

    def get_classification_accuracy(self):
        num_correct = 0
        for data, labels in self.test_data_loader:
            out_dict = self._run_model(data, labels, train=False)
            y_out = np.argmax(out_dict['y_out'].cpu().data.numpy(), axis=1)
            y_in = out_dict['y_in'].cpu().data.numpy()
            num_correct += (y_out - y_in == 0).sum()
        return num_correct / len(self.test_data_loader.dataset)
