from matplotlib import pyplot as plt
from ml.data_generator import DataGenerator
from torch.autograd import Variable
from ml.journal import DataframeJournal
from ml.progress_monitor import StdoutProgressMonitor
from ml.rate_controller import CosineRateController, ExponentialRateController

import math
import numpy as np
import pandas as pd
import tempfile
import torch


class Trainer:
    def __init__(self, *, model, training_data_loader, test_data_loader,
                 optimizer, journal=None, progress_monitor=None,
                 rate_controller=None):
        self.model = model

        self.training_data_loader = training_data_loader
        self.training_data_generator = DataGenerator(training_data_loader)

        self.test_data_loader = test_data_loader
        self.test_data_generator = DataGenerator(test_data_loader)

        if journal is None:
            journal = DataframeJournal()
        self.journal = journal

        if progress_monitor is None:
            progress_monitor = StdoutProgressMonitor()
        self.progress_monitor = progress_monitor

        if rate_controller is None:
            rate_controller = CosineRateController()
        self.rate_controller = rate_controller

        self.train_to_test_ratio = (len(training_data_loader) //
                                    len(test_data_loader))

        self.optimizer = optimizer

        self.checkpoint_filename = None

    def checkpoint(self, filename=None):
        filename = self.save_model_state(filename=filename)
        self.checkpoint_filename = filename
        return filename

    def save_model_state(self, filename=None):
        if filename is None:
            with tempfile.NamedTemporaryFile() as ofile:
                filename = ofile.name
        torch.save(self.model.state_dict(), filename)
        return filename

    def restore(self):
        if self.checkpoint_filename is None:
            raise RuntimeError("You haven't checkpointed this trainer's "
                    "model yet!")
        self.load_model_state(self.checkpoint_filename)

    def load_model_state(self, filename):
        self.model.load_state_dict(torch.load(filename))

    def multi_train(self, *, max_learning_rate,
            min_learning_rate=None, cycles=7,
            disable_progress_bar=False):
        for i in range(cycles):
            self.train(num_epochs=i+1,
                    max_learning_rate=max_learning_rate,
                    min_learning_rate=min_learning_rate,
                    disable_progress_bar=disable_progress_bar)

    def train(self, *, num_epochs, max_learning_rate=0.002,
            min_learning_rate=None, disable_progress_bar=False):
        rate_controller = CosineRateController()
        num_steps = math.ceil(num_epochs * len(self.training_data_loader))
        rate_controller.start_session(
                min_learning_rate=min_learning_rate,
                max_learning_rate=max_learning_rate,
                num_steps=num_steps)
        return self._train(num_steps=num_steps,
                rate_controller=rate_controller,
                disable_progress_bar=disable_progress_bar,
                journal=self.journal,
                progress_monitor=self.progress_monitor)

    def _train(self, *, num_steps, rate_controller,
            disable_progress_bar=False,
            journal, progress_monitor):
        iterations_since_test = self.train_to_test_ratio

        verbose = 0 if disable_progress_bar else 1
        progress_monitor.start_session(num_steps,
                metric_names={'training_total_loss': 'loss',
                              'test_total_loss': 'val_loss'},
                verbose=verbose)

        step_data = {}
        for i in range(num_steps):
            new_learning_rate = rate_controller.new_learning_rate(
                    step=i, data=step_data)
            self._update_learning_rate(new_learning_rate)

            training_step_dict = self._do_training_step()

            iterations_since_test += 1
            if iterations_since_test >= self.train_to_test_ratio:
                test_step_dict = self._do_test_step()
                iterations_since_test = 0

            step_data = {'learning_rate': new_learning_rate,
                    **training_step_dict,
                    **test_step_dict}

            journal.record_step(step_data)
            progress_monitor.step(step_data)

    def survey_learning_rate(self, *, num_epochs=1.0,
            min_learning_rate=1e-12,
            max_learning_rate=10,
            journal=None,
            progress_monitor=None):
        if journal is None:
            journal = DataframeJournal()

        if progress_monitor is None:
            progress_monitor = StdoutProgressMonitor()

        filename = self.save_model_state()

        num_steps = math.ceil(num_epochs * len(self.training_data_loader))
        rate_controller = ExponentialRateController()
        rate_controller.start_session(
                min_learning_rate=min_learning_rate,
                max_learning_rate=max_learning_rate,
                num_steps=num_steps)
        self._train(num_steps=num_steps,
                rate_controller=rate_controller,
                journal=journal,
                progress_monitor=progress_monitor)

        self.load_model_state(filename)

        return journal

    def _update_learning_rate(self, new_learning_rate):
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

        return {f'training_{k}': v for k, v in step_dict.items()}

    def _do_step(self, data, labels, train):
        run_dict = self._run_model(data, labels, train=train)
        loss_dict = self.model.calculate_losses(**run_dict)
        return {**loss_dict, **run_dict}

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
        return {f'test_{k}': v for k, v in step_dict.items()}

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
        return loss / len(self.test_data_loader)

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
