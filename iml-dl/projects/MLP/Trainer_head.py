import torch

from core.Trainer import Trainer
from time import time
import wandb
import logging
from optim.losses.image_losses import *
import matplotlib.pyplot as plt
import copy
from torchmetrics.functional import confusion_matrix, accuracy
import pandas as pd
import seaborn as sn


class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)

        self.dict_classes = training_params['dict_classes']
        self.num_classes = len(self.dict_classes) 

        self.weights = torch.tensor([5.0,1.25]) if self.num_classes == 2 else torch.ones([1, self.num_classes])

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        """
        if model_state is not None:
            self.model.load_state_dict(model_state)  # load weights
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer

        epoch_losses = []
        epoch_accuracies = []

        self.early_stop = False

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_loss, batch_acc, count_images = 1.0, 1.0, 0

            for data in self.train_ds:
                # Input
                x = data[0].to(self.device)
                y = data[1].to(self.device)
                b = y.shape[0]
                count_images += b

                # Forward Pass
                self.optimizer.zero_grad()
                y_hat = self.model(x)

                if self.num_classes == 2:
                    y = (y >=1) * 1
                loss = F.cross_entropy(y_hat, y, reduction="mean", weight=self.weights.to(self.device))
                acc = accuracy(y_hat, y, average="macro", num_classes=self.num_classes)

                # Backward Pass
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # to avoid nan loss
                self.optimizer.step()
                batch_loss += loss.item() * b
                batch_acc += acc.item()

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_losses.append(epoch_loss)

            epoch_acc = batch_acc / count_images if count_images > 0 else batch_acc
            epoch_accuracies.append(epoch_acc)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})



            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                           ,'epoch': epoch}, self.client_path + '/latest_model.pt')

            # Run validation
            self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            task + '_loss': 0,
            task + '_acc': 0,
        }

        test_total = 0
        pred = []
        labels = []

        with torch.no_grad():
            for data in test_data:
                x = data[0].to(self.device)
                y = data[1].to(self.device)
                b = y.shape[0]
                test_total += b
                y = y.to(self.device)

                # Forward pass
                y_hat = self.test_model(x)

                pred.append(y_hat)
                labels.append(y)

                if self.num_classes == 2:
                    y = (y >=1) * 1
                loss = F.cross_entropy(y_hat, y, reduction="mean", weight=self.weights.to(self.device))
                acc = accuracy(y_hat, y, average="macro", num_classes=self.num_classes)

                metrics[task + '_loss'] += loss.item() * x.size(0)
                metrics[task + '_acc'] += acc.item() * x.size(0)

        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})

        epoch_val_loss = metrics[task + '_loss'] / test_total
        # epoch_val_loss = acc
        # min_val_loss = 0
        if task == 'Val':
            if epoch_val_loss < self.min_val_loss:
            #if acc > min_val_loss:
                self.min_val_loss = epoch_val_loss
                torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights, 'epoch': epoch},
                           self.client_path + '/best_model.pt')
                self.best_weights = copy.deepcopy(model_weights)
                self.best_opt_weights = copy.deepcopy(opt_weights)
            self.early_stop = self.early_stopping(epoch_val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_val_loss)