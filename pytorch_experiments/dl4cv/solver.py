from random import shuffle
import torch
from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
import time
import datetime
from tensorboardX import SummaryWriter

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss(), tensorboard=True, tb_name="SOlver"):
        self.tb = tensorboard
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        thedate = datetime.date.today()
        if self.tb:
            self.writer = SummaryWriter(comment=tb_name + str(thedate))
        self._reset_histories()



    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=1000):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """

        optimizer=self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        # iter_per_epoch = len(train_loader)

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = self.loss_func
        n_iter = 0  # global iteration counter

        if torch.cuda.is_available():
            model.cuda()
        since=time.time()

        print('START TRAIN.')
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 80)
            iter_= 0

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Select loader and iterate over data.
                if phase == "train":
                    loader = train_loader
                else:
                    loader = val_loader

                for data in loader:
                    # get the inputs
                    inputs, labels = data
                    # wrap them in Variable
                    if torch.cuda.is_available():
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    if n_iter == 1:
                        if self.tb:
                            self.writer.add_graph(model, outputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                    # statistics

                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds == labels.data)

                    if (phase == "train"):
                        iter_ += 1
                        n_iter += 1
                        if ((self.tb) & (n_iter % log_nth == 0)):
                            self.writer.add_scalar('result/train_loss', loss.data[0], n_iter)
                            self.writer.add_scalar('result/train_acc', torch.sum(preds == labels.data), n_iter)

                    # Do Tensorboard logging
                    if ((self.tb) & (n_iter % log_nth == 0)):
                        for name, param in model.named_parameters():
                            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)
                            if "weight" in name:
                                self.writer.add_scalar("weight/mean/" + name, torch.mean(param).clone().cpu().data.numpy(), n_iter)
                                self.writer.add_scalar("weight/var/" + name, torch.var(param).clone().cpu().data.numpy(), n_iter)
                                try:
                                    self.writer.add_scalar("grad/var/" + name, torch.var(param.grad).clone().cpu().data.numpy(), n_iter)
                                    self.writer.add_scalar("grad/mean/" + name, torch.mean(param.grad).clone().cpu().data.numpy(), n_iter)
                                except:
                                    pass

                    # Save Iteration_info
                    if ((phase == "train") & (iter_ % log_nth == 0)):
                        print('[Iteration: ' + str(iter_ * len(labels)) + '/' + str(len(loader.dataset)) + '] TRAIN loss: ' + str(loss.data[0]))
                        self.train_loss_history.append(loss.data[0])

                epoch_loss = running_loss / (len(loader) * 100)
                epoch_acc = running_corrects / (len(loader) * 100)

                if (phase == "train"):
                    self.train_acc_history.append(epoch_acc)
                else:
                    self.val_acc_history.append(epoch_acc)
                    self.val_loss_history.append(epoch_loss)
                    # for name, param in model.named_parameters():
                    #     print(name + "para: " + str( param.clone().cpu().data.numpy()))
                    if self.tb:
                        self.writer.add_scalar('result/val_loss', running_loss, len(self.val_loss_history))
                        self.writer.add_scalar('result/val_acc', epoch_acc, len(self.val_acc_history))
                    # self.writer.add_embedding(inputs.data.view, metadata=preds, label_img=outputs)
                print('[{}] Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('FINISH.')
        if (self.tb):
            self.writer.close()


    def hypertune():
        pass