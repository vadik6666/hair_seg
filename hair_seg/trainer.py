import os
import math
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from utils import create_figure
from loss import iou_loss, HairMattingLoss

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class Trainer:
    def __init__(self, args, model, checkpoint_mng):
        self.model = model

        self.checkpoint_mng = checkpoint_mng

        if args.optimizer == "adam":
            print("Adam optimizer")
            self.optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)
        elif args.optimizer == "sgd":
            print("SGD optimizer")
            self.optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        # trained epochs
        self.trained_epoch = 0
        self.train_loss = {"loss": [], "iou": []}
        self.dev_loss = {"loss": [], "iou": []}

        self.loss = HairMattingLoss(args.grad_lambda).to(DEVICE)

    def log(self, *args):
        """formatted log output for training"""

        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{time}   ", *args)

    def resume(self, checkpoint):
        """load checkpoint"""

        self.trained_epoch = checkpoint["epoch"]
        self.train_loss = checkpoint["train_loss"]
        self.dev_loss = checkpoint["dev_loss"]
        self.optimizer.load_state_dict(checkpoint["opt"])

    def reset_epoch(self):
        self.trained_epoch = 0
        self.train_loss = []
        self.dev_loss = []

    def train_batch(self, training_batch, tf_rate=1, val=False):
        # extract fields from batch & set DEVICE options
        image, mask = (i.to(DEVICE) for i in training_batch)
        pred = self.model(image)
        loss = self.loss(pred, mask, image)

        # if in training, not validate
        if not val:
            # Zero gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Adjust model weights
            self.optimizer.step()

        iou = iou_loss(pred, mask)

        return loss.item(), iou.item(), pred

    def train(self, args, train_data, dev_data=None):
        n_epochs = args.ep
        batch_size = args.bs
        stage = args.mode
        start_epoch = self.trained_epoch + 1
        # tensorboard on
        writer = SummaryWriter("tb/{}".format(args.model_name))

        # Data loaders with custom batch builder
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.workers)

        # Learning rate scheduler initializing
        if args.lr_schedule == "multi_step_lr":
            print("Multi Step LR scheduler")
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 14], gamma=0.1)
        elif args.lr_schedule == "cosine":
            print("Cosine LR scheduler")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, n_epochs * len(trainloader), eta_min=1e-5, last_epoch=-1
            )

        curent_iteration = 0  # global counter of training iteration

        self.log(f"Start training from epoch {start_epoch} to {n_epochs}...")

        for epoch in range(start_epoch, n_epochs + 1):
            self.model.train()
            loss_sum, iou_sum = 0, 0

            for idx, training_batch in enumerate(trainloader):
                curent_iteration = (epoch - 1) * len(trainloader) + idx

                if curent_iteration < args.wup:
                    self.adjust_warmup_lr(curent_iteration, args.wup, args.lr)

                lr = self.optimizer.param_groups[0]["lr"]

                # run a training iteration with batch
                loss, iou, pred = self.train_batch(training_batch)

                # Accumulate losses to print
                loss_sum += loss
                iou_sum += iou

                # Print progress
                iteration = idx + 1
                if iteration % args.print_freq == 0:
                    avg_loss = loss_sum / iteration
                    avg_iou = iou_sum / iteration

                    self.log(
                        "Epoch {}; Iter: {}({}); LR: {:0.4g}; Percent: {:.1f}%; \
                        Avg loss: {:.4f}; Avg IOU: {:.4f};".format(
                            epoch,
                            iteration,
                            curent_iteration + 1,
                            lr,
                            iteration / len(trainloader) * 100,
                            avg_loss,
                            avg_iou,
                        )
                    )

                    writer.add_scalar("train/loss", avg_loss, curent_iteration)
                    writer.add_scalar("train/iou", avg_iou, curent_iteration)
                    writer.add_scalar("train/lr", lr, curent_iteration)

                if args.lr_schedule == "cosine":
                    # adjust learning rate
                    if curent_iteration >= args.wup:
                        scheduler.step()

            if args.lr_schedule == "multi_step_lr":
                # adjust learning rate
                if curent_iteration >= args.wup:
                    scheduler.step()

            self.trained_epoch = epoch
            self.train_loss["loss"].append(loss_sum / len(trainloader))
            self.train_loss["iou"].append(iou_sum / len(trainloader))

            if dev_data:
                loss_sum, iou_sum = 0, 0

                self.model.eval()

                devloader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

                for i, dev_batch in enumerate(devloader):
                    loss, iou, pred = self.train_batch(dev_batch, val=True)

                    # Accumulate losses to print
                    loss_sum += loss
                    iou_sum += iou

                avg_loss = loss_sum / len(devloader)
                avg_iou = iou_sum / len(devloader)

                self.log("Validation; Epoch {}; Avg loss: {:.4f}; Avg IOU: {:.4f};".format(epoch, avg_loss, avg_iou))
                writer.add_scalar("val/loss", avg_loss, epoch)
                writer.add_scalar("val/iou", avg_iou, epoch)

                self.dev_loss["loss"].append(avg_loss)
                self.dev_loss["iou"].append(avg_iou)

            # Save checkpoint
            if epoch % args.save_freq == 0:
                cp_name = f"{stage}_{epoch}"
                self.checkpoint_mng.save(
                    cp_name,
                    {
                        "epoch": epoch,
                        "train_loss": self.train_loss,
                        "dev_loss": self.dev_loss,
                        "model": self.model.state_dict(),
                        "opt": self.optimizer.state_dict(),
                    },
                )

                self.log("Save checkpoint:", cp_name)

        # tensorboard off
        writer.close()

    def save_sample_imgs(self, img, mask, prediction, epoch, iter):
        fig = create_figure(img, mask, prediction.float())

        self.checkpoint_mng.save_image(f"{epoch}-{iter}", fig)
        plt.savefig("result_deb.jpg")
        plt.close(fig)

    def adjust_warmup_lr(self, iteration, warmup_iterations, lr):
        """Warm up learning rate"""
        if warmup_iterations == 0:
            return

        factor = (iteration + 1) / warmup_iterations

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = factor * lr
