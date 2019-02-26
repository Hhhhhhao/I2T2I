import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Example Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target, lengths):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            if metric.__name__ == "bleu4":
                acc_metrics[i] += metric(output, target, lengths, self.data_loader.dataset.vocab)
            else:
                acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (_, batch_images, batch_captions, batch_caption_lengths) in enumerate(self.data_loader):

            batch_images = batch_images.to(self.device)
            batch_captions = batch_captions.to(self.device)
            # batch_caption_lengths = [l-1 for l in batch_caption_lengths]

            self.optimizer.zero_grad()
            outputs = self.model(batch_images, batch_captions, batch_caption_lengths)
            targets = pack_padded_sequence(batch_captions, batch_caption_lengths, batch_first=True)[0]
            loss = self.loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(outputs, targets, batch_caption_lengths)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                # self.writer.add_image('input', make_grid(batch_images.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (_, batch_images, batch_captions, batch_caption_lengths) in enumerate(self.valid_data_loader):
                batch_images = batch_images.to(self.device)
                batch_captions = batch_captions.to(self.device)

                outputs = self.model(batch_images, batch_captions, batch_caption_lengths)
                targets = pack_padded_sequence(batch_captions, batch_caption_lengths, batch_first=True)[0]
                loss = self.loss(outputs, targets)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(outputs, targets, batch_caption_lengths)
                # self.writer.add_image('input', make_grid(batch_images.cpu(), nrow=8, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }


