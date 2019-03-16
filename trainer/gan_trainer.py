import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.nn.utils.rnn import pack_padded_sequence
from base import BaseGANTrainer
from utils import get_caption_lengths, convert_back_to_text


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class Trainer(BaseGANTrainer):
    """
    Example Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, models, optimizers, losses, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):

        super(Trainer, self).__init__(
            generator=models["generator"],
            discriminator=models["discriminator"],
            generator_optimizer=optimizers["generator"],
            discriminator_optimizer=optimizers["discriminator"],
            losses=losses,
            metrics=metrics,
            resume=resume,
            config=config,
            train_logger=train_logger)

        self.config = config
        self.train_data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = False # self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.image_size = self.config["train_data_loader"]["args"]["image_size"]
        # loss weight for generator cross entropy loss
        self.lambda_1 = self.config["trainer"]["lambda_1"]
        # loss weight for generator roll out loss
        self.lambda_2 = self.config["trainer"]["lambda_2"]

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_generator_epoch(self, epoch):
        self.generator.train()
        total_loss = 0.0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.train_data_loader):
            batch_images = data["right_images_{}".format(self.image_size)].to(self.device)
            batch_captions = data["right_captions"].to(self.device)
            batch_caption_lengths = data["right_caption_lengths"].to(self.device)

            self.generator_optimizer.zero_grad()
            _, _, outputs = self.generator(batch_images, batch_captions, batch_caption_lengths)
            targets = pack_padded_sequence(batch_captions, batch_caption_lengths, batch_first=True)[0]
            loss = self.losses["Generator_CrossEntropyLoss"](outputs, targets)
            loss.backward()
            self.generator_optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('Generator_CrossEntropyLoss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(outputs, targets)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Generator Pre-Train Epoch: {} [{}/{} ({:.0f}%)] CCE Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.train_data_loader.batch_size,
                    self.train_data_loader.n_samples,
                    100.0 * batch_idx / len(self.train_data_loader),
                    loss.item()))

        self.predict(self.valid_data_loader, epoch, name='pretrain_epoch')

        log = {
            'Generator_CrossEntropyLoss': total_loss / len(self.train_data_loader),
            'metrics': (total_metrics / len(self.train_data_loader)).tolist()
        }
        return log

    def _train_discriminator_epoch(self, epoch):
        """
        Pre training logic for an epoch
        :param epoch: Current training epoch
        :return: A log that contrains all information you want to save
        Note:
        If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.generator.eval()
        self.discriminator.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.train_data_loader):
            batch_images = data["right_images_{}".format(self.image_size)].to(self.device)
            batch_captions = data["right_captions"].to(self.device)
            batch_caption_lengths = data["right_caption_lengths"].to(self.device)
            other_captions = data["wrong_captions"].to(self.device)
            other_caption_lengths = data["wrong_caption_lengths"].to(self.device)

            # use generator to generator image features and captions (one-hot)
            image_features, features, generator_outputs = self.generator(batch_images, batch_captions, batch_caption_lengths)
            generator_captions = self.generator.module.feature_to_text(features)
            generator_captions, generator_caption_lengths = get_caption_lengths(generator_captions)
            generator_captions.to(self.device)
            generator_caption_lengths.to(self.device)

            self.discriminator_optimizer.zero_grad()
            evaluator_scores = self.discriminator(image_features.detach(), batch_captions, batch_caption_lengths)
            generator_scores = self.discriminator(image_features.detach(), generator_captions, generator_caption_lengths)
            other_scores = self.discriminator(image_features.detach(), other_captions, other_caption_lengths)
            loss = self.losses["Discriminator_Loss"](evaluator_scores, generator_scores, other_scores)
            loss.backward()
            self.discriminator_optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('Evaluator_Loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(generator_scores, evaluator_scores)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Discriminator Pre-Train Epoch: {} [{}/{} ({:.0f}%)] Evaluator Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.train_data_loader.batch_size,
                    self.train_data_loader.n_samples,
                    100.0 * batch_idx / len(self.train_data_loader),
                    loss.item()))

            if batch_idx == int(len(self.train_data_loader) / 4):
                if "CoCo" in self.config["name"]:
                    break
        log = {
            'Evaluator_Loss': total_loss / len(self.train_data_loader),
            'metrics': (total_metrics / len(self.train_data_loader)).tolist()
        }

        for p in self.generator.parameters():
            p.requires_grad = True
        return log

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

        self.generator.train()
        self.discriminator.train()

        total_generator_rl_loss = 0.0
        total_generator_cce_loss = 0.0
        total_generator_loss = 0.0
        total_discriminator_loss = 0.0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.train_data_loader):

            if batch_idx == len(self.train_data_loader) -1 :
                break

            batch_images = data["right_images_{}".format(self.image_size)].to(self.device)
            batch_captions = data["right_captions"].to(self.device)
            batch_caption_lengths = data["right_caption_lengths"].to(self.device)
            other_captions = data["wrong_captions"].to(self.device)
            other_caption_lengths = data["wrong_caption_lengths"].to(self.device)

            # train the discriminator first
            for p in self.discriminator.parameters():
                p.requires_grad = True

            # forward
            image_features, features, outputs = self.generator(batch_images, batch_captions, batch_caption_lengths)
            generator_captions = self.generator.module.feature_to_text(features)
            generator_captions, generator_caption_lengths = get_caption_lengths(generator_captions)
            generator_captions.to(self.device)
            generator_caption_lengths.to(self.device)

            # D backward
            self.discriminator_optimizer.zero_grad()
            evaluator_scores = self.discriminator(image_features.detach(), batch_captions, batch_caption_lengths)
            generator_scores = self.discriminator(image_features.detach(), generator_captions, generator_caption_lengths)
            other_scores = self.discriminator(image_features.detach(), other_captions, other_caption_lengths)
            discriminator_loss = self.losses["Discriminator_Loss"](evaluator_scores, generator_scores, other_scores)
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            # G backward
            self.generator_optimizer.zero_grad()
            targets = pack_padded_sequence(batch_captions, batch_caption_lengths, batch_first=True)[0]
            generator_cce_loss = self.losses["Generator_CrossEntropyLoss"](outputs, targets)
            rewards, props = self.generator.module.reward_forward(image_features, self.discriminator, monte_carlo_count=16)
            generator_rl_loss = self.losses["Generator_RLLoss"](rewards, props)
            generator_loss = self.lambda_1 * generator_cce_loss + self.lambda_2 * generator_rl_loss
            generator_loss.backward()
            self.generator_optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('Generator_Total_Loss', generator_loss.item())
            self.writer.add_scalar('Generator_CrossEntropyLoss', generator_cce_loss.item())
            self.writer.add_scalar('Generator_RLLoss', generator_rl_loss.item())
            self.writer.add_scalar('Evaluator_Loss', discriminator_loss.item())
            total_generator_loss += generator_loss.item()
            total_generator_cce_loss += generator_cce_loss.item()
            total_generator_rl_loss += generator_rl_loss.item()
            total_discriminator_loss += discriminator_loss.item()
            total_metrics += self._eval_metrics(outputs, targets)
            
            # break            

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] '
                                 'Generator Loss: [CCE:{:.6f}, RL:{:.6f}, Total:{:.6f}] '
                                 'Discriminator Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.train_data_loader.batch_size,
                    self.train_data_loader.n_samples,
                    100.0 * batch_idx / len(self.train_data_loader),
                    generator_cce_loss.item(),
                    generator_rl_loss.item(),
                    generator_loss.item(),
                    discriminator_loss.item()
                ))

        log = {
            'Generator_CrossEntropyLoss': total_generator_cce_loss / len(self.train_data_loader),
            'Generator_RLLoss': total_generator_rl_loss / len(self.train_data_loader),
            'Generator_Total_Loss': total_generator_loss / len(self.train_data_loader),
            'Discriminator_Loss': total_discriminator_loss / len(self.train_data_loader),
            'metrics': (total_metrics / len(self.train_data_loader)).tolist()
        }

        self.predict(self.valid_data_loader, epoch)

        if self.do_validation:
            pass
            # val_log = self._valid_epoch(epoch)
            # log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.generator.eval()
        self.discriminator.eval()
        total_generator_val_loss = 0.0
        total_discriminator_val_loss = 0.0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                batch_images = data["right_images_{}".format(self.image_size)].to(self.device)
                batch_captions = data["right_captions"].to(self.device)
                batch_caption_lengths = data["right_caption_lengths"].to(self.device)
                other_captions = data["wrong_captions"].to(self.device)
                other_caption_lengths = data["wrong_caption_lengths"].to(self.device)

                image_features, outputs = self.generator(batch_images, batch_captions, batch_caption_lengths)
                targets = pack_padded_sequence(batch_captions, batch_caption_lengths, batch_first=True)[0]
                generator_cce_loss = self.losses["Generator_CrossEntropyLoss"](outputs, targets)

                rewards, props = self.generator.module.reward_forward(batch_images, self.discriminator, monte_carlo_count=16)
                generator_rl_loss = self.losses["Generator_RLLoss"](rewards, props)
                generator_loss = self.lambda_1 * generator_cce_loss + self.lambda_2 * generator_rl_loss

                # generator_captions = []
                # for image_feature in image_features:
                #     generator_captions.append(
                #         self.generator.module.sample(image_feature.unsqueeze(0)))
                # generator_captions, generator_caption_lengths = get_caption_lengths(generator_captions)
                # generator_captions.to(self.device)
                # generator_caption_lengths.to(self.device)
                #
                # evaluator_scores = self.discriminator(batch_images, batch_captions, batch_caption_lengths)
                # generator_scores = self.discriminator(batch_images, generator_captions, generator_caption_lengths)
                # other_scores = self.discriminator(batch_images, other_captions, other_caption_lengths)
                # discriminator_loss = self.losses["Discriminator_Loss"](evaluator_scores, generator_scores, other_scores)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('Generator_Total_Loss', generator_loss.item())
                # self.writer.add_scalar('Discriminator_Total_Loss', discriminator_loss.item())
                total_generator_val_loss += generator_loss.item()
                # total_discriminator_val_loss += discriminator_loss.item()
                total_val_metrics += self._eval_metrics(image_features, outputs)

        return {
            'generator_val_loss': total_generator_val_loss / len(self.valid_data_loader),
            # 'discriminator_val_loss': total_generator_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def predict(self, data_loader, epoch=None, name='epoch'):
        self.generator.eval()
        self.discriminator.eval()

        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

        transform = transforms.Compose([
            transforms.Normalize(mean=(-mean / std).tolist(), std=(1.0 / std).tolist()),
            transforms.ToPILImage()]
        )

        for batch_idx, data in enumerate(data_loader):
            if batch_idx == 1:
                break

            batch_images = data["right_images_{}".format(self.image_size)].to(self.device)
            batch_captions = data["right_captions"].to(self.device)
            batch_caption_lengths = data["right_caption_lengths"].to(self.device)

            if not os.path.exists('{0}/results/{1}_{2}'.format(self.checkpoint_dir, name, epoch)):
                os.makedirs('{0}/results/{1}_{2}'.format(self.checkpoint_dir, name, epoch))

            image_features, features, outputs = self.generator(batch_images, batch_captions, batch_caption_lengths)
            generator_captions = self.generator.module.feature_to_text(features)

            for generated_caption, image in zip(generator_captions, batch_images):
                generated_sentence = convert_back_to_text(generated_caption, self.train_data_loader.dataset.vocab)
                image = transform(image.cpu())
                image.save('{0}/results/{1}_{2}/{3}.png'.format(self.checkpoint_dir, name, epoch, generated_sentence.replace("/", "")[:100]))


