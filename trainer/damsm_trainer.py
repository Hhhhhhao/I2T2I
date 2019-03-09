import numpy as np
import torch
import skimage
from PIL import Image, ImageDraw, ImageFont
from base.base_damsm_trainer import BaseTrainer


# For visualization ################################################
COLOR_DIC = {0:[128,64,128],  1:[244, 35,232],
             2:[70, 70, 70],  3:[102,102,156],
             4:[190,153,153], 5:[153,153,153],
             6:[250,170, 30], 7:[220, 220, 0],
             8:[107,142, 35], 9:[152,251,152],
             10:[70,130,180], 11:[220,20, 60],
             12:[255, 0, 0],  13:[0, 0, 142],
             14:[119,11, 32], 15:[0, 60,100],
             16:[0, 80, 100], 17:[0, 0, 230],
             18:[0,  0, 70],  19:[0, 0,  0]}
FONT_MAX = 50


class Trainer(BaseTrainer):
    """
    Example Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, rnn_encoder, cnn_encoder, loss, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(rnn_encoder, cnn_encoder, loss, optimizer, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

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
        self.rnn_encoder.train()
        self.cnn_encoder.train()

        total_loss = 0
        word_total_loss_0 = 0
        word_total_loss_1 = 0
        sent_total_loss_0 = 0
        sent_total_loss_1 = 0

        for batch_idx, data in enumerate(self.data_loader):
            batch_images = data['right_images_256'].to(self.device)
            batch_size = batch_images.size(0)
            batch_captions = data['right_captions'].to(self.device)
            batch_caption_lengths = data['right_caption_lengths'].to(self.device)
            labels = torch.LongTensor(range(batch_size)).to(self.device)
            class_ids = data['class_id']

            self.optimizer.zero_grad()
            image_features, image_emb = self.cnn_encoder(batch_images)
            att_size = image_features.size(2)
            states = self.rnn_encoder.init_hidden(batch_size)
            words_emb, sent_emb = self.rnn_encoder(batch_captions, batch_caption_lengths, states)


            word_loss_0, word_loss_1, attntion_maps = self.loss["word"](
                img_features=image_features,
                words_emb=words_emb,
                labels=labels,
                cap_lens=batch_caption_lengths,
                class_ids=class_ids,
                batch_size=batch_size)

            sent_loss_0, sent_loss_1 = self.loss["sent"](
                cnn_code=image_emb,
                rnn_code=sent_emb,
                labels=labels,
                class_ids=class_ids,
                batch_size=batch_size)

            loss = word_loss_0 + word_loss_1 + sent_loss_0 + sent_loss_1
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.rnn_encoder.parameters(), self.config["trainer"]["clip_grad"])
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            self.writer.add_scalar('word_loss_0', word_loss_0.item())
            self.writer.add_scalar('word_loss_1', word_loss_1.item())
            self.writer.add_scalar('sent_loss_0', sent_loss_0.item())
            self.writer.add_scalar('sent_loss_0', sent_loss_1.item())
            word_total_loss_0 += word_loss_0.item()
            word_total_loss_1 += word_loss_1.item()
            sent_total_loss_0 += sent_loss_0.item()
            sent_total_loss_1 += sent_loss_1.item()
            total_loss += loss.item()

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] [Total Loss: {:.6f}] [Sent. Loss: {:.6f}|{:.6f}] [Word Loss: {:.6f}|{:.6f}]'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item(),
                    sent_loss_0.item(),
                    sent_loss_1.item(),
                    word_loss_0.item(),
                    word_loss_1.item()))
                # self.writer.add_image('input', make_grid(batch_images.cpu(), nrow=8, normalize=True))

            # todo remove
            break

        log = {
            'loss': total_loss / len(self.data_loader),
            'word_loss_0': word_total_loss_0 / len(self.data_loader),
            'word_loss_1': word_total_loss_1 / len(self.data_loader),
            'sent_loss_0': sent_total_loss_0 / len(self.data_loader),
            'sent_loss_1': sent_total_loss_1 / len(self.data_loader)
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}
            # attention Maps
            img_set, _ = \
                self.build_super_images(batch_images.cpu(), batch_captions.cpu(),
                                   self.data_loader.dataset.vocab.idx2word, attntion_maps, att_size)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps%d.png' % (self.results_dir, epoch)
                im.save(fullpath)

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
        self.rnn_encoder.eval()
        self.cnn_encoder.eval()
        total_val_loss = 0
        word_total_val_loss_0 = 0
        word_total_val_loss_1 = 0
        sent_total_val_loss_0 = 0
        sent_total_val_loss_1 = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                batch_images = data['right_images_256'].to(self.device)
                batch_size = batch_images.size(0)
                batch_captions = data['right_captions'].to(self.device)
                batch_caption_lengths = data['right_caption_lengths'].to(self.device)
                labels = torch.LongTensor(range(batch_size)).to(self.device)
                class_ids = data['class_id']

                image_features, image_emb = self.cnn_encoder(batch_images)
                states = self.rnn_encoder.init_hidden(batch_size)
                words_emb, sent_emb = self.rnn_encoder(batch_captions, batch_caption_lengths, states)

                word_loss_0, word_loss_1, attntion_maps = self.loss["word"](
                    img_features=image_features,
                    words_emb=words_emb,
                    labels=labels,
                    cap_lens=batch_caption_lengths,
                    class_ids=class_ids,
                    batch_size=batch_size)

                sent_loss_0, sent_loss_1 = self.loss["sent"](
                    cnn_code=image_emb,
                    rnn_code=sent_emb,
                    labels=labels,
                    class_ids=class_ids,
                    batch_size=batch_size)

                loss = word_loss_0 + word_loss_1 + sent_loss_0 + sent_loss_1

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                self.writer.add_scalar('word_loss_0', word_loss_0.item())
                self.writer.add_scalar('word_loss_1', word_loss_1.item())
                self.writer.add_scalar('sent_loss_0', sent_loss_0.item())
                self.writer.add_scalar('sent_loss_0', sent_loss_1.item())
                word_total_val_loss_0 += word_loss_0.item()
                word_total_val_loss_1 += word_loss_1.item()
                sent_total_val_loss_0 += sent_loss_0.item()
                sent_total_val_loss_1 += sent_loss_1.item()
                total_val_loss += loss.item()

                #todo remove
                break

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'word_val_loss_0': word_total_val_loss_0 / len(self.data_loader),
            'word_val_loss_1': word_total_val_loss_1 / len(self.data_loader),
            'sent_val_loss_0': sent_total_val_loss_0 / len(self.data_loader),
            'sent_val_loss_1': sent_total_val_loss_1 / len(self.data_loader)
        }

    def build_super_images(self,
                           real_imgs,
                           captions,
                           ixtoword,
                           attn_maps,
                           att_sze,
                           lr_imgs=None,
                           batch_size=16,
                           max_word_num=20):
        nvis = 8
        real_imgs = real_imgs[:nvis]

        if lr_imgs is not None:
            lr_imgs = lr_imgs[:nvis]

        if att_sze == 17:
            vis_size = att_sze * 16
        else:
            vis_size = real_imgs.size(2)

        text_convas = \
            np.ones([batch_size * 50,
                     (max_word_num + 2) * (vis_size + 2), 3],
                    dtype=np.uint8)

        for i in range(max_word_num):
            istart = (i + 2) * (vis_size + 2)
            iend = (i + 3) * (vis_size + 2)
            text_convas[:, istart:iend, :] = COLOR_DIC[i]


        real_imgs = \
            torch.nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
        # [-1, 1] --> [0, 1]
        real_imgs.add_(1).div_(2).mul_(255)
        real_imgs = real_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
        pad_sze = real_imgs.shape
        middle_pad = np.zeros([pad_sze[2], 2, 3])
        post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
        if lr_imgs is not None:
            lr_imgs = \
                torch.nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
            # [-1, 1] --> [0, 1]
            lr_imgs.add_(1).div_(2).mul_(255)
            lr_imgs = lr_imgs.data.numpy()
            # b x c x h x w --> b x h x w x c
            lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

        # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
        seq_len = max_word_num
        img_set = []
        num = nvis  # len(attn_maps)

        text_map, sentences = \
            self.drawCaption(text_convas, captions, ixtoword, vis_size)
        text_map = np.asarray(text_map).astype(np.uint8)

        bUpdate = 1
        for i in range(num):
            attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
            # --> 1 x 1 x 17 x 17
            attn_max = attn.max(dim=1, keepdim=True)
            attn = torch.cat([attn_max[0], attn], 1)
            #
            attn = attn.view(-1, 1, att_sze, att_sze)
            attn = attn.repeat(1, 3, 1, 1).data.numpy()
            # n x c x h x w --> n x h x w x c
            attn = np.transpose(attn, (0, 2, 3, 1))
            num_attn = attn.shape[0]
            #
            img = real_imgs[i]
            if lr_imgs is None:
                lrI = img
            else:
                lrI = lr_imgs[i]
            row = [lrI, middle_pad]
            row_merge = [img, middle_pad]
            row_beforeNorm = []
            minVglobal, maxVglobal = 1, 0
            for j in range(num_attn):
                one_map = attn[j]
                if (vis_size // att_sze) > 1:
                    one_map = \
                        skimage.transform.pyramid_expand(one_map, sigma=20,
                                                         upscale=vis_size // att_sze)
                row_beforeNorm.append(one_map)
                minV = one_map.min()
                maxV = one_map.max()
                if minVglobal > minV:
                    minVglobal = minV
                if maxVglobal < maxV:
                    maxVglobal = maxV
            for j in range(seq_len + 1):
                if j < num_attn:
                    one_map = row_beforeNorm[j]
                    one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                    one_map *= 255
                    #
                    PIL_im = Image.fromarray(np.uint8(img))
                    PIL_att = Image.fromarray(np.uint8(one_map))
                    merged = \
                        Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                    mask = Image.new('L', (vis_size, vis_size), (210))
                    merged.paste(PIL_im, (0, 0))
                    merged.paste(PIL_att, (0, 0), mask)
                    merged = np.array(merged)[:, :, :3]
                else:
                    one_map = post_pad
                    merged = post_pad
                row.append(one_map)
                row.append(middle_pad)
                #
                row_merge.append(merged)
                row_merge.append(middle_pad)
            row = np.concatenate(row, 1)
            row_merge = np.concatenate(row_merge, 1)
            txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
            if txt.shape[1] != row.shape[1]:
                print('txt', txt.shape, 'row', row.shape)
                bUpdate = 0
                break
            row = np.concatenate([txt, row, row_merge], 0)
            img_set.append(row)
        if bUpdate:
            img_set = np.concatenate(img_set, 0)
            img_set = img_set.astype(np.uint8)
            return img_set, sentences
        else:
            return None

    def drawCaption(self, convas, captions, ixtoword, vis_size, off1=2, off2=2):
        num = captions.size(0)
        img_txt = Image.fromarray(convas)
        # get a font
        fnt = None  # ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
        # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
        # get a drawing context
        d = ImageDraw.Draw(img_txt)
        sentence_list = []
        for i in range(num):
            cap = captions[i].data.cpu().numpy()
            sentence = []
            for j in range(len(cap)):
                if cap[j] == 0:
                    break
                word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
                d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                       font=fnt, fill=(255, 255, 255, 255))
                sentence.append(word)
            sentence_list.append(sentence)
        return img_txt, sentence_list
