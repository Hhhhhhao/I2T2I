import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from model.global_attention_modules import func_attention
n_gpu = torch.cuda.device_count()


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)


# ################## Loss for matching text-image -- DAMSM ###################

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class SentLoss(torch.nn.Module):
    def __init__(self, opt, eps=1e-8):
        super(SentLoss, self).__init__()
        self.gamma3 = opt.gamma3
        self.eps = eps
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, cnn_code, rnn_code, labels, class_ids, batch_size):
        # ### Mask mis-match samples  ###
        # that come from the same class as the real sample ###
        masks = []
        if class_ids is not None:
            for i in range(batch_size):
                mask = (class_ids == class_ids[i]).astype(np.uint8)
                mask[i] = 0
                masks.append(mask.reshape((1, -1)))
            masks = np.concatenate(masks, 0)
            # masks: batch_size x batch_size
            masks = torch.ByteTensor(masks)
            if torch.cuda.is_available():
                masks = masks.cuda()

        # --> seq_len x batch_size x nef
        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)

        # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
        cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
        rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
        # scores* / norm*: seq_len x batch_size x batch_size
        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=self.eps) * self.gamma3

        # --> batch_size x batch_size
        scores0 = scores0.squeeze()
        if class_ids is not None:
            scores0.data.masked_fill_(masks, -float('inf'))
        scores1 = scores0.transpose(0, 1)
        if labels is not None:
            loss0 = self.loss(scores0, labels)
            loss1 = self.loss(scores1, labels)
        else:
            loss0, loss1 = None, None
        return loss0, loss1


class WordLoss(torch.nn.Module):
    def __init__(self, opt, eps=1e-8):
        super(WordLoss, self).__init__()
        self.gamma1 = opt.gamma1
        self.gamma2 = opt.gamma2
        self.gamma3 = opt.gamma3
        self.eps = eps
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, img_features, words_emb, labels, cap_lens, class_ids, batch_size):
        """
            words_emb(query): batch x nef x seq_len
            img_features(context): batch x nef x 17 x 17
        """
        masks = []
        att_maps = []
        similarities = []
        cap_lens = cap_lens.cpu().tolist()
        for i in range(batch_size):
            if class_ids is not None:
                mask = (class_ids == class_ids[i]).astype(np.uint8)
                mask[i] = 0
                masks.append(mask.reshape((1, -1)))
            # Get the i-th text description
            words_num = cap_lens[i]
            # -> 1 x nef x words_num
            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
            # -> batch_size x nef x words_num
            word = word.repeat(batch_size, 1, 1)
            # batch x nef x 17*17
            context = img_features
            """
                word(query): batch x nef x words_num
                context: batch x nef x 17 x 17
                weiContext: batch x nef x words_num
                attn: batch x words_num x 17 x 17
            """
            weiContext, attn = func_attention(word, context, self.gamma1)
            att_maps.append(attn[i].unsqueeze(0).contiguous())
            # --> batch_size x words_num x nef
            word = word.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()
            # --> batch_size*words_num x nef
            word = word.view(batch_size * words_num, -1)
            weiContext = weiContext.view(batch_size * words_num, -1)
            #
            # -->batch_size*words_num
            row_sim = cosine_similarity(word, weiContext)
            # --> batch_size x words_num
            row_sim = row_sim.view(batch_size, words_num)

            # Eq. (10)
            row_sim.mul_(self.gamma2).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)

            # --> 1 x batch_size
            # similarities(i, j): the similarity between the i-th image and the j-th text description
            similarities.append(row_sim)

        # batch_size x batch_size
        similarities = torch.cat(similarities, 1)
        if class_ids is not None:
            masks = np.concatenate(masks, 0)
            # masks: batch_size x batch_size
            masks = torch.ByteTensor(masks)
            if torch.cuda.is_available():
                masks = masks.cuda()

        similarities = similarities * self.gamma3
        if class_ids is not None:
            similarities.data.masked_fill_(masks, -float('inf'))
        similarities1 = similarities.transpose(0, 1)
        if labels is not None:
            loss0 = self.loss(similarities, labels)
            loss1 = self.loss(similarities1, labels)
        else:
            loss0, loss1 = None, None
        return loss0, loss1, att_maps



# ################## Loss for AttnGAN G and Ds ##############################
class AttnDiscriminatorLoss(torch.nn.Module):
    def __init__(self):
        super(AttnDiscriminatorLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, netD, real_imgs, fake_imgs, conditions, real_labels, fake_labels):
        # Forward
        real_features = netD(real_imgs)
        fake_features = netD(fake_imgs.detach())

        if n_gpu > 1:
            # loss
            #
            cond_real_logits = netD.module.COND_DNET(real_features, conditions)
            cond_real_errD = self.loss(cond_real_logits, real_labels)
            cond_fake_logits = netD.module.COND_DNET(fake_features, conditions)
            cond_fake_errD = self.loss(cond_fake_logits, fake_labels)
            #
            batch_size = real_features.size(0)
            cond_wrong_logits = netD.module.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
            cond_wrong_errD = self.loss(cond_wrong_logits, fake_labels[1:batch_size])

            if netD.module.UNCOND_DNET is not None:
                real_logits = netD.module.UNCOND_DNET(real_features)
                fake_logits = netD.module.UNCOND_DNET(fake_features)
                real_errD = self.loss(real_logits, real_labels)
                fake_errD = self.loss(fake_logits, fake_labels)
                errD = ((real_errD + cond_real_errD) / 2. +
                        (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
            else:
                errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
        else:
            # loss
            #
            cond_real_logits = netD.COND_DNET(real_features, conditions)
            cond_real_errD = self.loss(cond_real_logits, real_labels)
            cond_fake_logits = netD.COND_DNET(fake_features, conditions)
            cond_fake_errD = self.loss(cond_fake_logits, fake_labels)
            #
            batch_size = real_features.size(0)
            cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
            cond_wrong_errD = self.loss(cond_wrong_logits, fake_labels[1:batch_size])

            if netD.UNCOND_DNET is not None:
                real_logits = netD.UNCOND_DNET(real_features)
                fake_logits = netD.UNCOND_DNET(fake_features)
                real_errD = self.loss(real_logits, real_labels)
                fake_errD = self.loss(fake_logits, fake_labels)
                errD = ((real_errD + cond_real_errD) / 2. +
                        (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
            else:
                errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
        return errD

class AttnGeneratorLoss(torch.nn.Module):
    def __init__(self, opt):
        super(AttnGeneratorLoss, self).__init__()
        self.opt = opt
        self.g_lambda = opt.g_lambda
        self.mse_loss = torch.nn.MSELoss()
        self.sent_loss = SentLoss(opt)
        self.word_loss = WordLoss(opt)

    def forward(self, netsD, image_encoder, fake_imgs, real_labels, words_embs, sent_emb, match_labels, cap_lens, class_ids):
        numDs = len(netsD)
        batch_size = real_labels.size(0)
        # Forward
        errG_total = 0
        if n_gpu > 1:
            for i in range(numDs):
                features = netsD[i](fake_imgs[i])
                cond_logits = netsD[i].module.COND_DNET(features, sent_emb)
                cond_errG = self.mse_loss(cond_logits, real_labels)
                if netsD[i].module.UNCOND_DNET is not None:
                    logits = netsD[i].module.UNCOND_DNET(features)
                    errG = self.mse_loss(logits, real_labels)
                    g_loss = errG + cond_errG
                else:
                    g_loss = cond_errG
                errG_total += g_loss

                # Ranking loss
                if i == (numDs - 1):
                    # words_features: batch_size x nef x 17 x 17
                    # sent_code: batch_size x nef
                    region_features, cnn_code = image_encoder(fake_imgs[i])
                    w_loss0, w_loss1, _ = self.word_loss(region_features, words_embs,
                                                     match_labels, cap_lens,
                                                     class_ids, batch_size)
                    w_loss = (w_loss0 + w_loss1) * \
                             self.g_lambda
                    # err_words = err_words + w_loss.data[0]

                    s_loss0, s_loss1 = self.sent_loss(cnn_code, sent_emb,
                                                 match_labels, class_ids, batch_size)
                    s_loss = (s_loss0 + s_loss1) * \
                             self.g_lambda
                    # err_sent = err_sent + s_loss.data[0]

                    errG_total += w_loss + s_loss
        else:
            for i in range(numDs):
                features = netsD[i](fake_imgs[i])
                cond_logits = netsD[i].COND_DNET(features, sent_emb)
                cond_errG = self.mse_loss(cond_logits, real_labels)
                if netsD[i].UNCOND_DNET is not None:
                    logits = netsD[i].UNCOND_DNET(features)
                    errG = self.mse_loss(logits, real_labels)
                    g_loss = errG + cond_errG
                else:
                    g_loss = cond_errG
                errG_total += g_loss

                # Ranking loss
                if i == (numDs - 1):
                    # words_features: batch_size x nef x 17 x 17
                    # sent_code: batch_size x nef
                    region_features, cnn_code = image_encoder(fake_imgs[i])
                    w_loss0, w_loss1, _ = self.word_loss(region_features, words_embs,
                                                     match_labels, cap_lens,
                                                     class_ids, batch_size)
                    w_loss = (w_loss0 + w_loss1) * \
                             self.g_lambda
                    # err_words = err_words + w_loss.data[0]

                    s_loss0, s_loss1 = self.sent_loss(cnn_code, sent_emb,
                                                 match_labels, class_ids, batch_size)
                    s_loss = (s_loss0 + s_loss1) * \
                             self.g_lambda
                    # err_sent = err_sent + s_loss.data[0]

                    errG_total += w_loss + s_loss
        return errG_total


################## Kl loss for Conditional Augmentation #########################

class KLLoss(torch.nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mu, logvar):
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.mean(KLD_element).mul_(-0.5)
        return KLD


################# Caption GAN loss ##############################################
class CaptGANGeneratorLoss(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super(CaptGANGeneratorLoss, self).__init__()
        self.eps = eps

    def forward(self, rewards, props):
        loss = rewards * torch.log(torch.clamp(props, min=self.eps, max=1.0))
        # TODO decide to take log or not
        # loss = rewards * torch.log(props)
        # loss = rewards * props
        loss = -torch.sum(loss)
        return loss


class CaptGANDiscriminatorLoss(torch.nn.Module):

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.loss = torch.nn.BCELoss()
        # self.loss = torch.nn.CrossEntropyLoss()
        n_gpu = torch.cuda.device_count()
        self.device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')

    def forward(self, evaluator_outputs, generator_outputs, other_outputs):
        batch_size = evaluator_outputs.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        real_labels = Variable(real_labels).to(self.device)
        fake_labels = Variable(fake_labels).to(self.device)

        true_loss = self.loss(evaluator_outputs, real_labels)
        fake_loss = self.loss(generator_outputs, fake_labels)
        other_loss = self.loss(other_outputs, fake_labels)
        loss = true_loss + self.alpha * fake_loss + self.beta * other_loss
        return loss