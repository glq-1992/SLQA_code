#-*-coding:utf-8-*- 
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class JointLoss():

    def __init__(self, ce_weight, ctc_weight, kl_weight, ctc_blank_label, cross_entroy_loss_label):
        self.alpha = ce_weight
        self.beta = ctc_weight
        self.ctc_blank_label = ctc_blank_label
        self.gama = kl_weight
        self.cross_entroy_loss_label = cross_entroy_loss_label
        self.ctc_loss = nn.CTCLoss(blank=ctc_blank_label, reduction='mean')
        self.crossEntroy_loss = nn.CrossEntropyLoss(ignore_index=self.cross_entroy_loss_label)
        self.loss_value = 0
        self.batch_nums = 0
        self.klloss = nn.L1Loss()

    def get_loss(self):
        return self.loss_value / self.batch_nums

    def reset(self):
        self.loss_value = 0
        self.batch_nums = 0

    def eval_batch(self, encoder_outputs, decoder_outputs, targets,
                   targets_length,decoder_hidden, q_mix=None, q_mix_hat=None):

        # ctc loss
        encoder_lengths = torch.tensor([encoder_outputs.shape[1]]*encoder_outputs.shape[0]).long().cuda(non_blocking=True)
        encoder_outputs = F.log_softmax(encoder_outputs.transpose(0, 1), dim=2)
        # ctc_target = targets[:, 1:]
        # index = (ctc_target == 2).nonzero()
        # tmp_target = []
        # batch_size = ctc_target.shape[0]
        # for i in range(batch_size):
        #     tmp_target.extend(ctc_target[i][0:index[i][1]])
        # ctc_target = torch.tensor(tmp_target).long().cuda(non_blocking=True)
        # ctc_lengths = targets_length - 2
        # ctc_loss = self.ctc_loss(encoder_outputs, ctc_target, encoder_lengths, ctc_lengths)
        ctc_targets = []
        for i , j in enumerate(targets_length):
            if self.alpha == 0 and self.beta == 1:
                ctc_targets.extend(targets[i, :j])
            else:
                ctc_targets.extend(targets[i, 1:j-1])
        ctc_targets = torch.Tensor(ctc_targets).long().cuda()
        if self.alpha == 0 and self.beta == 1:
            ctc_lengths = targets_length
        else:
            ctc_lengths = targets_length - 2

        ctc_loss = self.ctc_loss(encoder_outputs, ctc_targets, encoder_lengths, ctc_lengths)

        # ce loss
        decoder_outputs = torch.stack(decoder_outputs, dim=1)

        ce_loss = self.crossEntroy_loss(decoder_outputs.contiguous().view(decoder_outputs.shape[0]*decoder_outputs.shape[1], -1)
                                        , targets[:, 1:].contiguous().view(-1))



        ### MoCo Loss
        # N = logits_q.shape[0]
        # labels = torch.zeros(N, dtype=torch.long).cuda()
        # labels = torch.zeros(N, dtype=torch.long,requires_grad=False).cuda()
        # dist1=self.cross_loss(logits_q, labels)
        # dist1 = F.pairwise_distance(logits_q, labels, p=2)
        # dist1=torch.sum(dist1)/batch_size

        # dist2 = self.cross_loss(logits_k, labels)
        # dist2 = F.pairwise_distance(logits_k, labels, p=2)
        # dist2 = torch.sum(dist2) / batch_size



        # loss_moco = self.moco_loss(logits_q / self.temp, labels)
        # q1=q1.squeeze(1)
        # print(q1.shape)
        # print(k1.shape)
        # dist_sim = F.pairwise_distance(q1, k1, p=2)
        # dist_sim = torch.sum(dist_sim) / batch_size
        # loss_q=self.loss1(q1,k1)
        ######## 余弦距离
        # q = q.cuda().data.cpu().numpy()
        # k = k.cuda().data.cpu().numpy()
        # Q=Q.cuda().data.cpu().numpy()
        # K = K.cuda().data.cpu().numpy()
        # dist3_sum=0
        # # print(type(q))
        # for i in range(batch_size):
        #     q_d=q[i,:,:].squeeze()
        #     Q_d=Q[i,:,:].squeeze()
        #     dist3 = 1 - np.dot(q_d, Q_d) / (np.linalg.norm(q_d) * np.linalg.norm(Q_d))
        #     dist3_sum+=dist3
        #
        # dist4_sum=0
        # for i in range(batch_size):
        #     k_d = k[i, :, :].squeeze()
        #     K_d = K[i, :, :].squeeze()
        #     dist4 = 1 - np.dot(k_d, K_d) / (np.linalg.norm(k_d) * np.linalg.norm(K_d))
        #     dist4_sum += dist4




        # dist3 = 1 - np.dot(q, Q) / (np.linalg.norm(q) * np.linalg.norm(Q))
        # dist4 = 1 - np.dot(k, K) / (np.linalg.norm(k) * np.linalg.norm(K))

        #### 欧氏距离
        # self.T_v.cuda()
        # self.T_s.cuda()
        # encoder_rnn_input_s=torch.mean(encoder_rnn_input_s,dim=1)
        # embedded_s=torch.mean(embedded_s,dim=1)
        # v_e=self.T_v(encoder_rnn_input_s)
        # v_s=self.T_s(embedded_s)
        #
        # dist2 = F.pairwise_distance(v_e, v_s, p=2)
        # dist2=torch.sum(dist2)/batch_size

        ###### 问题和回答的相似性
        # dist_sim = F.pairwise_distance(decoder_hidden,q_hidden, p=2)
        #
        # dist_sim =torch.sum(dist_sim)/batch_size
        if isinstance(q_mix, torch.Tensor) and isinstance(q_mix_hat, torch.Tensor):
            # need to consider q_mix and q_mix_hat loss
            q_mix = q_mix.squeeze(1)
            q_mix_hat = q_mix_hat.squeeze(1)
            klloss = self.klloss(q_mix, q_mix_hat)
            self.loss_value += self.gama*klloss

        # total loss
        self.loss_value += self.alpha * ce_loss + self.beta * ctc_loss
        # self.loss_value += self.alpha * ce_loss
        self.batch_nums += 1

    def cuda(self):
        self.ctc_loss = self.ctc_loss.cuda()
        self.crossEntroy_loss = self.crossEntroy_loss.cuda()
        self.klloss = self.klloss.cuda()


    def backward(self):
        self.loss_value.backward( )


