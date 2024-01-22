#-*-coding:utf-8-*- 
import os
import socket
import logging
from datetime import datetime
import math
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
import operator
from evaluator_entire_inter import Evaluator
from utils.checkpoint import Checkpoint
import numpy as np
import torch.nn.functional as F


def reduce_tensor(tensor, world_size):
    if isinstance(tensor, int):
        return tensor

    rt = tensor.data.clone()

    if world_size != 1:
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= world_size
    return rt


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor,dst=0)
        if rank == 0:
            tensor /= world_size


def queue_data(data, k):
    return torch.cat([data, k], dim=0)

def dequeue_data(data, K=996):
    if len(data) > K:
        return data[-K:]
    else:
        return data

def cosine_similarity(k, K):
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(k, K):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return 0
        else:
            return round(dot_product / ((normA ** 0.5) * (normB ** 0.5)) * 100, 2)


class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        criterion: criterion function for training
        optimizer (seq2seq_model.optim.Optimizer): optimizer for training
        expt_dir (optional, str): experiment Directory to store details of the experiment,
                by default it makes a folder in the current directory to store the details (default: `experiment`).
    """

    def __init__(self, criterion, ch_greedy_decoder, optimizer, expt_dir='experiment', print_every=10,
                 evaluate_every=100, writer_path='/home/disk3/zy_data/runs/', train_sampler=None, rank=0, world_size=2,
                 logger = None,batch_size=None):
        self._trainer = "Simple Trainer"
        # self.logger = logging.getLogger(__name__)
        self.logger = logger
        self.criterion = criterion
        self.ch_greedy_decoder = ch_greedy_decoder

        self.evaluator = Evaluator(criterion=self.criterion, ch_greedy_decoder=ch_greedy_decoder, world_size=world_size, rank=rank)
        self.optimizer = optimizer
        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))
        self.print_every = print_every
        self.evaluate_every = evaluate_every
        self.min_er = 9999999999
        self.min_cost = 9999999999
        self.train_sampler = train_sampler
        self.rank = rank
        self.batch_size = batch_size
        self.world_size = world_size
        # self.queue_q = torch.zeros((0, 512), dtype=torch.float32)
        #
        # self.queue_k = torch.zeros((0,512), dtype=torch.float)
        self.crit = torch.nn.SmoothL1Loss()

        # self.frame_attention = FrameAttention()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        writer_path_final=os.path.join(writer_path,'runs')
        if self.rank == 0:
            self.writer = SummaryWriter(logdir=os.path.join(writer_path_final, current_time + '_' + socket.gethostname()))

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir


    def loss_function(self, q, k, queue_q, queue_k, temp,bs):
        q1 = q.squeeze(1)
        k1 = k.squeeze(1)
        queue_k1 = queue_k.squeeze(1)
        queue_q1 = queue_q.squeeze(1)

        l_pos_q = torch.bmm(q, k.transpose(2, 1)).squeeze(2)
        l_neg_q = torch.mm(q1, queue_k1.transpose(1, 0))

        l_pos_k = torch.bmm(k, q.transpose(2, 1)).squeeze(2)
        l_neg_k = torch.mm(k1, queue_q1.transpose(1, 0))

        logits_q = torch.cat((l_pos_q, l_neg_q), dim=1) / temp
        logits_k = torch.cat((l_pos_k, l_neg_k), dim=1) / temp

        #### 计算队列的输出
        queue_q=queue_q.cuda().data.cpu().numpy()
        q = q.cuda().data.cpu().numpy()
        q_out=[]

        q_out =torch.zeros(bs,1,512)
        for j in range(len(q)):
            s_j=[]
            index=0
            max=0
            for i in range(len(queue_q)):
                # a = len(queue_q)
                # b = queue_q[i]
                # c = len(q)
                # d = q[j].squeeze()

                num = np.dot(queue_q[i].squeeze(), q[j].squeeze())
                denom = np.linalg.norm(queue_q[i].squeeze()) * np.linalg.norm(q[j].squeeze())
                s_j_i = num / denom
                s_j.append(s_j_i)
                if s_j_i > max:
                    max=s_j_i
                    index=i
            # print(type(queue_q))
            # print(queue_q.size())
            if len(queue_q) == 0:
                q_new=q[j]
                # q_new = q_new.unsqueeze(0)
                q_new=torch.Tensor(q_new)
                bb=type(q_new)
                q_out[j, :, :] = q_new
            else:
                q_inedx=queue_q[index,:,:]
                q_new=q[j]+q_inedx
                # q_new = q_new.unsqueeze(0)
                q_new = torch.Tensor(q_new)
                cc = type(q_new)
                q_out[j,:,:]=q_new
            # print(q_new)
            aa=q_out
            # q_out[]
            # q_out.append(q_new)

        # a=q_out



            # bb=s_j.index(max(s_j))
            # max_index, max_number = max(enumerate(encoder_rnn_input), key=operator.itemgetter(1))
        # index, value = max(s_j, key=lambda item: item[1])
        # aa,bb=max(s_j, key=lambda item: item[1])

        return logits_q, logits_k,q_out

    def train(self, model, data_loader, dev_data_loader, n_epochs, start_epoch, min_er=999999999, min_cost=999999999, method=None,batch_size=None):

        log = self.logger
        self.min_er = min_er
        self.min_cost = min_cost
        self.batch_size=batch_size
        steps_per_epoch = len(data_loader)
        steps_in_epoch = int(np.floor(len(data_loader) / self.batch_size))

        log.info("from epoch {} to epoch {}".format(start_epoch, n_epochs))

        print_loss_total = 0  # Reset every print_every
        print_ch_loss_total = 0
        temp=7


        ####新增加的
        total = 0
        total_bleu1, total_bleu2, total_bleu3, total_bleu4 = 0, 0, 0, 0
        total_er, total_bleu, total_acc_sentence, total_acc_word, total_meteor, total_rouge, total_cider \
            = 0, 0, 0, 0, 0, 0, 0

        total_bleu1_step=0
        total_bleu2_step=0
        total_bleu3_step = 0
        total_bleu4_step = 0
        total_meteor_step = 0
        total_rouge_step = 0
        total_cider_step = 0
        total_er_step = 0
        total_acc_word_step = 0
        total_acc_sentence_step = 0



        ####新增加的

        #####???????????????

        for epoch in range(start_epoch, n_epochs):
            # data_loader.dataset.clean_name_video_dict()

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            batch_iterator = data_loader.__iter__()

            # queue_q=torch.zeros((0, 1, 512), dtype=torch.float).cuda()
            # queue_k = torch.zeros((0, 1, 512), dtype=torch.float).cuda()
            for step, batch in enumerate(batch_iterator):

                global_step = epoch * steps_per_epoch + step

                model.train(True)


                video_batch, targets_batch, questions_batch, history_video_batch, \
                history_question_batch, dialogue_batch, video_lengths_batch, targets_length_batch, \
                question_lengths_batch, history_video_lengths_batch, history_question_lengths_batch = batch

                if torch.cuda.is_available():
                    video_batch = video_batch.cuda()
                    history_video_batch = history_video_batch.cuda()
                    targets_batch = targets_batch.cuda(non_blocking=True)
                    questions_batch = questions_batch.cuda(non_blocking=True)
                    history_question_batch = history_question_batch.cuda(non_blocking=True)
                    video_lengths_batch = video_lengths_batch.cuda(non_blocking=True)
                    targets_length_batch = targets_length_batch.cuda(non_blocking=True)
                    question_lengths_batch = question_lengths_batch.cuda(non_blocking=True)
                    history_question_lengths_batch = history_question_lengths_batch.cuda(non_blocking=True)
                    history_video_lengths_batch = history_video_lengths_batch.cuda(non_blocking=True)
                    dialogue_batch = dialogue_batch.cuda(non_blocking=True)



                model.zero_grad()
                start_time = datetime.now()

                # teacher_forcing_ratio = 0.2 * (1 - float(epoch) / n_epochs)
                teacher_forcing_ratio = 0.5

                decoder_outputs, decoder_hidden, other, target_variable,decoder_hidden, q_mix, q_mix_hat = model(video_batch, targets_batch, questions_batch,
                                                           history_video_batch,history_question_batch, dialogue_batch,
                                                           video_lengths_batch, targets_length_batch,question_lengths_batch,
                                                           history_video_lengths_batch, history_question_lengths_batch,
                                                           teacher_forcing_ratio = 0, method = None, #'beam-search',
                                                           mode="train",
                                                           use_dialogue_info=True,
                                                           use_question_model=True)
                loss_ = self.criterion
                loss_.reset()

                loss_.eval_batch(encoder_outputs=other['video_encoder_outputs'],
                                 decoder_outputs=decoder_outputs,
                                 targets=target_variable,
                                 targets_length=targets_length_batch,
                                 decoder_hidden=decoder_hidden,
                                 q_mix=q_mix,
                                 q_mix_hat=q_mix_hat)
                seqlist = other['sequence']
                seqlist_tensor = torch.ones(target_variable.size(1),target_variable.size(0))
                for i in range(len(seqlist)):
                    seqlist_tensor[i] = seqlist[i].view(-1)
                if torch.cuda.is_available():
                    seqlist_tensor = seqlist_tensor.cuda()
                seqlist_tensor = seqlist_tensor.transpose(0, 1).long()  # batch_size * seqlen
                decoded_output = self.ch_greedy_decoder.seq2seq_decode(seqlist_tensor)
                split_targets = []
                for i, size in enumerate(targets_length_batch):
                    split_targets.append(targets_batch[i, 1:size - 1])
                target_strings = self.ch_greedy_decoder.convert_to_strings(split_targets)
                metrics = self.ch_greedy_decoder.metric_batch(decoded_output, target_strings, targets_length_batch.size(0))
                total_bleu1 += metrics['Bleu_1']
                total_bleu2 += metrics['Bleu_2']
                total_bleu3 += metrics['Bleu_3']
                total_bleu4 += metrics['Bleu_4']
                total_meteor += metrics['METEOR']
                total_rouge += metrics['ROUGE_L']
                total_cider += metrics['CIDEr']
                total_er += metrics['er']
                total_acc_word += metrics['acc_word']
                total_acc_sentence += metrics['acc_sentence']
                ######新增加的

                total_bleu1_step+= metrics['Bleu_1']
                total_bleu2_step+= metrics['Bleu_2']
                total_bleu3_step+= metrics['Bleu_3']
                total_bleu4_step+= metrics['Bleu_4']
                total_meteor_step+= metrics['METEOR']
                total_rouge_step+= metrics['ROUGE_L']
                total_cider_step+= metrics['CIDEr']
                total_er_step+= metrics['er']
                total_acc_word_step+= metrics['acc_word']
                total_acc_sentence_step+=metrics['acc_sentence']






            # Backward propagation
                model.zero_grad()
                loss_.backward()
                self.optimizer.step()

                loss = loss_.get_loss()

                end_time = datetime.now()

                # Record average criterion
                # print_loss_total += reduce_tensor(loss, self.world_size)
                print_loss_total += loss.item()

                if (global_step + 1) % self.print_every == 0 and (global_step + 1) >= self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    # print_ch_loss_avg = print_ch_loss_total / self.print_every
                    print_loss_bleu1=total_bleu1/self.print_every
                    print_loss_bleu2 = total_bleu2 / self.print_every
                    print_loss_bleu3 = total_bleu3 / self.print_every
                    print_loss_bleu4 = total_bleu4 / self.print_every
                    print_loss_meteor = total_meteor / self.print_every
                    print_loss_rouge = total_rouge / self.print_every
                    print_loss_cider = total_cider / self.print_every
                    print_loss_er = total_er / self.print_every
                    print_loss_acc_word = total_acc_word / self.print_every
                    print_loss_acc_sentence = total_acc_sentence / self.print_every
                    print_loss_total = 0
                    total_bleu1=0
                    total_bleu2=0
                    total_bleu3=0
                    total_bleu4=0
                    total_meteor=0
                    total_rouge=0
                    total_cider=0
                    total_er=0
                    total_acc_word=0
                    total_acc_sentence=0
                    # print_ch_loss_total = 0
                    if self.rank == 0:
                        # log_msg = 'Progress: %d: %d / %d, Train: total: %.4f, time %s' % (
                        #     epoch, step, steps_per_epoch, print_loss_avg,
                        #     end_time - start_time)
                        # lr = self.optimizer.optimizer.param_groups[0]['lr']
                        # log_msg += ", lr: {}".format(lr)
                        # log.info(log_msg)
                        log_msg = 'Progress: %d: %d / %d, Train: total: %.4f, bleu1:%.4f,bleu2:%.4f,blue3:%.4f,blue4:%.4f,meteor: %.4f,rouge:%.4f,cider:%.4f,wer:%.4f,acc_word:%.4f,acc_sentence:%.4f,time %s' % (
                            epoch, step, steps_per_epoch, print_loss_avg, print_loss_bleu1, print_loss_bleu2,
                            print_loss_bleu3, print_loss_bleu4,
                            print_loss_meteor, print_loss_rouge, print_loss_cider, print_loss_er, print_loss_acc_word,
                            print_loss_acc_sentence,
                            end_time - start_time)
                        lr = self.optimizer.optimizer.param_groups[0]['lr']
                        log_msg += ", lr: {}".format(lr)
                        log.info(log_msg)
                        #########可以显示的代码
                        # self.writer.add_scalar("data/train_loss",
                        #                        print_loss_avg, step)





            # train_loss_total = print_loss_total/steps_per_epoch
            train_bleu1 =total_bleu1_step/steps_per_epoch
            train_bleu2 =total_bleu2_step/steps_per_epoch
            train_bleu3 =total_bleu3_step/steps_per_epoch
            train_bleu4 =total_bleu4_step/steps_per_epoch
            train_meteor =total_meteor_step/steps_per_epoch
            train_rouge =total_rouge_step/steps_per_epoch
            train_cider =total_cider_step/steps_per_epoch
            train_er =total_er_step/steps_per_epoch
            train_acc_word =total_acc_word_step/steps_per_epoch
            train_acc_sentence =total_acc_sentence_step/steps_per_epoch

            total_bleu1_step=0
            total_bleu2_step=0
            total_bleu3_step=0
            total_bleu4_step=0
            total_meteor_step=0
            total_rouge_step=0
            total_cider_step=0
            total_er_step=0
            total_acc_word_step=0
            total_acc_sentence_step=0

            log_msg = 'Epoch: %d, bleu1:%.4f,bleu2:%.4f,blue3:%.4f,blue4:%.4f,meteor: %.4f,rouge:%.4f,cider:%.4f,wer:%.4f,acc_word:%.4f,acc_sentence:%.4f' % (
                            epoch, train_bleu1, train_bleu2,
                            train_bleu3, train_bleu4,
                            train_meteor, train_rouge, train_cider, train_er, train_acc_word,
                            train_acc_sentence)

            log.info(log_msg)

            model.eval()

            # dev_loss, dev_ch_loss, average_er, average_bleu = self.evaluator.evaluate(model, dev_data_loader,
            #                                                                           loss_weight_ratio=loss_weight_ratio)
            dev_loss, average_er, average_bleu1, average_bleu2, average_bleu3, average_bleu4,\
            average_acc_word, average_acc_sentence\
                , average_meteor, average_rouge, average_cider\
                = self.evaluator.evaluate(model, dev_data_loader, method=method,batch_size=self.batch_size)

            log_msg = "evaluate in epoch %d: %d / %d, Dev: total: %.4f, " \
                      "Average_er: %.4f, Average_bleu1: %.4f, Average_bleu2: %.4f, Average_bleu3: %.4f, Average_bleu4: %.4f " \
                      "Average_acc_word: %.4f, Average_acc_sentence: %.4f, " \
                      "Average_meteor: %.4f, Average_rouge: %.4f, " \
                      "Average_cider: %.4f, min_er: %.4f " \
                      % (epoch, step, steps_per_epoch, dev_loss,
                         average_er, average_bleu1, average_bleu2, average_bleu3, average_bleu4,  average_acc_word, average_acc_sentence,
                         average_meteor, average_rouge, average_cider, self.min_er)
            #######可以显示的代码


            model.train(mode=True)


            if dev_loss < self.min_cost:
                self.min_cost = dev_loss

            if average_er < self.min_er:
                self.min_er = average_er
                if self.rank == 0:
                    Checkpoint(model_state=model.state_dict(),
                               optimizer_state=self.optimizer.optimizer.state_dict(),
                               epoch=epoch,
                               min_er=self.min_er,
                               min_cost=self.min_cost
                               ).save(os.path.join(self.expt_dir, 'min_er'))
                log_msg += ", checkpoint saved min_er"
            log.info(log_msg)


            #### 保存模型
            model_path = os.path.join(self.expt_dir, 'model_checkpoint')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if epoch % 5 == 0:
                save_file_path = os.path.join(model_path,
                                              'save_{}.pth'.format(epoch))
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': self.optimizer.optimizer.state_dict(),
                    'min_er': self.min_er,
                    'min_cost': self.min_cost,
                }
                torch.save(states, save_file_path)




