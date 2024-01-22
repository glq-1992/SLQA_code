#-*-coding:utf-8-*- 
from __future__ import print_function, division
import torch
import torch.distributed as dist
import tqdm
from utils.util import displayAttention
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import torch.nn.functional as F
import operator

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

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        criterion: criterion function for evaluator
    """

    def __init__(self, criterion, ch_greedy_decoder, world_size, rank):
        self.criterion = criterion
        self.ch_greedy_decoder = ch_greedy_decoder
        self.world_size = world_size
        self.rank = rank

    def BLEU_process_data(self, transcript, reference):
        # Process the  reference
        candidate_ = []
        transcript = transcript.split()
        for i in range(len(transcript)):
            candidate_.append(transcript[i])

        # Process the  reference
        reference_ = []
        temp_reference = []
        reference = reference.split()
        for i in range(len(reference)):
            temp_reference.append(reference[i])
        reference_.append(temp_reference)

        return candidate_, reference_

    def loss_function(self, q, k, queue_q, queue_k, temp):
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

        return logits_q, logits_k

    def evaluate(self, model, data_loader, method=None,batch_size=None):
        """ Evaluate a model_state on given dataset and return performance.

        Args:
            model (seq2seq.models): model_state to evaluate
            data_loader (torch.util.DataLoader): data loader to evaluate against

        Returns:
            criterion(float): criterion of the given model_state on the given dataset
        """

        model.eval()
        loss = self.criterion
        loss.reset()
        total = 0
        total_er, total_bleu, total_acc_sentence, total_acc_word, total_meteor, total_rouge, total_cider \
            = 0, 0, 0, 0, 0, 0, 0
        total_bleu1, total_bleu2, total_bleu3, total_bleu4 = 0, 0, 0, 0
        eval_epochs = len(data_loader)
        self.batch_size = batch_size
        temp=7
        queue_q = torch.zeros((0, 1, 512), dtype=torch.float).cuda()
        queue_k = torch.zeros((0, 1, 512), dtype=torch.float).cuda()
        # steps_in_epoch_val = int(np.ceil(len(data_loader.dataset) / self.batch_size))
        # loss = 0
        # global_step = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader, desc="evaluate"):

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

                if method is None:

                    decoder_outputs, decoder_hidden, other, target_variable, decoder_hidden,q_mix, q_mix_hat= model(video_batch,    ######q_mix, q_mix_hat,根据有无Q决定要不要，有Q要，无Q不要
                                                                                                    targets_batch,
                                                                                                    questions_batch,
                                                                                                    history_video_batch,
                                                                                                    history_question_batch,
                                                                                                    dialogue_batch,
                                                                                                    video_lengths_batch,
                                                                                                    targets_length_batch,
                                                                                                    question_lengths_batch,
                                                                                                    history_video_lengths_batch,
                                                                                                    history_question_lengths_batch,
                                                                                                    teacher_forcing_ratio=0,
                                                                                                    method=None,
                                                                                                    mode="test",
                                                                                                    use_dialogue_info=True,
                                                                                                    use_question_model=True)

                elif method == 'beam-search':
                    decoded_batch, target_variable = model(video_batch,
                                                        targets_batch,
                                                        questions_batch,
                                                        history_video_batch,
                                                        history_question_batch,
                                                        dialogue_batch,
                                                        video_lengths_batch,
                                                        targets_length_batch,
                                                        question_lengths_batch,
                                                        history_video_lengths_batch,
                                                        history_question_lengths_batch,
                                                        teacher_forcing_ratio=0,
                                                        method=method)
                    # todo 需要重新写一个针对beam-search的decode方法
                    seqlist_tensor = decoded_batch
                    decoded_output = self.ch_greedy_decoder.seq2seq_decode(seqlist_tensor, method=method)
                    # decoded_batch [batch_size, lenseq]

                # embedded_s = embedded_s[:, 1:, :].cuda()
                if method is None:

                    loss.reset()
                    loss.eval_batch(encoder_outputs=other['video_encoder_outputs'],
                                     decoder_outputs=decoder_outputs,
                                     targets=target_variable,
                                     targets_length=targets_length_batch,
                                     decoder_hidden=decoder_hidden,
                                    q_mix=q_mix,
                                    q_mix_hat=q_mix_hat)

                    total += targets_length_batch.size(0)

                    # Translate the decoder output
                    seqlist = other['sequence']
                    seqlist_tensor = torch.ones(target_variable.size(1), target_variable.size(0))
                    for i in range(len(seqlist)):
                        seqlist_tensor[i] = seqlist[i].view(-1)
                    if torch.cuda.is_available():
                        seqlist_tensor = seqlist_tensor.cuda()
                    seqlist_tensor = seqlist_tensor.transpose(0, 1).long()  # batch_size * seqlen
                    decoded_output = self.ch_greedy_decoder.seq2seq_decode(seqlist_tensor)
                    # print(decoded_output)

                # unflatten targets
                split_targets = []
                for i, size in enumerate(targets_length_batch):
                    split_targets.append(targets_batch[i, 1:size - 1])
                target_strings = self.ch_greedy_decoder.convert_to_strings(split_targets)

                # display attention score map
                # displayAttention(target_strings, other['video_attention_score'], type='enc2dec')

                er, bleu, acc_sentence, acc_word, rouge, meteor, cider = 0, 0, 0, 0, 0 , 0, 0
                bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0

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

            #####新增加的
            '''metric batch'''

        dev_loss = loss.get_loss()
        # dev_loss = reduce_tensor(dev_loss, self.world_size)
        dev_loss = dev_loss.item()
        # dev_loss = reduce_loss(dev_loss, self.rank, self.world_size)
        # dev_loss = loss.item() / global_step


        '''metric batch'''
        average_er = total_er / eval_epochs
        average_acc_sentence = total_acc_sentence / eval_epochs
        average_acc_word = total_acc_word / eval_epochs
        average_cider = total_cider / eval_epochs
        average_meteor = total_meteor / eval_epochs
        average_rouge = total_rouge / eval_epochs
        average_bleu1 = total_bleu1 / eval_epochs
        average_bleu2 = total_bleu2 / eval_epochs
        average_bleu3 = total_bleu3 / eval_epochs
        average_bleu4 = total_bleu4 / eval_epochs

        return dev_loss, average_er, average_bleu1, average_bleu2, average_bleu3, average_bleu4,\
               average_acc_word, average_acc_sentence, average_meteor, average_rouge, average_cider


