#-*-coding:utf-8-*-?
import itertools
import numpy as np
from queue import PriorityQueue
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Question_Model_entire_inter import QuestionModel,QuestionEmbedding, Context_QuestionModel

import torchvision.models as models
from .resnet import resnet18
import math
from .attention_entire import Attention,Frame_Attention,Clip_Attention
from .GCM_new_gai import GCM



####����clip��question��word��������
num_clip=[]
num_question_word=[]
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # ȡ��model�ĺ�����
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        # self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.resnet_layer(x)
        # x = self.relu(x)
        return x



class Seq2SeqClassifier(nn.Module):
    # VGG-M parameter
    cfg = [96, "M", 256, "M", 512, 512, 512, "M"]

    def __init__(self, encoder_rnn_size, encoder_hidden_size, ch_decoder_rnn, vocab_size,
                 question_vocab_size, decoder_rnn_input_size, use_question_model=True, criterion=None,batch_size=None, theme_size=None,
                 dt_memory_size=None, dtqv_memory_size=None):
        super(Seq2SeqClassifier, self).__init__()

        #### 2d_resnet
        self.features = models.resnet18(pretrained=True)
        self.features = Net(self.features)

        # Question Modelling
        self.batch_size=batch_size
        self.use_question_model = use_question_model
        self.question_hidden_size = encoder_hidden_size * 2
        if use_question_model:
            self.question = QuestionModel(vocab_size=encoder_hidden_size,
                                          vocab_nums=question_vocab_size,
                                          hidden_size=encoder_hidden_size*2,
                                          p=0.2, use_attention=True,batch_size=self.batch_size)
            self.question_embedding = QuestionEmbedding(vocab_size=encoder_hidden_size,
                                          vocab_nums=question_vocab_size,
                                          hidden_size=encoder_hidden_size*2,
                                          p=0.2, use_attention=True,batch_size=self.batch_size)
            self.question_context = Context_QuestionModel(vocab_size=encoder_hidden_size,
                                                          vocab_nums=question_vocab_size,
                                                          hidden_size=encoder_hidden_size * 2,
                                                          p=0.2, use_attention=True, batch_size=self.batch_size)
        else:
            self.question = None

        # Encoder RNN
        self.encoder_rnn_size = encoder_rnn_size
        self.video_encoder_rnn = nn.GRU(input_size=encoder_rnn_size, hidden_size=encoder_hidden_size,
                                        num_layers=2, dropout=0.2, bidirectional=True, batch_first=True)

        # GCM
        self.GCM = GCM(vocab_size=question_vocab_size, embeded_size=decoder_rnn_input_size,
                       hidden_size=encoder_hidden_size, video_feature_size=encoder_rnn_size,
                       theme_size=theme_size, dt_memory_size=dt_memory_size, dtqv_memory_size=dtqv_memory_size)
        self.theme_size = theme_size
        self.dt_memory_size = dt_memory_size
        self.dtqv_memory_size = dtqv_memory_size
        self.decoder_rnn_input_size = decoder_rnn_input_size

        # Decoder RNN
        self.ch_decoder_rnn = ch_decoder_rnn
        self.decode_function = F.log_softmax

        # vocab size
        self.vocab_size = vocab_size

        # criterion
        self.criterion = criterion
        ### attention
        self.frame_attention = Frame_Attention(dim=encoder_rnn_size)
        self.clip_attention = Clip_Attention(dim=encoder_rnn_size)
        # self.attention = Attention(dim=encoder_rnn_size)
        # self.M = nn.Parameter(torch.rand(self.batch_size,1,self.encoder_rnn_size))
        self.MLP = nn.Linear(self.encoder_rnn_size*3,1)


        self.max_question_length = 10

        ##### key action
        self.window_size_frame = 4
        self.window_size_clip = 2
        self.video_encoder_rnn_frame = nn.GRU(input_size=encoder_rnn_size, hidden_size=encoder_hidden_size,
                                              num_layers=1, dropout=0.2, bidirectional=True, batch_first=True)
        self.video_encoder_rnn_clip = nn.GRU(input_size=encoder_rnn_size, hidden_size=encoder_hidden_size,
                                             num_layers=1, dropout=0.2, bidirectional=True, batch_first=True)


    def encoder_features(self, input_var, input_lengths, ):
        batch, seq_len, image_channel, image_height, image_width = input_var.size()

        ### previous
        cnn_input = []
        for i in range(batch):
            cnn_input.append(input_var[i, :input_lengths[i]])
        cnn_input = torch.cat(cnn_input, dim=0)

        #### 特征提取
        features = self.features(cnn_input)  # cnn_batch * 512 * image_height * image_width
        features = features.view(features.size(0), features.size(1))  #### 直接拉成一个（batch_size,512）的向量
        ranges = list(itertools.accumulate(input_lengths))  # example:[6, 12, 15, 17]
        ranges.insert(0, 0)  # example:[0, 6, 12, 15, 17]
        ranges = list(zip(ranges[:-1], ranges[1:]))  # example:[(0,6),(6,12),(12,15),(15,17)], len = batch
        delimit_length = 8
        input_lengths = torch.div(input_lengths, delimit_length // 2) - 1
        input_lengths=input_lengths.int()
        seq_len_new = torch.max(input_lengths)
        encoder_rnn_input = torch.zeros(features.size(0), self.encoder_rnn_size)
        frame_level_feature = []
        for i, (begin, end) in enumerate(ranges):
            encoder_rnn_input[begin:end] = features[begin:end]

            for jj in range(delimit_length, len(encoder_rnn_input[begin:end]) + 1, delimit_length // 2):
                ii = jj - delimit_length
                frame_attn_input = encoder_rnn_input[begin:end][ii:jj].cuda()
                frame_attn_output = self.frame_attention(frame_attn_input)
                frame_level_feature.append(frame_attn_output)
        frame_level_feature = torch.cat(frame_level_feature, dim=0)  ###把经过frame-level attention 的结果合并在一起
        encoder_rnn_input_new = torch.zeros(seq_len_new, batch, self.encoder_rnn_size)
        if torch.cuda.is_available():
            encoder_rnn_input_new = encoder_rnn_input_new.cuda()

        ranges_new = list(itertools.accumulate(input_lengths))  # example:[6, 12, 15, 17]
        ranges_new.insert(0, 0)  # example:[0, 6, 12, 15, 17]
        ranges_new = list(zip(ranges_new[:-1], ranges_new[1:]))  # example:[(0,6),(6,12),(12,15),(15,17)], len = batch
        for i, (begin, end) in enumerate(ranges_new):
            length = end - begin
            encoder_rnn_input_new[:length, i] = frame_level_feature[begin:end]

        # encoder processing
        self.video_encoder_rnn.flatten_parameters()
        encoder_rnn_input_new = encoder_rnn_input_new.transpose(0, 1)  # [batch, seq_len, rnn_size]
        _, idx_sort = torch.sort(input_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = idx_sort.cuda()
        idx_unsort = idx_unsort.cuda()
        encoder_rnn_input_new = encoder_rnn_input_new.index_select(dim=0, index=idx_sort)
        input_lengths = input_lengths.cuda().index_select(dim=0, index=idx_sort)

        encoder_rnn_input_new = nn.utils.rnn.pack_padded_sequence(encoder_rnn_input_new, input_lengths.cpu(),
                                                                  batch_first=True)
        encoder_output, encoder_hidden = self.video_encoder_rnn(encoder_rnn_input_new)
        encoder_output, _ = nn.utils.rnn.pad_packed_sequence(encoder_output, batch_first=True)
        encoder_output = encoder_output.index_select(dim=0, index=idx_unsort)
        encoder_hidden = encoder_hidden.index_select(dim=1, index=idx_unsort)
        # encoder_hidden = encoder_hidden

        input_lengths_clip = input_lengths
        encoder_output_clip = encoder_output
        encoder_hidden_clip = encoder_hidden
        return encoder_output_clip, input_lengths_clip, encoder_output_clip, encoder_hidden_clip

    def dialogue_encoder_features_video(self, video_batch, lengths_batch):
        """

        :param video_batch:  [bsz, seq_len, hidden]
        :param lengths_batch: [bsz]
        :return:
        """

        # Frame Attention
        bsz, seq_len, hidden = video_batch.size()
        frame_duration = 8
        frame_lengths = torch.div(lengths_batch, frame_duration//2) -1
        frame_max_length = torch.max(frame_lengths)
        frame_features = torch.zeros((bsz, int(frame_max_length), hidden)).cuda()
        for i in range(bsz):
            length_item = lengths_batch[i].item()
            for j in range(0, length_item, frame_duration // 2):
                end_index = j + frame_duration
                if end_index < length_item:
                    frame_input = video_batch[i][j:end_index].cuda()
                    frame_output = self.frame_attention(frame_input)
                    frame_features[i][j//(frame_duration // 2)] = frame_output


        # Frame GRU
        _, idx_sort = torch.sort(frame_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = idx_sort.cuda()
        idx_unsort = idx_unsort.cuda()
        frame_features = frame_features.index_select(dim=0, index=idx_sort)
        frame_lengths = frame_lengths.index_select(dim=0, index=idx_sort)

        frame_features = nn.utils.rnn.pack_padded_sequence(frame_features, frame_lengths.cpu(),
                                                                  batch_first=True)
        frame_features, frame_hidden = self.video_encoder_rnn(frame_features)
        frame_features, _ = nn.utils.rnn.pad_packed_sequence(frame_features, batch_first=True)
        frame_features = frame_features.index_select(dim=0, index=idx_unsort)
        frame_hidden = frame_hidden[1]
        frame_lengths = frame_lengths.index_select(dim=0, index=idx_unsort)
        clip_features=frame_features
        clip_lengths=frame_lengths



        return clip_features, clip_lengths

    def forward(self, video_batch, targets_batch, questions_batch, history_video_batch,
                history_question_batch, dialogue_batch, video_lengths_batch, targets_length_batch,
                question_lengths_batch, history_video_lengths_batch, history_question_lengths_batch,
                teacher_forcing_ratio=0, method=None, mode=None, use_dialogue_info=True, use_question_model=True):
        """
        :param video_batch: [bsz, max_frames_current, channel, height, width]
        :param targets_batch: [bsz, max_words_gts]
        :param questions_batch: [bsz, max_words_question_current]
        :param history_video_batch: [bsz, dialogue_window_size, max_frames_last, channel, height, width]
        :param history_question_batch: [bsz, dialogue_window_size, max_words_question_last]
        :param dialogue_batch: [bsz]
        :param video_lengths_batch: [bsz]
        :param targets_length_batch: [bsz]
        :param question_lengths_batch: [bsz]
        :param history_video_lengths_batch: [bsz, dialogue_window]
        :param history_question_lengths_batch: [bsz, dialogue_window]
        :param teacher_forcing_ratio: item
        :param method: str
        :return: None
        """

        # if not use_dialogue_info:   ######��ע����
        #     assert mode == "test"
        input_var = video_batch
        input_lengths = video_lengths_batch
        target_batch = targets_batch
        target_lengths = targets_length_batch
        question_batch = questions_batch
        question_lengths = question_lengths_batch
        batch = input_var.shape[0]

        encoder_output, input_lengths, batch_video, encoder_hidden = self.encoder_features(input_var, input_lengths.cpu())
        v_local=torch.mean(encoder_output,dim=1).unsqueeze(1)
        if use_question_model:
            q_hidden, q_outputs = self.question_embedding(question_batch, question_lengths)
            q_mix, v_mix, q_v = self.question_context(encoder_output, input_lengths, q_outputs)
            self.max_question_length=question_batch.shape[1]
            mix_qv = q_mix + v_mix
            q_local=q_hidden
            # q_local=q_mix.squeeze()
            # v_local=v_mix.squeeze()
            # qv_local=torch.cat((q_local,v_local),dim=1)
            v_new=v_mix
            q_new=q_mix

            if mode == "train":
                q_hat_features = torch.randn(batch, self.max_question_length, self.question_hidden_size).cuda()
                q_mix_hat, v_mix_hat, q_v_hat = self.question_context(encoder_output, input_lengths, q_hat_features)
                v_new=v_mix_hat
                q_new=q_mix_hat
                # q_mix_hat = q_mix_hat.squeeze()   ####without global , with local
                # v_mix_hat =v_mix_hat.squeeze()
                # qv_local = torch.cat((q_mix_hat, v_mix_hat), dim=1)
                # q_mix_hat and q_mix to caculate loss
            elif mode == "test":
                q_hat_features = torch.randn(batch, self.max_question_length, self.question_hidden_size).cuda()
                q_mix_hat, v_mix_hat, q_v_hat = self.question_context(encoder_output, input_lengths, q_hat_features)
                v_new=v_mix_hat
                q_new=q_mix_hat
                # q_mix_hat = q_mix_hat.squeeze()  ####without global , with local
                # v_mix_hat = v_mix_hat.squeeze()
                # qv_local = torch.cat((q_mix_hat, v_mix_hat), dim=1)
                # do nothing
                pass
            else:
                raise NotImplementedError
        else:
            assert mode == "test"
            q_hat_features = torch.randn(batch, self.max_question_length, self.question_hidden_size).cuda()
            q_mix, v_mix, q_v = self.question_context(encoder_output, input_lengths, q_hat_features)
            mix_qv = q_mix + v_mix
            q_mix_hat=q_mix
            v_new = v_mix
            q_new = q_mix

        # dialogue_video_features


        if mode == "train" or (mode == "test" and use_dialogue_info):
            _, dialogue_window_size, seq_len_history, _, _, _ = history_video_batch.size()
            history_video_batch_feature = torch.zeros(batch, dialogue_window_size, seq_len_history, self.encoder_rnn_size).cuda()

            features_history = []
            for i in range(batch):
                for j in range(dialogue_window_size):
                    features_history.append(history_video_batch[i][j][:history_video_lengths_batch[i][j]])
            features_history = torch.cat(features_history, dim=0)
            features_history = self.features(features_history)
            features_history = features_history.view(features_history.size(0), features_history.size(1))

            start_index = 0
            for i in range(batch):
                for j in range(dialogue_window_size):
                    length = history_video_lengths_batch[i][j]
                    history_video_batch_feature[i][j][:length] = features_history[start_index:start_index + length]
                    start_index += length


            GCM_video_batch = []
            GCM_lengths_batch = []
            for i in range(dialogue_window_size):
                info = self.dialogue_encoder_features_video(history_video_batch_feature[:,i,:,:], history_video_lengths_batch[:,i])
                GCM_video_batch.append(info[0])
                aa=(torch.from_numpy(info[1].cpu().detach().numpy())).int()
                GCM_lengths_batch.append(aa)
            history_video_max_length = max([video_feature.shape[1] for video_feature in GCM_video_batch])
            history_video_batch_feature = torch.zeros((batch, dialogue_window_size, history_video_max_length, GCM_video_batch[0].shape[2])).cuda()

            for i in range(dialogue_window_size):
                for j in range(batch):
                    tmp = GCM_video_batch[i][j][:GCM_lengths_batch[i][j]]
                    history_video_batch_feature[j,i,:GCM_lengths_batch[i][j],:] = tmp
            history_video_lengths_batch = torch.stack(GCM_lengths_batch, dim=1)


            history_video_batch_feature_new = []
            history_video_batch_lengths_new = []
            history_question_batch_new = []
            history_question_lengths_new = []

            history_video_batch_feature_new_local = []
            history_question_batch_feature_new_local = []

            for i in range(dialogue_window_size):
                per_video_feature = history_video_batch_feature[:,i,:,:]
                per_video_length = history_video_lengths_batch[:,i]
                per_question = history_question_batch[:,i,:]
                per_question_length = history_question_lengths_batch[:, i]
                per_q_hidden, per_q_outputs = self.question_embedding(per_question, per_question_length)
                per_q_mix, per_v_mix, per_q_v = self.question_context(per_video_feature, per_video_length, per_q_outputs)
                per_mix_qv = per_q_mix + per_v_mix
                per_q_hidden_local=per_q_hidden.squeeze()       #### without local
                per_video_feature_local=torch.mean(per_video_feature, dim=1)
                history_video_batch_feature_new.append(per_v_mix.squeeze(1))
                history_question_batch_new.append(per_q_mix.squeeze(1))
                history_video_batch_feature_new_local.append(per_video_feature_local)   #### without local
                history_question_batch_feature_new_local.append(per_q_hidden_local)
            history_video_batch_feature_new = torch.stack(history_video_batch_feature_new, dim=1).cuda()
            history_question_batch_new = torch.stack(history_question_batch_new, dim=1).cuda()



            dialogue_feature_batch = torch.cat((history_video_batch_feature_new, history_question_batch_new), dim=2)

            # dialogue_feature_batch_local=torch.cat((history_video_batch_feature_new_local,history_question_batch_feature_new_local),dim=2)   #### without local

        elif mode == "test" and not use_dialogue_info:

            # if use_question_model:
            #     # already hasds mix_qv
            #     pass
            # else:
            #     # Q_hat
            #     q_hat_features_dailogue = torch.randn(batch, self.max_question_length, self.question_hidden_size).cuda()
            #     q_mix_dialogue, v_mix_dialogue, q_v_dialogue = self.question_context(encoder_output, input_lengths, q_hat_features_dailogue)
            #     mix_qv_dialogue = q_mix_dialogue + q_mix_dialogue
            dialogue_feature_batch = torch.cat((q_mix, v_mix), dim=2)  # [bsz, 1, question_hidden + video_hidden]
        else:
            raise NotImplementedError


        if torch.cuda.is_available():
            batch_video = batch_video.cuda()
            question_batch = question_batch.long().cuda()
            input_lengths = input_lengths.long().cuda()
            question_lengths_batch = question_lengths_batch.long().cuda()
            dialogue_feature_batch = dialogue_feature_batch.cuda()  # [bsz, dialogue_window_size, hidden*2]
            # dialogue_feature_batch_local=dialogue_feature_batch_local.cuda()
        # history_video_batch_feature [bsz, dialogue_window_size, len_seq, hidden_dim]
        # GCMhidden [bsz, hidden*4]  -> b,t,q,v

        GCMhidden = self.GCM(batch_video=v_new,   ####v_new(with local); v_local(without local)
                             batch_question=q_new,    #####q_local(without local); q_new(with local)
                             batch_video_lengths=input_lengths,
                             batch_question_lengths=question_lengths_batch,
                             dialogue_feature=dialogue_feature_batch,      ####dialogue_feature_batch(with local); dialogue_feature_batch_local(without local)
                             batch_history_video=None,
                             batch_history_question=None,
                             batch_history_video_lengths=None,
                             batch_history_question_lengths=None)
        # GCMhidden = self.GCM(batch_video, question_batch, history_video_batch_feature, history_question_batch, input_lengths,
        #         question_lengths_batch, history_video_lengths_batch, history_question_lengths_batch)
        # GCMhidden = self.GCM(batch_video,  history_video_batch_feature,
        #                      input_lengths,
        #                      history_video_lengths_batch)

        if torch.cuda.is_available():
            target_batch = target_batch.cuda()
        target_batch = target_batch.long()
        if method is None:
            decoder_outputs, decoder_embedding, decoder_hidden, other = self.ch_decoder_rnn(inputs=target_batch,
                                                                                            input_lengths=input_lengths,     ###input_lengths  or  input_lengths
                                                                                            encoder_hidden=encoder_hidden,
                                                                                            encoder_outputs=encoder_output,   #####encoder_output  or  encoder_output
                                                                                            function=self.decode_function,
                                                                                            teacher_forcing_ratio=teacher_forcing_ratio,
                                                                                            q=mix_qv,
                                                                                            GCMhidden=GCMhidden)  ####GCMhidden(with global)  ; qv_local (without global)                        ###### q=q_k_new(����global context module����ں�����) or q_v(����local context module����ں�����)

            if use_question_model:
                return decoder_outputs, decoder_hidden, other, target_batch, decoder_hidden, q_mix, q_mix_hat
            else:
                return decoder_outputs, decoder_hidden, other, target_batch, decoder_hidden, q_mix, q_mix_hat
            # if mode == "train":
            #     return decoder_outputs, decoder_hidden, other, target_batch, decoder_hidden, q_mix, q_mix_hat
            # elif mode == "test":
            #     return decoder_outputs, decoder_hidden, other, target_batch, decoder_hidden, q_mix, q_mix_hat
            # else:
            #     raise NotImplementedError

        elif method == 'beam-search':
            # beam-search
            assert teacher_forcing_ratio == 0
            decoded_batch = self.ch_decoder_rnn.beam_decode(inputs=target_batch,
                                                            input_lengths=input_lengths,     ###input_lengths  or  input_lengths
                                                            encoder_hidden=encoder_hidden,
                                                            encoder_outputs=encoder_output,   #####encoder_output  or  encoder_output
                                                            function=self.decode_function,
                                                            teacher_forcing_ratio=teacher_forcing_ratio,
                                                            q=mix_qv,
                                                            GCMhidden=GCMhidden)
            return decoded_batch, target_batch
        elif method == 'greedy-search':
            # gready-search
            self.greedy_decode(target_batch, encoder_hidden, encoder_output)


    def greedy_decode(self, trg, decoder_hidden, encoder_outputs, ):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        seq_len, batch_size = trg.size()
        decoded_batch = torch.zeros((batch_size, seq_len))
        # decoder_input = torch.LongTensor([[EN.vocab.stoi['<sos>']] for _ in range(batch_size)]).cuda()
        decoder_input = Variable(trg.data[0, :]).cuda()  # sos
        # print(decoder_input.shape)
        for t in range(seq_len):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)  # [32, 10004] get candidates
            topi = topi.view(-1)
            decoded_batch[:, t] = topi

            decoder_input = topi.detach().view(-1)

        return decoded_batch


