#-*-coding:utf-8-*- 
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_entire import Attention
from .baseRNN import BaseRNN
from queue import PriorityQueue
import operator
import gensim
import pickle



class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_VIDEO_ATTN_CONTEXT_FEATURE = 'video_context_feature'
    KEY_VIDEO_ATTN_SCORE = 'video_attention_score'
    KEY_VIDEO_ENCODER_OUTPUTS = 'video_encoder_outputs'
    KEY_VIDEO_ENCODER_HIDDEN = 'video_encoder_hidden'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, hidden_size, sos_id, eos_id, n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False, use_Question=True, attention_type="Global", window_size=3):
        super(DecoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.use_Question = use_Question
        # if use_Question:
        #     self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        # else:
        #     self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.hidden_size = hidden_size
        self.attention_type = attention_type
        self.window_size = window_size

        self.init_input = None
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            if self.attention_type == 'Global':
                self.attention = Attention(self.hidden_size)
            elif self.attention_type == 'Local':
                self.attention = LocalAttention(self.window_size, self.hidden_size)

        self.out = nn.Linear(self.hidden_size * 3, self.output_size)   # with q
        # self.out = nn.Linear(self.hidden_size * 2, self.output_size)     # without q
        # self.out = nn.Linear(self.hidden_size, self.output_size)      #### without global

        self.max_length = 50



    def forward_step(self, input_var, hidden, encoder_outputs, function, q=None, GCMhidden=None):

        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        GCMhidden = GCMhidden.unsqueeze(1)
        if hidden.shape[0] == 2:
            hidden = hidden[1].unsqueeze(0)

        self.rnn.flatten_parameters()
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn, context_feature = self.attention(output, encoder_outputs)

        if output.shape[1] != GCMhidden.shape[1]:
            GCMhidden = GCMhidden.repeat(1, output.shape[1], 1)
        output_withhidden = torch.cat((output, GCMhidden), dim=2)  #### with global

        #### with q
        predicted_softmax = function(self.out(output_withhidden.contiguous().view(-1, self.hidden_size * 3)),
                                     dim=1).view(batch_size,
                                                 output_size,
                                                 -1)
        return predicted_softmax, output, hidden, attn, context_feature

    def forward(self, inputs=None, input_lengths=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0, cnn_outputs=None, q=None, GCMhidden=None):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_VIDEO_ATTN_SCORE] = list()
            ret_dict[DecoderRNN.KEY_VIDEO_ATTN_CONTEXT_FEATURE] = list()
        ret_dict[DecoderRNN.KEY_VIDEO_ENCODER_OUTPUTS] = encoder_outputs

        # max_length 为当前batch中减去<sos>的长度
        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)
        ret_dict[DecoderRNN.KEY_VIDEO_ENCODER_HIDDEN] = decoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        self.attention.set_mask(input_lengths, encoder_outputs)

        def decode(step, step_output, step_attn, step_context_feature):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_VIDEO_ATTN_SCORE].append(step_attn)
                ret_dict[DecoderRNN.KEY_VIDEO_ATTN_CONTEXT_FEATURE].append(step_context_feature)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        decoder_embeddings = torch.zeros(batch_size, max_length - 1, self.hidden_size)
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_embedding, decoder_hidden, attn, context_feature = self.forward_step(decoder_input,
                                                                                                         decoder_hidden,
                                                                                                         encoder_outputs,
                                                                                                         function=function,
                                                                                                         q=None,
                                                                                                         GCMhidden=GCMhidden)
            decoder_embeddings = decoder_embedding[:, :-1, :]

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn.unsqueeze(1), context_feature[:, di, :].unsqueeze(1))
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_embedding, decoder_hidden, step_attn, step_context_feature = self.forward_step(
                    decoder_input,
                    decoder_hidden,
                    encoder_outputs,
                    function=function,
                    q=None,
                    GCMhidden=GCMhidden)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn, step_context_feature)
                decoder_input = symbols
                if di < max_length - 1:
                    decoder_embeddings[:, di, :] = decoder_embedding.squeeze()

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        # TODO 需要看懂为什么lengths不行呢
        # ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()
        attn_context_feature = ret_dict[DecoderRNN.KEY_VIDEO_ATTN_CONTEXT_FEATURE]
        attn_context_feature = torch.cat(attn_context_feature, 1)
        ret_dict[DecoderRNN.KEY_VIDEO_ATTN_CONTEXT_FEATURE] = attn_context_feature

        # KEY_VIDEO_ATTN_CONTEXT_FEATURE = 'video_context_feature'
        # KEY_VIDEO_ATTN_SCORE = 'video_attention_score'
        # KEY_VIDEO_ENCODER_OUTPUTS = 'video_encoder_outputs'
        # KEY_VIDEO_ENCODER_HIDDEN = 'video_encoder_hidden'
        # KEY_LENGTH = 'length'   *
        # KEY_SEQUENCE = 'sequence'

        # ret = dict()
        # ret[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        return decoder_outputs, decoder_embeddings, decoder_hidden, ret_dict

    def infer(self, inputs=None, input_lengths=None, encoder_hidden=None, encoder_outputs=None,
              function=F.log_softmax):

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs, function, 0)
        decoder_hidden = self._init_state(encoder_hidden)

        sequence_symbols = []

        self.attention.set_mask(input_lengths, encoder_outputs)

        def decode(step_output):
            symbols = step_output.topk(1)[1]
            sequence_symbols.append(symbols)
            return symbols

        decoder_input = inputs[:, 0].unsqueeze(1)

        for di in range(max_length):
            decoder_output, decoder_embedding, decoder_hidden, step_attn, step_context_feature = self.forward_step(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
                function=function)
            step_output = decoder_output.squeeze(1)
            symbols = decode(step_output)
            eos = symbols.data.eq(self.eos_id).cpu().view(-1).numpy()
            if eos == 1:
                break
            decoder_input = symbols

        return sequence_symbols

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None

        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2).contiguous()
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length

    def beam_decode(self, target_tensor, encoder_hidden, encoder_outputs=None, input_lengths=None, q=None):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hiddens: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        beam_width = 10
        topk = 5  # how many sentence do you want to generate
        decoded_batch = []

        target_tensor, batch_size, max_length = self._validate_args(target_tensor, encoder_hidden, encoder_outputs,
                                                             function=None, teacher_forcing_ratio=0)
        decoder_hiddens = self._init_state(encoder_hidden)
        # decoder_hidden = decoder_hidden.contiguous()
        # decoding goes sentence by sentence
        for idx in range(batch_size):  # batch_size

            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (
                    decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(
                    1).contiguous()  # [Directions*Layers, B, H]=>[Directions*Layers,H]=>[Directions*Layers,1,H]

            encoder_output = encoder_outputs[idx, :, :].unsqueeze(0)  # [B,T,H] -> [T, H]->[1, T, H]

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([self.sos_id]).cuda()

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                # print('--best node seqs len {} '.format(n.leng))
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == self.eos_id and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                _, decoder_output, decoder_hidden, _, _ = self.forward_step(decoder_input.unsqueeze(1),
                    decoder_hidden,
                    encoder_output,
                    function=F.log_softmax,
                    q=q)
                decoder_output = decoder_output.squeeze(1)
                # decoder_output, decoder_hidden, _ = self.forward(inputs=decoder_input,
                #                                                         input_lengths=input_lengths,
                #                                                         encoder_hidden=encoder_hidden,
                #                                                         encoder_outputs=encoder_output,
                #                                                         function=F.log_softmax,
                #                                                         teacher_forcing_ratio=0,
                #                                                         q=q)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)  #log_prob [1,1,beam_size]
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(-1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        # return None, None, None, None
        return decoded_batch


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward  # 注意这里是有惩罚参数的，参考恩达的 beam-search

    def __lt__(self, other):
        return self.leng < other.leng  # 这里展示分数相同的时候怎么处理冲突，具体使用什么指标，根据具体情况讨论

    def __gt__(self, other):
        return self.leng > other.leng

