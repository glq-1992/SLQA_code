#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import Levenshtein as Lev
import torch
from six.moves import xrange
from nlgeval import NLGEval
from datetime import datetime
from nlgeval import compute_individual_metrics


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (string): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, labels, blank_index=1):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        self.nlg = NLGEval(metrics_to_omit=[
                                             'SkipThoughtCS', 'EmbeddingAverageCosineSimilarity',
                                            'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity',
                                            'GreedyMatchingScore'])

    def metric_batch(self, decoderouts_, references_, batch_size):
        transcripts = []
        _references = list()
        er , acc_word, acc_sentence = 0, 0, 0

        if len(decoderouts_[0]) > 1:
            # beam-search situation
            for x in range(len(references_)):
                _min_cer = float('inf')
                _min_index = -1
                for j in range(len(decoderouts_[x])):
                    transcript, reference = decoderouts_[x][j], references_[x][0]
                    _er = self.er(transcript, reference) / float(len(reference.split()))
                    if _er < _min_cer:
                        _min_cer = _er
                        _min_index = j
                        print(transcript)
                        print(reference)
                        _acc_sentence = self.acc(transcript, reference, 'acc_sentence')
                        _acc_word = self.acc(transcript, reference, 'acc_word') / float(len(transcript.split()))
                transcripts.append(decoderouts_[x][_min_index])
                _references.append(references_[x][0])
                er += _min_cer
                acc_sentence += _acc_sentence
                acc_word += _acc_word
        else:
            for x in range(len(references_)):
                transcript, reference = decoderouts_[x][0], references_[x][0]
                transcripts.append(transcript)
                _references.append(reference)
                er += self.er(transcript, reference) / float(len(reference.split()))
                acc_sentence += self.acc(transcript, reference, 'acc_sentence')
                # print(float(len(transcript.split())))
                # print(float(self.acc(transcript, reference, 'acc_word')))
                if len(transcript.split())==0:
                    pass
                else:
                    acc_word += self.acc(transcript, reference, 'acc_word') / float(len(transcript.split()))



        references = [_references]

        metrics_dict = self.nlg.compute_metrics(references, transcripts)
        metrics_dict['er'] = er / batch_size
        metrics_dict['acc_word'] = acc_word / batch_size
        metrics_dict['acc_sentence'] = acc_sentence / batch_size
        return metrics_dict

    def acc(self, s1, s2, acc_type):

        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))
        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]
        if acc_type == 'acc_sentence':
            distance = Lev.distance(''.join(w1), ''.join(w2))
            return 1 if distance == 0 else 0
        elif acc_type == 'acc_word':
            acc_word = [v for v in w1 if v in w2]
            return len(acc_word)
        else :
            raise NotImplementedError

    def metric_translate(self, s1, s2):
        # s1 transcript
        # s2 reference
        # b = set(s1.split() + s2.split())
        # word2char = dict(zip(b, range(len(b))))
        # w1 = [chr(word2char[w]) for w in s1.split()]
        # w2 = [chr(word2char[w]) for w in s2.split()]
        # metrics_dict = self.nlg.compute_individual_metrics([' '.join(w2)], ' '.join(w1))
        metrics_dict = self.nlg.compute_individual_metrics([s2], s1)

        return metrics_dict['Bleu_1'], metrics_dict['Bleu_2'], metrics_dict['Bleu_3'],metrics_dict['Bleu_4'], \
               metrics_dict['METEOR'], metrics_dict['ROUGE_L'], metrics_dict['CIDEr']

    def er(self, s1, s2):
        """
        Computes the pinyin Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        distance = Lev.distance(''.join(w1), ''.join(w2))

        return distance

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=1):
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                else:
                    string += ' '
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                                   remove_repetitions=True, return_offsets=True)
        return strings, offsets

    def seq2seq_decode(self, seqlist_tensor, method=None):

        if method == 'beam-search':
            utterances = []
            batch_size = len(seqlist_tensor)
            for i in range(batch_size):
                utterance = []
                beam_result = seqlist_tensor[i]
                for item in beam_result:
                    string = ''
                    for s in range(len(item)):
                        char = self.int_to_char[item[s].item()]
                        if char == '<sos>' or char == '<pad>':
                            continue
                        elif char == '<eos>':
                            break
                        else:
                            string += ' '
                            string = string + char
                    utterance.append(string)
                utterances.append(utterance)
            return utterances
        else:
            strings = []
            batch_size = seqlist_tensor.size(0)
            seq_len = seqlist_tensor.size(1)
            if seq_len > 10000:
                print(seq_len)
            for i in range(batch_size):
                string = ''
                for s in range(seq_len):
                    char = self.int_to_char[seqlist_tensor[i][s].item()]
                    if char == '<sos>' or char == '<pad>':
                        continue
                    elif char == '<eos>':
                        break
                    else:
                        string += ' '
                        string = string + char
                strings.append([string])
            return strings
