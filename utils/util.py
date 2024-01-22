#coding:utf-8
from collections import OrderedDict
# from pypinyin import pinyin, Style
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import torch
import numpy as np
from matplotlib.font_manager import _rebuild

_WEIGHT_KEYS = ['kernel', 'beta', 'alpha']
_WEIGHT_KEYS += [key + ':0' for key in _WEIGHT_KEYS]


def displayAttention(decoder_outputs, attentions, type='enc2dec'):
    """

    :param encoder_outputs:
    :param decoder_outputs:
    :param attentions:
    :return:
    """
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # _rebuild()
    if type == 'enc2dec':
        attentions = torch.stack(attentions, dim=1).squeeze().cpu().detach().numpy()  # decoder_steps * encoder_steps
    else:
        # question attention
        attentions = attentions.squeeze(0).cpu().detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone', aspect='auto')
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title('Attention score map')
    if type == 'question':
        ax.set_xlabel('encoder_output')
        ax.set_ylabel('question feature')
        attention_path = './tmp_question.png'
        plt.savefig(attention_path, format='png')
    elif type == 'enc2dec':
        ax.set_xlabel('encoder_output')
        ax.set_ylabel('decoder_step')
        # ax.set_yticklabels([''] + ['<sos>'] + decoder_outputs[0][0].split(' ')[1:])
        # ax.set_yticklabels([''] + ['<sos>'] + ['my'] +['name'] +['is'] + ['lihaibo'] + ['ha'] +['ha'])
        attention_path = './tmp_encoder_decoder.png'
        plt.savefig(attention_path, format='png')
    else:
        raise NotImplementedError
    plt.close()




def state_dict_layer_names(state_dict):
    layer_names = [".".join(k.split('.')[:-1]) for k in state_dict.keys()]
    # Order preserving unique set of names
    return list(OrderedDict.fromkeys(layer_names))


def _contains_weights(keras_h5_layer):
    for key in _WEIGHT_KEYS:
        if key in keras_h5_layer:
            return True
    return False


def dig_to_params(keras_h5_layer):
    # Params are hidden many layers deep in keras HDF5 files for
    # some reason. e.g. h5['model_weights']['conv1']['dense_1'] \
    # ['dense_2']['dense_3']['conv2d_7']['dense_4']['conv1']
    while not _contains_weights(keras_h5_layer):
        keras_h5_layer = keras_h5_layer[list(keras_h5_layer.keys())[0]]

    return keras_h5_layer


initial_confusion_set_for_lipreading = [['b', 'p', 'm', 'f'], ['d', 't', 'n', 'l'], ['g', 'k', 'h'], ['j', 'q', 'x'],
                                        ['zh', 'ch', 'sh', 'r'], ['z', 'c', 's'], ['w', 'y']]
final_confusion_set_for_lipreading = [['a', 'ai', 'an', 'ang'], ['i', 'in', 'ing', 'ie'], ['e', 'ei', 'en', 'eng'],
                                      ['ia', 'ian', 'iang'], ['uan', 'uang'], ['o', 'ou'], ['ong' 'iong'],
                                      ['v', 'u', 've', 'uo', 'ui', 'un', 'ue', 'iu'], ['ua', 'uai'], ['iao', 'ao']]


def get_final_list(s):
    finals = pinyin(s, style=Style.FINALS, heteronym=False, strict=False)
    strict_finals = pinyin(s, style=Style.FINALS, heteronym=False)

    final_list = []
    for final, strict_final in zip(finals, strict_finals):
        write_final = final[0]
        if final[0] == 'u' and strict_final[0] == 'v':
            write_final = 'v'
        final_list.append(write_final)
    return final_list


def is_equal(s1, s2, index1, index2):
    final_list_1 = get_final_list(s1)
    final_list_2 = get_final_list(s2)

    if final_list_1[index1] == final_list_2[index2]:
        return True
    else:
        return False


def find_lcseque(s1, s2):
    s1 = s1[0]
    s1 = s1.split()
    s2 = s2.split()
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if is_equal(s1, s2, p1, p2):
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'

    s = []
    (p1, p2) = (len(s1), len(s2))
    index1 = []
    index2 = []

    while m[p1][p2]:
        c = d[p1][p2]
        if c == 'ok':
            index1.append(p1 - 1)
            index2.append(p2 - 1)
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':
            p2 -= 1
        if c == 'up':
            p1 -= 1
    s.reverse()
    index1.reverse()
    index2.reverse()

    return index1, index2
