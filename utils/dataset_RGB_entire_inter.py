#-*-coding:utf-8-*- 
import os
import cv2
import numpy as np
# import jpeg4py as jpeg
import torch
from torch.utils.data import Dataset
from PIL import Image
import random


def token2id(vocab, string):
    # convert token sentence to id target:
    target = [vocab.index('<sos>')]

    for ch in string:
        if ch in vocab:
            target.append(vocab.index(ch))
        else:
            target.append(vocab.index('<unk>'))
    target.append(vocab.index('<eos>'))
    return target


def q_token2id(vocab, string):
    target = []
    for ch in string:
        if ch in vocab:
            target.append(vocab.index(ch))
        else:
            target.append(vocab.index('<unk>'))
    return target


class LipReadingDataSet(Dataset):
    def __init__(self, index_file, ch_vocab, q_ch_vocab, history, history_window, transforms=None, dialogue_ch_vocab=None,mode=None):
        with open(index_file, encoding="GBK") as f:
            self.index = []
            for line in f.readlines():
                self.index.append(line.strip())
        self.ch_vocab = ch_vocab
        self.q_ch_vocab = q_ch_vocab
        self.transforms = transforms
        self.name_video = {}
        self.history = history
        self.history_window = history_window
        self.dialogue_ch_vocab = dialogue_ch_vocab
        self.mode=mode


    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        # /ssd1/dataset/Interactive_SLR_image/银行/存款/0049_V003_05Q我知道 还有 业务 办理 要Q您 可以 选择 自助 办理Q银行Q存款Q1
        video_path, gts, question, theme, dialogue, turns_id = self.index[item].split("Q")   ####我们的
        # video_path, gts, question, dialogue, turns_id = self.index[item].split("Q")   ##### CSL


        # 每个gts做五遍，在5遍中每取一个，有5个人在做
        turns_qv = self.history[theme][dialogue]      ####CSL


        qvs = []
        if self.mode=='train':
            for i in range(0, len(turns_qv)-15, 15):
                qvs.append(random.choice(turns_qv[i:i+15]))

        else:
            for i in range(0, len(turns_qv)-5, 5):
                qvs.append(random.choice(turns_qv[i:i+5]))

        turns_id = int(turns_id) + 1



        qv_pairs = qvs[:turns_id]
        if len(qv_pairs) > self.history_window:
            qv_pairs = qv_pairs[len(qv_pairs)-self.history_window:]


        while len(qv_pairs) < self.history_window:
            qv_pairs.insert(0, qv_pairs[0])
            # 对话轮数还不够
        pairs = []
        for pair in qv_pairs:
            #### CSL
            pri_question, pri_path = pair  #### CSL
            pri_video = self._load_mp4(pri_path) #### CSL
            #### CSL
            ##### 我们的
            # pri_path, pri_question = pair
            # pri_video = self._load_mp4(pri_path)
            ##### 我们的
            pri_question_character = q_token2id(vocab=self.q_ch_vocab, string=pri_question.split(" "))
            pairs.append((pri_video, pri_question_character))
        frames = self._load_mp4(video_path)
        gts_character = token2id(vocab=self.ch_vocab, string=gts.split(" "))
        question_character = q_token2id(vocab=self.q_ch_vocab, string=question.split(" "))

        dialogue = self.dialogue_ch_vocab.index(dialogue)

        return frames, gts_character, question_character, pairs, dialogue



    def _load_mp4(self, mp4_name):
        frames = []
        filenames = os.listdir(mp4_name)
        filenames.sort()
        filenames = filenames[10:-10]

        for i in range(0, len(filenames),2):
            path = os.path.join(mp4_name, filenames[i])
            frame = Image.open(path)  ####原来的

            frames.append(frame)
        if self.transforms:
            frames = self.transforms(frames)

        # frames = np.array(frames)
        return frames

def project_collect_fn(batch):
    frames_batch, gts_character_batch, question_character_batch, pairs_batch, dialogue_batch = zip(*batch)
    frames_batch = list(frames_batch)
    gts_character_batch = list(gts_character_batch)
    question_character_batch = list(question_character_batch)
    pairs_batch = list(pairs_batch)
    dialogue_batch = list(dialogue_batch)

    # frames
    video_input_length = [len(video) for video in frames_batch]
    video_max_length = max(video_input_length)
    shape, dtype = frames_batch[0][0].shape, frames_batch[0][0].dtype
    frames_tensor = []
    for video in frames_batch:
        video.extend([torch.zeros(shape, dtype=dtype)]*(video_max_length - len(video)))
        video = torch.stack(video, dim=0)
        frames_tensor.append(video)
    frames_tensor = torch.stack(frames_tensor, dim=0)
    video_input_length = np.asarray(video_input_length)
    decreasing_indices = (-video_input_length).argsort()
    frames_tensor = frames_tensor[decreasing_indices]
    video_input_length = video_input_length[decreasing_indices]
    # q_ch
    q_ch_batch = [question_character_batch[i] for i in decreasing_indices]
    question_lengths = [len(question) for question in q_ch_batch]
    question_max_length = max(question_lengths)
    question_lengths = np.array(question_lengths, np.int32)
    for q_ch_list in q_ch_batch:
        q_ch_list.extend([1] * (question_max_length - len(q_ch_list)))
    q_ch_batch = np.array(q_ch_batch, np.int32)
    # ch
    ch_batch = [gts_character_batch[i] for i in decreasing_indices]
    target_lengths = [len(target) for target in ch_batch]
    target_max_length = max(target_lengths)
    target_lengths = np.array(target_lengths, np.int32)
    for ch_list in ch_batch:
        ch_list.extend([1] * (target_max_length - len(ch_list)))
    ch_batch = np.array(ch_batch, np.int32)

    dialogue_batch = [dialogue_batch[i] for i in decreasing_indices]
    dialogue_batch = np.array(dialogue_batch, np.int32)

    # pairs
    pairs_batch = [pairs_batch[i] for i in decreasing_indices]
    inner_pairs_video_length = []
    inner_pairs_question_length = []
    for pairs in pairs_batch:
        # pairs
        # type:
        # list [(video(list format), question)]
        video_length = []
        question_length = []
        for pair in pairs:
            video_list, question_list = pair
            # print(question_list)
            # print("len(question_list)"+str(len(question_list)))
            video_length.append(len(video_list))
            question_length.append(len(question_list))

        # video_length = [len(pair[0]) for pair in pairs]
        # question_length = [len(pair[1]) for pair in pairs]
        inner_pairs_question_length.append(question_length)
        inner_pairs_video_length.append(video_length)
    inner_pairs_video_length = np.asarray(inner_pairs_video_length, np.int32)
    inner_pairs_question_length = np.asarray(inner_pairs_question_length, np.int32)
    inner_max_video_length = np.max(inner_pairs_video_length)
    inner_max_question_length = np.max(inner_pairs_question_length)

    pairs_video_tensor = []
    pairs_question_tensor = []
    for pairs in pairs_batch:
        video_tensors = []
        question_tensors = []
        for pair in pairs:
            video_tensor, question_tensor = pair
            video_tensor.extend([torch.zeros(shape, dtype=dtype)]*(inner_max_video_length - len(video_tensor)))
            question_tensor.extend([1]*(inner_max_question_length - len(question_tensor)))
            video_tensor = torch.stack(video_tensor, dim=0)
            question_tensor = torch.Tensor(question_tensor)
            # question_tensor = torch.stack(question_tensor, dim=0)
            video_tensors.append(video_tensor)
            question_tensors.append(question_tensor)
        # video_tensors.extend([]*(pairs_max_length - len(pairs)))
        # question_tensors.extend([]*(pairs_max_length - len(pairs)))

        video_tensors = torch.stack(video_tensors, dim=0)
        question_tensors = torch.stack(question_tensors, dim=0)

        pairs_video_tensor.append(video_tensors)
        pairs_question_tensor.append(question_tensors)
    pairs_video_tensor = torch.stack(pairs_video_tensor, dim=0)
    pairs_question_tensor = torch.stack(pairs_question_tensor, dim=0)

    # tensor value
    ch_batch = torch.from_numpy(ch_batch).long()
    q_ch_batch = torch.from_numpy(q_ch_batch).long()


    # length
    target_lengths = torch.from_numpy(target_lengths).long()
    question_lengths = torch.from_numpy(question_lengths).long()
    inner_pairs_video_length = torch.from_numpy(inner_pairs_video_length).long()
    inner_pairs_question_length = torch.from_numpy(inner_pairs_question_length).long()
    video_input_length = torch.from_numpy(video_input_length).long()
    dialogue_batch = torch.from_numpy(dialogue_batch).long()


    # target_lengths = torch.from_numpy(target_lengths)
    #
    # q_ch_batch = torch.from_numpy(q_ch_batch)
    # question_lengths = torch.from_numpy(question_lengths)

    return frames_tensor, ch_batch, q_ch_batch, pairs_video_tensor, pairs_question_tensor, dialogue_batch, video_input_length, target_lengths, question_lengths,  inner_pairs_video_length, inner_pairs_question_length


def q_collect_fn(batch):
    images_list_batch, ch_batch, q_ch_batch = zip(*batch)

    images_list_batch = list(images_list_batch)
    ch_batch = list(ch_batch)
    q_ch_batch = list(q_ch_batch)

    video_input_lengths = [len(images_list) for images_list in images_list_batch]
    video_max_length = max(video_input_lengths)
    image_shape, dtype = images_list_batch[0][0].shape, images_list_batch[0][0].dtype
    # for images_list in images_list_batch:
    #     images_list.extend([np.zeros(image_shape, dtype=dtype)] * (video_max_length - len(images_list)))

    images_tensor = []
    for images_list in images_list_batch:
        images_list.extend([torch.zeros(image_shape, dtype=dtype)] * (video_max_length - len(images_list)))
        images_list = torch.stack(images_list, dim=0)
        images_tensor.append(images_list)

    # images_list_batch = np.asarray(images_list_batch)
    images_list_batch = torch.stack(images_tensor, dim=0)

    video_input_lengths = np.asarray(video_input_lengths)

    decreasing_indices = (-video_input_lengths).argsort()
    images_list_batch = images_list_batch[decreasing_indices]
    video_input_lengths = video_input_lengths[decreasing_indices]

    # q_ch
    q_ch_batch = [q_ch_batch[i] for i in decreasing_indices]
    question_lengths = [len(question) for question in q_ch_batch]
    question_max_length = max(question_lengths)
    question_lengths = np.array(question_lengths,np.int32)

    for q_ch_list in q_ch_batch:
        q_ch_list.extend([1] * (question_max_length - len(q_ch_list)))
    q_ch_batch = np.array(q_ch_batch, np.int32)

    # ch
    ch_batch = [ch_batch[i] for i in decreasing_indices]

    target_lengths = [len(target) for target in ch_batch]
    target_max_length = max(target_lengths)
    target_lengths = np.array(target_lengths, np.int32)

    for ch_list in ch_batch:
        ch_list.extend([1] * (target_max_length - len(ch_list)))
    ch_batch = np.array(ch_batch, np.int32)

    # to tensor
    # images_list_batch = torch.from_numpy(images_list_batch).float().div(255)

    # video_input_lengths = video_input_lengths.tolist()
    video_input_lengths = torch.from_numpy(video_input_lengths).long()
    ch_batch = torch.from_numpy(ch_batch)

    target_lengths = torch.from_numpy(target_lengths)

    q_ch_batch = torch.from_numpy(q_ch_batch)
    question_lengths = torch.from_numpy(question_lengths)

    return images_list_batch, video_input_lengths, ch_batch, target_lengths, q_ch_batch, question_lengths


def collect_fn(batch):
    images_list_batch, ch_batch, = zip(*batch)

    images_list_batch = list(images_list_batch)
    ch_batch = list(ch_batch)

    video_input_lengths = [len(images_list) for images_list in images_list_batch]
    # video_input_lengths = [len(image_list) for images_list in images_list_batch]
    video_max_length = max(video_input_lengths)
    image_shape, dtype = images_list_batch[0][0].shape, images_list_batch[0][0].dtype
    images_tensor = []
    for images_list in images_list_batch:
        images_list.extend([torch.zeros(image_shape, dtype=dtype)] * (video_max_length - len(images_list)))
        images_list = torch.stack(images_list, dim=0)
        images_tensor.append(images_list)
    images_list_batch = torch.stack(images_tensor, dim=0)

    # images_list_batch = np.asarray(images_list_batch)
    video_input_lengths = np.asarray(video_input_lengths)

    decreasing_indices = (-video_input_lengths).argsort()
    images_list_batch = images_list_batch[decreasing_indices]
    video_input_lengths = video_input_lengths[decreasing_indices]

    ch_batch = [ch_batch[i] for i in decreasing_indices]

    target_lengths = [len(target) for target in ch_batch]
    target_max_length = max(target_lengths)
    target_lengths = np.array(target_lengths, np.int32)

    for ch_list in ch_batch:
        ch_list.extend([1] * (target_max_length - len(ch_list)))
    ch_batch = np.array(ch_batch, np.int32)

    # images_list_batch = torch.from_numpy(images_list_batch).float().div(255)
    video_input_lengths = video_input_lengths.tolist()

    ch_batch = torch.from_numpy(ch_batch)

    target_lengths = torch.from_numpy(target_lengths)

    return images_list_batch, video_input_lengths, ch_batch, target_lengths
