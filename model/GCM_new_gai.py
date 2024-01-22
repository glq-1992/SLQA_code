import torch
import torch.nn as nn
import torch.nn.functional as F

class GCM(nn.Module):

    def __init__(self, vocab_size, embeded_size, hidden_size, video_feature_size, theme_size, dt_memory_size, dtqv_memory_size):
        super(GCM, self).__init__()

        # Module
        self.embed = nn.Embedding(vocab_size, embeded_size)
        self.themeembeded = nn.Embedding(theme_size, embeded_size)
        self.dropout = nn.Dropout(p=0.5)
        self.question_rnn = nn.LSTM(input_size=embeded_size, hidden_size=hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.video_rnn = nn.LSTM(input_size=video_feature_size, hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.video_rnn_fc = nn.Linear(in_features=hidden_size*2, out_features=hidden_size)
        # self.drnn = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, batch_first=True, num_layers=1, bidirectional=False)
        self.drnn = nn.LSTM(input_size=hidden_size*4, hidden_size=hidden_size, batch_first=True, num_layers=1, bidirectional=False)
        self.fc = nn.Linear(hidden_size*3, hidden_size*2)
        self.fc_v = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cm = nn.Linear(hidden_size*2, hidden_size*4)
        self.fc_final = nn.Linear(hidden_size * 2, hidden_size * 4)
        # variable
        self.vocab_size = vocab_size
        self.embeded_size = embeded_size
        self.hidden_size = hidden_size
        self.video_feature_size = video_feature_size
        self.theme_size = theme_size
        self.dt_memory_size = dt_memory_size
        self.dtqv_memory_size = dtqv_memory_size


        self.dt_look_fc = nn.Linear(hidden_size*3, hidden_size*2)
        # self.dtqv_look_fc = nn.Linear(hidden_size*4, hidden_size*2)
        # memory
        # self.dt_memory = []
        # self.dtqv_memory = []
        self.tau = 5
        self.tau_decay = -0.05

    def update(self, key, memory_name):
        bsz, _ ,dim= key.size()
        if memory_name == "dt":
            if bsz + len(self.dt_memory) > self.dt_memory_size:
                # delete
                aranges = bsz + len(self.dt_memory) - self.dt_memory_size
                for i in range(aranges):
                    self.dt_memory.pop(0)

                # add
                for i in range(bsz):
                    self.dt_memory.append(key[i].data)
            else:
                for i in range(bsz):
                    self.dt_memory.append(key[i].data)

        elif memory_name == "dtqv":
            if bsz + len(self.dtqv_memory) > self.dtqv_memory_size:
                # delete
                aranges = bsz + len(self.dtqv_memory) - self.dtqv_memory_size
                for i in range(aranges):
                    self.dtqv_memory.pop(0)

                # add
                for i in range(bsz):
                    self.dtqv_memory.append(key[i].data)
            else:
                for i in range(bsz):
                    self.dtqv_memory.append(key[i].data)
        else:
            raise NotImplementedError


    def _init_state(self, hidden, bidirectional=False):
        """ Initialize the hidden state. """
        if hidden is None:
            return None

        if isinstance(hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h, bidirectional) for h in hidden])
        else:
            encoder_hidden = self._cat_directions(hidden)
        return encoder_hidden

    def _cat_directions(self, h, bidirectional):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2).contiguous()
        return h

    def lookup(self, query, memory):
        """

        :param query: [bsz, quert_feature_size]
        :param memory: [bsz, memory_size, memory_feature_size]
        :return: result : [bsz, memory_feature_size]
        """
        bsz, _ = query.size()
        if isinstance(memory, str):
            # dt, dtqv situation
            if memory == "dt":
                if len(self.dt_memory) == 0:
                    return None
                else:
                    # dt_memory [len(self.dt_memory), dt_hidden]
                    dt_memory = torch.stack(self.dt_memory, dim=0)
                    # dt_memory = dt_memory
                    # dt_memory [bsz, len(self.dt_memory), dt_hidden]
                    # dt_memory = dt_memory.repeat(bsz, 1, 1)
                    query = query.unsqueeze(1)
                    query = self.dt_look_fc(query)
                    context = torch.bmm(query, dt_memory.permute(0,2,1))
                    # context = context.squeeze(1)
                    # context_probability = context.softmax(dim=1)
                    # _, Maxindex = torch.max(context_probability, dim=1)
                    location = torch.log(context.clamp(min=1e-8))
                    if self.training:
                        action = F.gumbel_softmax(location, self.tau, hard=True, dim=2)
                    else:
                        action = F.gumbel_softmax(location, 1e-5, hard=True, dim=2)
                    result = torch.bmm(action, dt_memory).squeeze(1)
                    # # lookup the best result
                    # result = torch.zeros(bsz, dt_memory.shape[2]).cuda()
                    # for index in range(bsz):
                    #     result[index] = dt_memory[index][Maxindex[index]]
                    return result
            elif memory == "dtqv":
                if len(self.dtqv_memory) == 0:
                    return None
                else:
                    # dt_memory [len(self.dt_memory), dt_hidden]
                    dtqv_memory = torch.stack(self.dtqv_memory, dim=0)
                    # dtqv_memory = dtqv_memory.unsqueeze(0)
                    # # dt_memory [bsz, len(self.dt_memory), dt_hidden]
                    # dtqv_memory = dtqv_memory.repeat(bsz, 1, 1)
                    query = self.fc(query)
                    query = query.unsqueeze(1)
                    # query = self.dtqv_look_fc(query)
                    context = torch.bmm(query, dtqv_memory.permute(0,2,1))
                    # context = context.squeeze(1)
                    # context_probability = context.softmax(dim=1)
                    # _, Maxindex = torch.max(context_probability, dim=1)
                    location = torch.log(context.clamp(min=1e-8))
                    if self.training:
                        action = F.gumbel_softmax(location, self.tau, hard=True, dim=2)
                    else:
                        action = F.gumbel_softmax(location, 1e-5, hard=True, dim=2)
                    result = torch.bmm(action, dtqv_memory).squeeze(1)
                    # lookup the best result
                    # result = torch.zeros(bsz, dtqv_memory.shape[2]).cuda()
                    # for index in range(bsz):
                    #     result[index] = dtqv_memory[index][Maxindex[index]]
                    return result
            else:
                raise NotImplementedError
        elif isinstance(memory, torch.Tensor):
            _, query_feature_size = query.size()
            bsz, memory_size, memory_feature_size = memory.size()
            # query: [bsz, 1, quert_feature_size]
            query = query.unsqueeze(1)
            # context: [bsz, 1, memory_size]
            context = torch.bmm(query, memory.permute(0,2,1))
            # context: [bsz, memory_size]
            # context = context.squeeze(1)
            # context_probability = context.softmax(dim=1)
            # # Maxindex [bsz]
            # _, Maxindex = torch.max(context_probability, dim=1)
            #
            # # lookup the best result
            # result = torch.zeros(bsz, memory_feature_size).cuda()
            # for index in range(bsz):
            #     result[index] = memory[index][Maxindex[index]]
            location = torch.log(context.clamp(min=1e-8))
            if self.training:
                action = F.gumbel_softmax(location, self.tau, hard=True, dim=2)
            else:
                action = F.gumbel_softmax(location, 1e-5, hard=True, dim=2)
            result = torch.bmm(action, memory).squeeze(1)
            return result
        else:
            raise NotImplementedError



    def forward(self, batch_video, batch_question, batch_video_lengths, batch_question_lengths,dialogue_feature=None, batch_history_video=None, batch_history_question=None,
                 batch_history_video_lengths=None, batch_history_question_lengths=None):

        """
        :param batch_video: [bsz, max_frames, video_feature_size]
        :param batch_question: [bsz, max_question_words]
        :param batch_history_video: [bsz, dialogue_window, max_frames, video_feature_size]
        :param batch_history_question: [bsz, dialogue_window, max_question_words]
        :param batch_video_lengths: [bsz]
        :param batch_question_lengths: [bsz]
        :param batch_history_video_lengths: [bsz, dialogue_window]
        :param batch_history_question_lengths: [bsz, dialogue_window]
        :param dialogue_feature: [bsz, dialogue_window_size, hidden*2]
        :return:
        """
        self.dt_memory = []
        self.dtqv_memory = []

        batch_history_video_question_concat, batch_history_video_question_concat_hidden = self.drnn(dialogue_feature)
        # [bsz, hidden_concat]
        # batch_history_video_question_concat_hidden = self._init_state(batch_history_video_question_concat_hidden, bidirectional=False)[0].squeeze(0)
        batch_history_video_question_concat_hidden = self._init_state(batch_history_video_question_concat_hidden, bidirectional=False)[0][-1]

        # theme memory

        # new added
        bsz = batch_history_video_question_concat.shape[0]


        theme_memory = torch.Tensor(list(range(0, self.theme_size))).long().cuda()
        theme_memory = theme_memory.unsqueeze(1)
        # [theme_nums, embed_size]
        theme_memory = self.dropout(self.themeembeded(theme_memory)).squeeze(1)
        theme_memory = theme_memory.unsqueeze(0)
        # [bsz, theme_nums, embed_size]
        theme_memory = theme_memory.repeat(bsz, 1, 1)
        # theme_star [bsz, embed_size]
        theme_star = self.lookup(batch_history_video_question_concat_hidden, theme_memory)

        theme_star_new = theme_star.unsqueeze(1).repeat(1, batch_history_video_question_concat.size(1), 1)
        d_t = torch.cat((batch_history_video_question_concat, theme_star_new), dim=2)

        selected_s=torch.cat((batch_history_video_question_concat_hidden,theme_star),dim=1)
        d_t_global = self.fc_cm(selected_s)


        # update
        self.update(d_t, "dt")


        batch_video = self.fc_v(batch_video)
        batch_question = self.fc_v(batch_question)
        # 新增加的
        batch_qv = torch.cat((batch_video, batch_question), dim=2)
        theme_star = theme_star.unsqueeze(1)
        batch_lookup_qv = torch.cat((batch_qv, theme_star), dim=2)
        selected_dt = self.lookup(batch_lookup_qv.squeeze(1), "dt")
        selected_dt_cm = self.fc_cm(selected_dt)
        batch_dtqv = torch.cat((batch_qv, selected_dt.unsqueeze(1)), dim=1)
        self.update(batch_dtqv, "dtqv")

        batch_lookup_dtqv = torch.cat((batch_video, selected_dt.unsqueeze(1)), dim=2)
        selected_dtqv = self.lookup(batch_lookup_dtqv.squeeze(1), "dtqv")
        selected_dtqv=self.fc_final(selected_dtqv)
        return selected_dtqv     ##### without cm:selected_dt_cm; 原来：selected_dtqv; d_t_global:without DM


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    model =GCM(vocab_size=500, embeded_size=256, hidden_size=256,
             video_feature_size=512, theme_size=21, dt_memory_size=40,
             dtqv_memory_size=40)
    model = model.cuda()
    # batch_video, batch_question, batch_history_video, batch_history_question, batch_video_lengths, batch_question_lengths,
    # batch_history_video_lengths, batch_history_question_lengths
    bsz = 2
    max_frames = 400
    video_feature_size = 512
    max_question_words = 20
    dialogue_window = 4

    batch_video = torch.randn((bsz, max_frames, video_feature_size)).cuda()
    batch_question = torch.randint(low=0, high=max_question_words-1, size=(bsz, max_question_words)).cuda()
    batch_history_video = torch.randn((bsz, dialogue_window, max_frames, video_feature_size)).cuda()
    batch_history_question = torch.randint(low=0, high=max_question_words-1, size=(bsz, dialogue_window, max_question_words)).cuda()
    batch_video_lengths = torch.randint(low=max_frames//2, high=max_frames-1, size=(bsz,1)).squeeze().cuda()
    batch_question_lengths = torch.randint(low=max_question_words//2, high=max_question_words-1, size=(bsz, 1)).squeeze().cuda()
    batch_history_question_lengths = torch.randint(low=max_question_words//2, high=max_question_words-1, size=(bsz, dialogue_window)).cuda()
    batch_history_video_lengths = torch.randint(low=max_frames//2, high=max_frames-1, size=(bsz, dialogue_window)).cuda()

    # writer = SummaryWriter(log_dir="/hd2/lihaibo/exp_result/GCMbaseline")

    # with SummaryWriter(log_dir="/hd2/lihaibo/exp_result/GCMbaseline") as writer:
    # writer.add_graph(model, (batch_video, batch_question, batch_history_video, batch_history_question, batch_video_lengths,
    #             batch_question_lengths, batch_history_video_lengths, batch_history_question_lengths))
    # writer.close()
    res = model(batch_video, batch_question, batch_history_video, batch_history_question, batch_video_lengths,
                batch_question_lengths, batch_history_video_lengths, batch_history_question_lengths)
    # writer.close()
    # print(model)
    print(res)
