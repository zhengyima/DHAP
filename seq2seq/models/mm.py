import torch
import torch.nn as nn

class MemoryNN(nn.module):
    def __init__(self, vocab_size, max_len, hidden_size, embedding_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
                 embedding=None, update_embedding=True):
    
        super(MemoryNN,self).__init__(vocab_size, max_len, hidden_size, embedding_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.attention_grucell = nn.GRUCell(hidden_size, hidden_size)
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        self.memory_grucell = nn.GRUCell(hidden_size, hidden_size)
        self.gate = nn.Sequential(
                        nn.Linear(hidden_size * 4, hidden_size),
                        nn.Tanh(),
                        nn.Linear(hidden_size, 1),
                        nn.Sigmoid()
                    )
        self.init_weight()


    def init_hidden(self, batch_size):
        '''GRU的初始hidden。单层单向'''
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        hidden = get_variable(hidden)
        return hidden

    def init_weight(self):
        components = [self.rnn, self.gate, self.attention_grucell,
                     self.memory_grucell]
        for component in components:
            for name, param in component.state_dict().items():
                if 'weight' in name:
                    nn.init.xavier_normal(param)

        
    def forward(self, allhis, allhis_mask, encoded_questions, n_episode=3):
        bsize = allhis.size(0)
        n_his = allhis.size(1)
        len_his = allhis.size(2)
        encoded_his = []
        # 对每一条数据，计算his编码
        for his, his_mask in zip(allhis, allhis_mask):
            his_embeds = self.embedding(his)
            hidden = self.init_hidden(n_his)
            # 1.1 把输入(多条句子)给到GRU
            # b=nf, [nf, flen, h], [1, nf, h]
            outputs, hidden = self.rnn(his_embeds, hidden)
            # 1.2 每条句子真正结束时(real_len)对应的输出，作为该句子的hidden。GRU：ouput=hidden
            real_hiddens = []

            for i, o in enumerate(outputs):
                real_len = his_mask[i].data.tolist().count(0)
                real_hiddens.append(o[real_len - 1])
            # 1.3 把所有单个_his连接起来，unsqueeze(0)是为了后面的所有batch的cat
            hiddens = torch.cat(real_hiddens).view(n_his, -1).unsqueeze(0)
            encoded_his.append(hiddens)
        # [b, n_his, h]
        encoded_his = torch.cat(encoded_his)
        # 3. Memory模块

        memory = encoded_questions
        for i in range(n_episode):
            # e
            e = self.init_hidden(bsize).squeeze(0)
            # [nhis, b, h]
            encoded_hiss_t = encoded_his.transpose(0, 1)
            # 根据memory, episode，计算每一时刻的e。最终的e和memory来计算新的memory
            for t in range(n_his):
                # [b, h]
                bhis = encoded_his_t[t]
                # TODO 计算4个特征，论文是9个
                f1 = bhis * encoded_questions
                f2 = bhis * memory
                f3 = torch.abs(bhis - encoded_questions)
                f4 = torch.abs(bhis - memory)
                z = torch.cat([f1, f2, f3, f4], dim=1)
                # [b, 1] 对每个his的注意力
                gt = self.gate(z)
                e = gt * self.attention_grucell(bhis, e) + (1 - gt) * e
            # 每一轮的e和旧memory计算新的memory
            memory = self.memory_grucell(e, memory)
        return memory

















