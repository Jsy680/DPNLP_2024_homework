import os  
import random  
from matplotlib import pyplot as plt  
import numpy as np  
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class CorpusDataset(Dataset):
    def __init__(self, source_data, target_data, source_word_2_idx, target_word_2_idx, device):
        self.source_data = source_data  # 源数据
        self.target_data = target_data  # 目标数据
        self.source_word_2_idx = source_word_2_idx  # 源数据词汇到索引的映射
        self.target_word_2_idx = target_word_2_idx  # 目标数据词汇到索引的映射
        self.device = device

    def __getitem__(self, index):
        # 获取指定索引处的数据并转换为索引
        src = self.source_data[index]
        tgt = self.target_data[index]

        src_index = [self.source_word_2_idx[i] for i in src]
        tgt_index = [self.target_word_2_idx[i] for i in tgt]

        return src_index, tgt_index

    
    def batch_data_alignment(self, batch_datas):
        # 对批处理数据进行对齐
        src_index, tgt_index = [], []  # 存储源句子和目标句子的索引列表
        src_len, tgt_len = [], []  # 存储源句子和目标句子的长度列表

        for src, tgt in batch_datas:
            src_index.append(src)
            tgt_index.append(tgt)
            src_len.append(len(src))
            tgt_len.append(len(tgt))

        max_src_len = max(src_len)
        max_tgt_len = max(tgt_len)
        src_index = [[self.source_word_2_idx["<BOS>"]] + tmp_src_index + [self.source_word_2_idx["<EOS>"]] +
                     [self.source_word_2_idx["<PAD>"]] * (max_src_len - len(tmp_src_index)) for tmp_src_index in src_index]
        # 为每个源句子添加开始和结束标志，并进行填充使其长度一致
        tgt_index = [[self.target_word_2_idx["<BOS>"]] + tmp_src_index + [self.target_word_2_idx["<EOS>"]] +
                     [self.target_word_2_idx["<PAD>"]] * (max_tgt_len - len(tmp_src_index)) for tmp_src_index in tgt_index]
        # 为每个目标句子添加开始和结束标志，并进行填充使其长度一致
        src_index = torch.tensor(src_index, device=self.device)
        tgt_index = torch.tensor(tgt_index, device=self.device)

        return src_index, tgt_index

    def __len__(self):
        assert len(self.source_data) == len(self.target_data)  # 确保源数据和目标数据长度一致
        return len(self.target_data)


class Encoder(nn.Module):
    def __init__(self, dim_encoder_embbeding, dim_encoder_hidden, source_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(source_corpus_len, dim_encoder_embbeding)  # 词嵌入层
        self.lstm = nn.LSTM(dim_encoder_embbeding, dim_encoder_hidden, batch_first=True)  # LSTM 层

    def forward(self, src_index):
        en_embedding = self.embedding(src_index)  # 获取嵌入表示
        _, encoder_hidden = self.lstm(en_embedding)  # 通过 LSTM 获取隐藏状态

        return encoder_hidden


class Decoder(nn.Module):
    def __init__(self, dim_decoder_embedding, dim_decoder_hidden, target_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(target_corpus_len, dim_decoder_embedding)  # 词嵌入层
        self.lstm = nn.LSTM(dim_decoder_embedding, dim_decoder_hidden, batch_first=True)  # LSTM 层

    def forward(self, decoder_input, hidden):
        embedding = self.embedding(decoder_input)  # 获取嵌入表示
        decoder_output, decoder_hidden = self.lstm(embedding, hidden)  # 通过 LSTM 获取输出和隐藏状态

        return decoder_output, decoder_hidden  # 返回输出和隐藏状态

# 定义序列到序列模型类
class Seq2Seq(nn.Module):
    def __init__(self, dim_encoder_embbeding, dim_encoder_hidden, source_corpus_len,
                 dim_decoder_embedding, dim_decoder_hidden, target_corpus_len):
        super().__init__()
        self.encoder = Encoder(dim_encoder_embbeding, dim_encoder_hidden, source_corpus_len)
        self.decoder = Decoder(dim_decoder_embedding, dim_decoder_hidden, target_corpus_len)
        self.classifier = nn.Linear(dim_decoder_hidden, target_corpus_len)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, src_index, tgt_index):
        decoder_input = tgt_index[:, :-1]  # 解码器输入为目标句子去掉最后一个词
        label = tgt_index[:, 1:]  # 标签为目标句子去掉第一个词

        encoder_hidden = self.encoder(src_index)  # 获取编码器隐藏状态
        decoder_output, _ = self.decoder(decoder_input, encoder_hidden)  # 获取解码器输出

        pre = self.classifier(decoder_output)  # 通过分类器得到预测结果
        loss = self.ce_loss(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))  # 计算损失

        return loss  # 返回损失

# 生成句子函数
def generate_sentence(sentence, source_word_2_idx, model, device, target_word_2_idx, target_idx_2_word):
    src_index = torch.tensor([[source_word_2_idx[i] for i in sentence]], device=device)  # 将句子转换为索引表示

    result = []  # 存储生成的句子
    encoder_hidden = model.encoder(src_index)  # 获取编码器隐藏状态
    decoder_input = torch.tensor([[target_word_2_idx["<BOS>"]]], device=device)  # 初始化解码器输入

    decoder_hidden = encoder_hidden  # 初始化解码器隐藏状态
    while True:
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)  # 获取解码器输出和隐藏状态
        pre = model.classifier(decoder_output)  # 通过分类器得到预测结果

        w_index = int(torch.argmax(pre, dim=-1))  # 获取预测词的索引
        word = target_idx_2_word[w_index]  # 将索引转换为词

        if word == "<EOS>" or len(result) > 40:  # 若遇到结束标志或生成句子长度超过40，则停止生成
            break

        result.append(word)  # 添加生成的词到结果中
        decoder_input = torch.tensor([[w_index]], device=device)  # 更新解码器输入

    return "".join(result)  # 返回生成的句子


def data_process(data_path, num_corpus=300, num_test_corpus=10):
    source_target_corpus_ori = []  # 原始源目标语料对

    # 需要替换的文本数据中的字符
    char_to_be_replaced = "\n 0123456789qwertyuiopasdfghjklzxcvbnm[]{};':\",./<>?ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ‘’「」" \
                          "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ“”" \
                          "ｕｖｗｙｚ￣\u3000\x1a"
    char_to_be_replaced = list(char_to_be_replaced)  # 将需要替换的字符转换为列表

    with open(data_path, 'r', encoding='gbk', errors='ignore') as tmp_file:
        tmp_file_context = tmp_file.read() 
        for tmp_char in char_to_be_replaced:
            tmp_file_context = tmp_file_context.replace(tmp_char, "")
        tmp_file_context = tmp_file_context.replace("本书来自免费小说下载站更多更新免费电子书请关注", "")

        tmp_file_sentences = tmp_file_context.split("。")
        for tmp_idx, tmp_sentence in enumerate(tmp_file_sentences):
            if ("她" in tmp_sentence) and (10 <= len(tmp_sentence) <= 40) and (10 <= len(tmp_file_sentences[tmp_idx + 1]) <= 40):
                source_target_corpus_ori.append((tmp_file_sentences[tmp_idx], tmp_file_sentences[tmp_idx + 1]))                

    sample_indexes = random.sample(list(range(len(source_target_corpus_ori))), num_corpus)
    source_corpus, target_corpus = [], []
    for idx in sample_indexes:
        source_corpus.append(source_target_corpus_ori[idx][0])  # 添加选中的源句子
        target_corpus.append(source_target_corpus_ori[idx][1])  # 添加选中的目标句子

    test_corpus = []
    for idx in range(len(source_target_corpus_ori)):
        if idx not in sample_indexes:
            test_corpus.append((source_target_corpus_ori[idx][0], source_target_corpus_ori[idx][1]))
    test_corpus = random.sample(test_corpus, num_test_corpus)  # 随机选择测试语料
    test_source_corpus, test_target_corpus = [], []  # 初始化测试源语料和测试目标语料
    for tmp_src, tmp_tgt in test_corpus:
        test_source_corpus.append(tmp_src)  # 添加测试源句子
        test_target_corpus.append(tmp_tgt)  # 添加测试目标句子

    return test_corpus, source_corpus, target_corpus, test_source_corpus, test_target_corpus

# 主函数
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2  # 批次大小
    num_corpus = 300  # 语料数量
    num_test_corpus = 10  # 测试语料数量
    txt_file_path = "./神雕侠侣.txt"  # 文本文件路径
    num_epochs = 200  # 训练轮数
    lr = 0.001  # 学习率

    dim_encoder_embbeding = 150  # 编码器嵌入维度
    dim_encoder_hidden = 100  # 编码器隐藏层维度
    dim_decoder_embedding = 150  # 解码器嵌入维度
    dim_decoder_hidden = 100  # 解码器隐藏层维度

    # 处理数据
    test_corpus, source_corpus, target_corpus, test_source_corpus, test_target_corpus = data_process(data_path=txt_file_path,
                                                                                       num_corpus=num_corpus,
                                                                                       num_test_corpus=num_test_corpus)

    # 生成 one-hot 编码字典
    idx_cnt = 0
    word_2_idx_dict = dict() 
    idx_2_word_list = list() 
    for tmp_corpus in [source_corpus, target_corpus, test_source_corpus, test_target_corpus]:
        for tmp_sentence in tmp_corpus:
            for tmp_word in tmp_sentence:
                if tmp_word not in word_2_idx_dict.keys():
                    word_2_idx_dict[tmp_word] = idx_cnt 
                    idx_2_word_list.append(tmp_word) 
                    idx_cnt += 1 

    one_hot_dict_len = len(word_2_idx_dict)
    word_2_idx_dict.update({"<PAD>": idx_cnt, "<BOS>": idx_cnt + 1, "<EOS>": idx_cnt + 2}) 
    idx_2_word_list.extend(["<PAD>", "<BOS>", "<EOS>"]) 

    source_word_2_idx = {word: idx for idx, word in enumerate(idx_2_word_list)}  # 生成源词到索引的映射
    target_word_2_idx = {word: idx for idx, word in enumerate(idx_2_word_list)}  # 生成目标词到索引的映射
    source_idx_2_word = {idx: word for idx, word in enumerate(idx_2_word_list)}  # 生成源索引到词的映射
    target_idx_2_word = {idx: word for idx, word in enumerate(idx_2_word_list)}  # 生成目标索引到词的映射

    # 创建数据集和数据加载器
    train_dataset = CorpusDataset(source_corpus, target_corpus, source_word_2_idx, target_word_2_idx, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.batch_data_alignment)

    # 初始化模型
    model = Seq2Seq(dim_encoder_embbeding, dim_encoder_hidden, len(source_word_2_idx),
                    dim_decoder_embedding, dim_decoder_hidden, len(target_word_2_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器

    losses = []

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0  # 初始化总损失
        for batch_idx, (src_index, tgt_index) in enumerate(train_loader):
            optimizer.zero_grad()  # 清空梯度
            loss = model(src_index, tgt_index)  # 前向传播计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加损失

        avg_loss = total_loss / len(train_loader)  # 计算平均损失
        losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')  # 打印当前轮次的平均损失

    
    plt.figure()
    plt.plot(np.array([i+1 for i in range(num_epochs)]), losses, label="Seq2Seq")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Seq2Seq")

    plt.savefig("./training_loss_seq2seq.png")


    model.eval()  # 评估模式
    for src_sentence, tgt_sentence in zip(test_source_corpus, test_target_corpus):
        generated_sentence = generate_sentence(src_sentence, source_word_2_idx, model, device, target_word_2_idx, target_idx_2_word)
        print(f'Source: {src_sentence}')  # 打印源句子
        print(f'Target: {tgt_sentence}')  # 打印目标句子
        print(f'Generated: {generated_sentence}')  # 打印生成的句子
        print('---')

    

if __name__ == "__main__":
    main()
