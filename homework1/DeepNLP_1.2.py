import os
import pdb
import time
import math
import jieba
from collections import defaultdict


def data_preprocessing(data_roots):
    listdir = os.listdir(data_roots)  # 列出目录中的所有文件
    # 需要替换的文本数据中的字符
    char_to_be_replaced = "\n `1234567890-=/*-~!@#$%^&*()_+qwertyuiop[]\\QWERTYUIOP{}|asdfghjkl;" \
                          "'ASDFGHJKL:\"zxcvbnm,./ZXCVBNM<>?~！@#￥%……&*（）——+【】：；“‘’”《》？，。" \
                          "、★「」『』～＂□ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ" \
                          "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ" \
                          "ｕｖｗｙｚ￣\u3000\x1a"
    char_to_be_replaced = list(char_to_be_replaced)
    txt_corpus = []
    txt_names = []
    for tmp_file_name in listdir:
        path = os.path.join(data_roots, tmp_file_name)
        if os.path.isfile(path):
            # 读取文件内容并替换不需要的字符
            with open(path, "r", encoding="gbk", errors="ignore") as tmp_file:
                tmp_file_context = tmp_file.read()
                for tmp_char in char_to_be_replaced:
                    tmp_file_context = tmp_file_context.replace(tmp_char, "")
                tmp_file_context = tmp_file_context.replace("本书来自免费小说下载站更多更新免费电子书请关注", "")
                txt_corpus.append(tmp_file_context)
                txt_names.append(tmp_file_name.split(".txt")[0])
    return txt_corpus, txt_names


def calculate_characters_entropy(txt_corpus):
    """计算字熵"""
    len_char = 0
    chars_counter = defaultdict(int)
    for tmp_txt_context in txt_corpus:
        # 统计每个字符的出现次数
        for tmp_char in tmp_txt_context:
            chars_counter[tmp_char] += 1
        len_char += len(tmp_txt_context)
    # 根据字符出现次数计算熵
    char_entropy = 0
    for char_item in chars_counter.items():
        char_entropy += (-(char_item[1] / len_char) * math.log(char_item[1] / len_char, 2))
    return len_char, char_entropy


def calculate_words_entropy(txt_corpus):
    """计算词熵"""
    len_words = 0
    words_counter = defaultdict(int)
    for tmp_txt_context in txt_corpus:
        # 使用结巴分词将文本分词
        for tmp_word in jieba.cut(tmp_txt_context):
            words_counter[tmp_word] += 1
            len_words += 1
    # 根据词出现次数计算熵
    word_entropy = 0
    for word_item in words_counter.items():
        word_entropy += (-(word_item[1] / len_words) * math.log(word_item[1] / len_words, 2))
    return len_words, word_entropy


if __name__ == '__main__':

    start_time = time.time()
    data_roots = 'data'
    txt_corpus, txt_names = data_preprocessing(data_roots)
    len_char = []
    char_entropy = []
    len_words = []
    words_entropy = []
    cost_times = []
    for i in range(len(txt_corpus)):
        time0 = time.time()
        file_len_char, file_char_entropy = calculate_characters_entropy(txt_corpus[i])
        file_len_words, file_words_entropy = calculate_words_entropy(txt_corpus[i])
        time1 = time.time()
        cost_times.append(round(time1 - time0, 2))
        len_char.append(file_len_char)
        char_entropy.append(file_char_entropy)
        len_words.append(file_len_words)
        words_entropy.append(file_words_entropy)
    end_time = time.time()

    with open('Entropy.txt', 'w') as f:
        for i in range(len(txt_corpus)):
            f.write('====================================================\n')
            f.write(f'小说题目: {txt_names[i]}\n')
            f.write(f'计算花费时间: {cost_times[i]}\n')
            f.write(f'小说总字数: {cost_times[i]}\n')
            f.write(f'小说分词总个数: {cost_times[i]}\n')
            f.write(f'小说平均词长: {cost_times[i]}\n')
            f.write(f'小说基于字的中文平均信息熵: {cost_times[i]}bits\n')
            f.write(f'小说基于词的中文平均信息熵: {cost_times[i]}s\n')

        f.write('====================================================\n')
        f.write(f"运行总时间：{round(end_time - start_time, 2)}\n")
        f.write(f"数据库总字数：{sum(len_char)}\n")
        f.write(f"数据库分词总个数：{sum(len_words)}\n")
        f.write(f"平均词长：{round(sum(len_char) / sum(len_words), 3)}\n")
        f.write(f"基于字的中文平均信息熵为：{round(sum(char_entropy), 2)} bits\n")
        f.write(f"基于词的中文平均信息熵为：{round(sum(words_entropy), 2)} bits\n")
        f.close()
