import os
import jieba
import matplotlib.pyplot as plt
from gensim.models.word2vec import LineSentence
from gensim import corpora, models
from collections import Counter
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def data_process(data_path, output_path='./corpus.txt'):
    output_file = open(output_path, 'w', encoding='utf-8')
    file_listdir = os.listdir(data_path)

    # if os.path.exists(output_path):
    #     print("Data already prepared!")
    #     return

    # 需要替换的文本数据中的字符
    char_to_be_replaced = "\n `1234567890-=/*-~!@#$%^&*()_+qwertyuiop[]\\QWERTYUIOP{}|asdfghjkl;" \
                          "'ASDFGHJKL:\"zxcvbnm,./ZXCVBNM<>?~！@#￥%……&*（）——+【】：；“‘’”《》？，。" \
                          "、★「」『』～＂□ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ" \
                          "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ" \
                          "ｕｖｗｙｚ￣\u3000\x1a"
    char_to_be_replaced = list(char_to_be_replaced)

    # stopwords_path = 'cn_stopwords.txt'
    # stop_words = []
    # 读取停用词
    # with open(stopwords_path, 'r', encoding='utf-8') as f:
    #     stop_words = set([word.strip() for word in f.readlines()])
    
    for file_name in file_listdir:
        if file_name == 'inf.txt':
            continue
        path = os.path.join(data_path, file_name)    
        if os.path.isfile(path):
            with open(path, 'r', encoding='GB18030', errors='ignore') as tmp_file:
                tmp_file_context = tmp_file.read()
                tmp_file_lines = tmp_file_context.split('。')
                for tmp_line in tmp_file_lines:
                    for tmp_char in char_to_be_replaced:
                        tmp_line = tmp_line.replace(tmp_char, "")
                    # for tmp_char in stop_words:
                    #     tmp_line = tmp_line.replace(tmp_char, "")
                    tmp_line = tmp_line.replace("本书来自免费小说下载站更多更新免费电子书请关注", "")
                    if tmp_line == "":
                        continue
                    tmp_line = list(jieba.cut(tmp_line))
                    tmp_line_seg = ""
                    for tmp_word in tmp_line:
                        tmp_line_seg += tmp_word + " "
                    output_file.write(tmp_line_seg.strip() + "\n")
    output_file.close()

def main():
    # 中文语料库的实际路径
    folder_path = 'data'
    output_path= './corpus.txt'
    stop_path = './cn_stopwords.txt'
    # data_process(folder_path, output_path)
    
    # sentences = LineSentence(output_path)

    # model_cbow = models.word2vec.Word2Vec(sg=0, vector_size=200, window=5, min_count=5, workers=8)
    # model_cbow.build_vocab(sentences)
    # model_cbow.train(sentences, total_examples=model_cbow.corpus_count, epochs=model_cbow.epochs)
    # model_cbow.save("./model_cbow.model")
    
    # model_skip_gram = models.word2vec.Word2Vec(sg=1, vector_size=200, window=5, min_count=5, workers=8)
    # model_skip_gram.build_vocab(sentences)
    # model_skip_gram.train(sentences, total_examples=model_skip_gram.corpus_count, epochs=model_skip_gram.epochs)
    # model_skip_gram.save("./model_skip_gram.model")

    model_cbow = models.word2vec.Word2Vec.load("./model_cbow.model")
    model_skip_gram = models.word2vec.Word2Vec.load("./model_skip_gram.model")

    character_names = ["黄蓉", "杨过", "张无忌", "令狐冲", "韦小宝", "峨嵋派", "屠龙刀", "蛤蟆功", "葵花宝典"]
    print("Results of CBOW:")
    for tmp_word in character_names:
        print("{}".format(tmp_word), model_cbow.wv.most_similar(tmp_word, topn=5))
    print("------------------")
    print("Results of Skip Gram:")
    for tmp_word in character_names:
        print("{}".format(tmp_word), model_skip_gram.wv.most_similar(tmp_word, topn=5))

    stop_words = []
    # 读取停用词
    with open(stop_path, 'r', encoding='utf-8') as f:
        stop_words = set([word.strip() for word in f.readlines()])
    
    # Getting the mostly frequent words in the corpus...
    with open("./corpus.txt", "r", encoding="utf-8", errors="ignore") as tmp_file:
        whole_corpus = tmp_file.read()
        whole_corpus = whole_corpus.replace("\n", "")
        whole_corpus = whole_corpus.split(" ")
        words_counter = Counter(whole_corpus)

    frequent_words = []
    for tmp_key, tmp_value in words_counter.items():
        if tmp_value >= 50:
            if tmp_key not in stop_words:
                frequent_words.append(tmp_key)
    
    # print(frequent_words)

    print("tSNE visualization...")
    word_vectors = []
    for tmp_word in frequent_words:
        try:
            word_vectors.append(model_skip_gram.wv[tmp_word])
        except:
            continue
    
    word_vectors = np.array(word_vectors)
    tSNE = TSNE()
    word_embeddings = tSNE.fit_transform(word_vectors)
    classifier = KMeans(n_clusters=16)
    classifier.fit(word_embeddings)
    labels = classifier.labels_

    min_left = min(word_embeddings[:, 0])
    max_right = max(word_embeddings[:, 0])
    min_bottom = min(word_embeddings[:, 1])
    max_top = max(word_embeddings[:, 1])

    markers = ["bo", "go", "ro", "co", "mo", "yo", "ko", "bx", "gx", "rx", "cx", "mx", "yx", "kx", "b>", "g>"]

    for i in range(len(word_embeddings)):
        plt.plot(word_embeddings[i][0], word_embeddings[i][1], markers[labels[i]])
    plt.axis([min_left, max_right, min_bottom, max_top])
    plt.savefig("./tSNE.png")





if __name__ == '__main__':
    main()