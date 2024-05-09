import os
import random
import jieba
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora, models
from sklearn.svm import SVC


# 设置matplotlib支持中文显示
# plt.rcParams['font.sans-serif'] = ['Black']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def dataset_process(data_path, is_abandon_stop_words=True, stopwords_path = 'cn_stopwords.txt'):
    # 读取小说的文本内容，合并为一个语料库
    # 根据小说的名字分类
    corpus = []
    labels = []

    # 需要替换的文本数据中的字符
    char_to_be_replaced = " `1234567890-=/*-~!@#$%^&*()_+qwertyuiop[]\\QWERTYUIOP{}|asdfghjkl;" \
                          "'ASDFGHJKL:\"zxcvbnm,./ZXCVBNM<>?~！@#￥%……&*（）——+【】：；“‘’”《》？，。" \
                          "、★「」『』～＂□ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ" \
                          "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ" \
                          "ｕｖｗｙｚ￣\u3000\x1a"
    
    stop_words = []
    if is_abandon_stop_words:
        # 读取停用词
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stop_words = set([word.strip() for word in f.readlines()])

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='GB18030') as file:
                text = file.read()
                words = text.replace("\n", " ")  # 去除文章中的换行符
                # words = list(jieba.cut(text))  # 进行分词
                if is_abandon_stop_words:  # 去除停用词
                    words = [word for word in words if word not in char_to_be_replaced]
                    words = [word for word in words if word not in stop_words]
                corpus.append("".join(words))
                labels.append(file_name.split(".txt")[0])
    
    return corpus, labels



def extract_dataset(books, labels, K, num_paragraphs, is_words=True):
    """
        利用语料库生成段落
        从生成的段落中抽取数据集
    """
    # 计算每个标签需要抽取的段落数量
    num_paragraphs_per_label = num_paragraphs // len(set(labels))
    paragraphs_list = []  # 段落

    for index, (book, label) in enumerate(zip(books, labels)):
        # print(type(book))
        if is_words:
            words = list(jieba.cut(book))
            # print(words)
        else:
            words = list(book)

        for i in range(num_paragraphs_per_label):
            start = random.randint(0, len(words) - K)
            paragraph = words[start:start+K]
            paragraphs_list.append((index, paragraph))

    # 随机打乱列表中的元素顺序
    random.shuffle(paragraphs_list)
    train_set = paragraphs_list[:900]
    test_set = paragraphs_list[900:]
    
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    for data in train_set:
        train_data.append(data[1])
        train_labels.append(data[0])

    for data in test_set:
        test_data.append(data[1])
        test_labels.append(data[0])


    return train_data, train_labels, test_data, test_labels


def train(T, K, train_data, train_labels, test_data, test_labels):
    dictionary = corpora.Dictionary(train_data)  # 创建词典
    lda_corpus_train = [dictionary.doc2bow(tmp_doc) for tmp_doc in train_data]  # 将训练数据转换为词袋表示
    lda = models.LdaModel(corpus=lda_corpus_train, id2word=dictionary, num_topics=T)  # 训练LDA模型

    train_topic_distribution = lda.get_document_topics(lda_corpus_train)  # 获取训练集的主题分布
    train_features = np.zeros((len(train_data), T))  # 初始化训练特征向量
    for i in range(len(train_topic_distribution)):
        tmp_topic_distribution = train_topic_distribution[i]
        for j in range(len(tmp_topic_distribution)):
            train_features[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]  # 构建特征向量

    # print(len(train_labels))
    # print(len(train_features))

    assert len(train_labels) == len(train_features)  # 断言训练标签数量等于训练特征向量数量
    train_labels = np.array(train_labels)  # 将训练标签列表转换为NumPy数组
    classifier = SVC(kernel='linear', probability=True)  # 初始化SVM分类器
    classifier.fit(train_features, train_labels)  # 训练分类器
    train_acc = sum(classifier.predict(train_features) == train_labels) / len(train_labels)

    # print("训练样本的预测准确率为 {:.4f}.".format(sum(classifier.predict(train_features) == train_labels) / len(train_labels)))  # 打印训练样本的预测准确率

    lda_corpus_test = [dictionary.doc2bow(tmp_doc) for tmp_doc in test_data]
    test_topic_distribution = lda.get_document_topics(lda_corpus_test)
    test_features = np.zeros((len(test_data), T))
    for i in range(len(test_topic_distribution)):
        tmp_topic_distribution = test_topic_distribution[i]
        for j in range(len(tmp_topic_distribution)):
            test_features[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]
    assert len(test_labels) == len(test_features)
    test_labels = np.array(test_labels)
    # print("Prediction accuracy of testing samples is {:.4f}.".format(sum(classifier.predict(test_features) == test_labels) / len(test_labels)))

    test_acc = sum(classifier.predict(test_features) == test_labels) / len(test_labels)

    return train_acc, test_acc


def main():
    Ts = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]  # 主题数
    Ks = [20, 100, 500, 1000, 3000]  # 段落长度
    num_paragraphs = 1000  # 段落数

    # 中文语料库的实际路径
    folder_path = 'data'
    
    # 读取小说的文本内容，合并为一个语料库
    # 根据小说的名字分类
    books = []
    labels = []
    books, labels = dataset_process(data_path=folder_path, is_abandon_stop_words=True)
    
    
    train_acc_list = []
    test_acc_list = []


    # # 实验一   
    # K = 1000
    # train_data, train_labels, test_data, test_labels = extract_dataset(books=books, labels=labels, K=K, num_paragraphs=num_paragraphs, is_words=True)

    # for T in Ts:
    #     train_acc, test_acc = train(T=T, K=K, train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
    #     print(f'T={T}, K={K}, train_acc={train_acc}, test_acc={test_acc}')
    #     train_acc_list.append(train_acc)
    #     test_acc_list.append(test_acc)

    # plt.figure(figsize=(6, 6))
    # plt.plot(Ts, train_acc_list, color='green', label='train_acc')
    # plt.plot(Ts, test_acc_list, color='blue', label='test_acc')
    # plt.legend()
    # plt.xlabel('T', fontsize=14, fontweight='bold')
    # plt.ylabel('acc', fontsize=14, fontweight='bold')
    # plt.grid(True)
    # # plt.show()
    # plt.savefig('result1.png', dpi=300)  # dpi参数指定图像的分辨率，300为常用的分辨率
    # plt.close()

    # 实验二
    K = 1000
    T = 200
    train_data, train_labels, test_data, test_labels = extract_dataset(books=books, labels=labels, K=K, num_paragraphs=num_paragraphs, is_words=True)
    train_acc, test_acc = train(T=T, K=K, train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
    print(f'train_acc = {train_acc}, test_acc={test_acc}')

    train_data, train_labels, test_data, test_labels = extract_dataset(books=books, labels=labels, K=K, num_paragraphs=num_paragraphs, is_words=False)
    train_acc, test_acc = train(T=T, K=K, train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
    print(f'train_acc = {train_acc}, test_acc={test_acc}')




    # # 实验三
    # T = 200
    # for K in Ks:
    #     train_data, train_labels, test_data, test_labels = extract_dataset(books=books, labels=labels, K=K, num_paragraphs=num_paragraphs, is_words=True)
    #     train_acc, test_acc = train(T=T, K=K, train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
    #     print(f'T={T}, K={K}, train_acc={train_acc}, test_acc={test_acc}')
    #     train_acc_list.append(train_acc)
    #     test_acc_list.append(test_acc)

    # plt.figure(figsize=(6, 6))
    # plt.plot(Ks, train_acc_list, color='green', label='train_acc')
    # plt.plot(Ks, test_acc_list, color='blue', label='test_acc')
    # plt.legend()
    # plt.xlabel('K', fontsize=14, fontweight='bold')
    # plt.ylabel('acc', fontsize=14, fontweight='bold')
    # plt.grid(True)
    # # plt.show()
    # plt.savefig('result3.png', dpi=300)  # dpi参数指定图像的分辨率，300为常用的分辨率
    # plt.close()
    




if __name__ == '__main__':
    main()