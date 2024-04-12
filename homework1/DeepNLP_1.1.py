import re
import matplotlib.pyplot as plt
from collections import Counter
import jieba
import os
import numpy as np

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 输入中文语料库的实际路径
folder_path = 'data'
corpus = ""
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
       with open(file_path, 'r', encoding='GB18030') as file:
           corpus += file.read()

# 读取停用词
stopwords_path = 'cn_stopwords.txt'
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = set([word.strip() for word in f.readlines()])

# 提取语料中的中文
text_processed = re.sub(r'[^\u4e00-\u9fa5]', '', corpus)
# 结巴分词
words = jieba.cut(text_processed)
# 计算汉字频率
word_freq = Counter(words)
# 根据频率排序
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
# 排名和频率
ranks = range(1, len(sorted_word_freq) + 1)
freqs = [freq for _, freq in sorted_word_freq]
# 重新生成 words 迭代器
words = jieba.cut(text_processed)
# 过滤停用词
filtered_words = [word for word in words if word not in stopwords]
filtered_words_freq = Counter(filtered_words)
sorted_filtered_words_freq = sorted(filtered_words_freq.items(), key=lambda x: x[1], reverse=True)
filtered_ranks = range(1, len(sorted_filtered_words_freq) + 1)
filtered_freqs = [freq for _, freq in sorted_filtered_words_freq]

# 选择未过滤停用词中的最高频率词语作为基准值
baseline_freq = sorted_word_freq[0][1]
baseline_ranks = range(1, len(sorted_word_freq) + 1)
baseline_freqs = [baseline_freq / rank for rank in baseline_ranks]
plt.figure(figsize=(6, 6))
plt.loglog(baseline_ranks, baseline_freqs, color='red', linestyle='--', label='基准值')
plt.loglog(ranks, freqs, color='blue', label='未过滤停用词')
plt.loglog(filtered_ranks, filtered_freqs, color='green', label='过滤停用词')
plt.legend()
plt.xlabel('词语排名', fontsize=14, fontweight='bold')
plt.ylabel('词频', fontsize=14, fontweight='bold')
plt.title('词频与词语排名的关系')
plt.grid(True)
# plt.show()
plt.savefig('Zip_Law.png', dpi=300)  # dpi参数指定图像的分辨率，300为常用的分辨率
plt.close()

