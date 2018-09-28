# -*- coding=utf-8 -*-
import jieba
import jieba.analyse
import numpy as np
import wordcloud
import matplotlib.pyplot as plt
import  seaborn as sns
import pandas as pd
s = '请你把 10 根放在篮子里的香蕉分给 10 只猴子，每只猴要得到一根，最后篮子里还要留下一根香蕉，你能做到吗？ 不好意思地说，这道题是小学二年级的一道数学竞赛题，给你一些时间，思索一下，这道题是不是更有脑洞大开的味道。'
# pynlpir.open(encoding='UTF-8')
# segments = pynlpir.segment(s)
# print('这个是nlpir分词')
# for seg in segments:
#     print(seg[0], end=',')
# pynlpir.close()


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子进行分词
def seg_sentence(sentence):
    jieba.load_userdict('./user_dict.txt')
    sentence_seged = jieba.cut(sentence.strip(), cut_all=False)
    stopwords = stopwordslist('./stop_words.txt')  # 这里加载停用词的路径
    outstr = []
    for word in sentence_seged:
        if word.strip() not in stopwords:
            if word != '\t':
                outstr.append(word)
    return outstr


word_count = pd.Series(seg_sentence(s)).value_counts().sort_values(ascending=False)[0:20]
fig = plt.figure(figsize=(20, 12))
x = word_count.index.tolist()
y = word_count.values.tolist()
sns.barplot(x, y, palette="BuPu_r")
plt.title("词频top20")
plt.ylabel('count')
sns.despine(bottom=True)
plt.show()

