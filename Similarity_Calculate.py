#coding=utf-8
from xml.dom.minidom import parse
import xml.dom.minidom
import os
from pyltp import Segmentor,Postagger,NamedEntityRecognizer
import sys
import string
import math
from pygraph.classes.graph import graph
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import subprocess
from matplotlib.font_manager import FontManager
import random

result_list=[]
chineseWordVecs=dict()
print("read CHINESE_WORD2VEC_FILENAME")
word2vec = open("/media/lyt312323529/c4175817-9d97-490b-95c6-636149e75a87/Graph_Generate/chinese-wiki-20160305.word2vec", "r")
for line in word2vec:
    if line.count(" ") < 5:
        continue
    else:
        index = line.find(" ")
        curWord = line[:index]
        rest = line[index + 1:]
        tokens = rest.strip().split(" ")
        numTokens = []
        for tok in tokens:
            numTokens.append(float(tok))
        chineseWordVecs[curWord] = numTokens
word2vec.close()
print("load CHINESE_WORD2VEC_FILENAME successfully")

for i in range(30):
    result_news=[]
    f_i=open("/media/lyt312323529/c4175817-9d97-490b-95c6-636149e75a87/Graph_Generate/news_"+str(i+1)+"_result.txt", "r")
    lines_i=f_i.readlines()
    if lines_i[0] == "\n":
        title_ner_i = []
    else:
        title_ner_i=lines_i[0].strip().split(" ")
    if lines_i[1] == "\n":
        title_trigger_i = []
    else:
        title_trigger_i = lines_i[1].strip().split(" ")
    if lines_i[2] == "\n":
        title_argument_i = []
    else:
        title_argument_i = lines_i[2].strip().split(" ")
    if lines_i[3] == "\n":
        article_ner_i = []
    else:
        article_ner_i = lines_i[3].strip().split(" ")
    if lines_i[4] == "\n":
        article_trigger_i = []
    else:
        article_trigger_i = lines_i[4].strip().split(" ")
    if lines_i[5] == "\n":
        article_argument_i = []
    else:
        article_argument_i = lines_i[5].strip().split(" ")
    f_i.close()
    print "第("+str(i+1)+")篇新闻的处理结果加载完成"
    for j in range(30):
        result_i = []
        if j==i:
            print len(result_news)
            print str(i)+":"+str(j)
            result_news.append(100.100)
            continue
        else:
            f_j=open("/media/lyt312323529/c4175817-9d97-490b-95c6-636149e75a87/Graph_Generate/news_"+str(j+1)+"_result.txt", "r")
            lines_j=f_j.readlines()
            if lines_j[0]=="\n":
                title_ner_j=[]
            else:
                title_ner_j=lines_j[0].strip().split(" ")
            if lines_j[1]=="\n":
                title_trigger_j=[]
            else:
                title_trigger_j=lines_j[1].strip().split(" ")
            if lines_j[2]=="\n":
                title_argument_j=[]
            else:
                title_argument_j=lines_j[2].strip().split(" ")
            if lines_j[3]=="\n":
                article_ner_j=[]
            else:
                article_ner_j=lines_j[3].strip().split(" ")
            if lines_j[4]=="\n":
                article_trigger_j=[]
            else:
                article_trigger_j=lines_j[4].strip().split(" ")
            if lines_j[5]=="\n":
                article_argument_j=[]
            else:
                article_argument_j=lines_j[5].strip().split(" ")
            f_j.close()
            print "\n相对于第("+str(i+1)+")篇新闻的第("+str(j+1)+")篇新闻的处理结果加载完成"


            if len(title_ner_i)>0 and len(title_ner_j)>0:
                title_ner_words_similarity = []
                for k in range(len(title_ner_i)):
                    max_cos = 0.0
                    if chineseWordVecs.has_key(title_ner_i[k]):
                        word2vec_i=chineseWordVecs[title_ner_i[k]]
                    else:
                        word2vec_i = []
                        for m in range(100):
                            word2vec_i.append(random.uniform(-2, 6))
                    if title_ner_i[k] in title_ner_j:
                        title_ner_words_similarity.append(5)
                        print "("+title_ner_i[k]+")同时存在于两个标题名实体列表中,该词的余弦相似度直接设置为5"
                    else:
                        for l in range(len(title_ner_j)):
                            if chineseWordVecs.has_key(title_ner_j[l]):
                                word2vec_j=chineseWordVecs[title_ner_j[l]]
                            else:
                                word2vec_j=[]
                                for m in range(100):
                                    word2vec_j.append(random.uniform(-2,6))
                            mn = 0
                            mm = 0
                            nn = 0
                            cos_value = 0.0
                            for o in range(100):
                                mn += word2vec_i[o] * word2vec_j[o]
                                mm += word2vec_i[o] * word2vec_i[o]
                                nn += word2vec_j[o] * word2vec_j[o]
                            cos_value = mn / (math.sqrt(mm) * math.sqrt(nn))
                            print "词语("+title_ner_i[k]+")与词语("+title_ner_j[l]+")的余弦相似度为:"+str(cos_value)
                            if cos_value>max_cos:
                                print "新闻("+str(i+1)+")的标题名实体("+title_ner_i[k]+")相对于新闻("+str(j+1)+")的标题名实体最大余弦值更新为:"+str(cos_value)
                                max_cos=cos_value
                            else:
                                print "新闻("+str(i+1)+")的标题名实体("+title_ner_i[k]+")相对于新闻("+str(j+1)+")的标题名实体最大余弦值保持不变,为:"+str(max_cos)
                        title_ner_words_similarity.append(max_cos)
                        print "新闻("+str(i+1)+")的标题名实体("+title_ner_i[k]+")相对于新闻("+str(j+1)+")的标题名实体最大余弦值为:"+str(max_cos)
                title_ner_words_cos=0
                for p in range(len(title_ner_words_similarity)):
                    title_ner_words_cos=title_ner_words_cos+title_ner_words_similarity[p]
                print "新闻("+str(i+1)+")的标题名实体与新闻("+str(j+1)+")的标题名实体的相似度为"+str(title_ner_words_cos/len(title_ner_words_similarity))
                result_i.append(title_ner_words_cos/len(title_ner_words_similarity))
            elif len(title_ner_i)==0 or len(title_ner_j)==0:
                print "\n新闻("+str(i+1)+")和新闻("+str(j+1)+")均不具备标题名实体"

            if len(title_trigger_i) > 0 and len(title_trigger_j) > 0:
                title_trigger_words_similarity = []
                for k in range(len(title_trigger_i)):
                    max_cos = 0.0
                    if chineseWordVecs.has_key(title_trigger_i[k]):
                        word2vec_i = chineseWordVecs[title_trigger_i[k]]
                    else:
                        word2vec_i = []
                        for m in range(100):
                            word2vec_i.append(random.uniform(-2, 6))
                    if title_trigger_i[k] in title_trigger_j:
                        title_trigger_words_similarity.append(5)
                        print "(" + title_trigger_i[k] + ")同时存在于两个标题触发词列表中,该词的余弦相似度直接设置为5"
                    else:
                        for l in range(len(title_trigger_j)):
                            if chineseWordVecs.has_key(title_trigger_j[l]):
                                word2vec_j = chineseWordVecs[title_trigger_j[l]]
                            else:
                                word2vec_j = []
                                for m in range(100):
                                    word2vec_j.append(random.uniform(-2, 6))
                            mn = 0
                            mm = 0
                            nn = 0
                            cos_value = 0.0
                            for o in range(100):
                                mn += word2vec_i[o] * word2vec_j[o]
                                mm += word2vec_i[o] * word2vec_i[o]
                                nn += word2vec_j[o] * word2vec_j[o]
                            cos_value = mn / (math.sqrt(mm) * math.sqrt(nn))
                            print "词语(" + title_trigger_i[k] + ")与词语(" + title_trigger_j[l] + ")的余弦相似度为:" + str(cos_value)
                            if cos_value > max_cos:
                                print "新闻(" + str(i + 1) + ")的标题触发词(" + title_trigger_i[k] + ")相对于新闻(" + str(j + 1) + ")的标题触发词最大余弦值更新为:" + str(cos_value)
                                max_cos = cos_value
                            else:
                                print "新闻(" + str(i + 1) + ")的标题触发词(" + title_trigger_i[k] + ")相对于新闻(" + str(j + 1) + ")的标题触发词最大余弦值保持不变,为:" + str(max_cos)
                        title_trigger_words_similarity.append(max_cos)
                        print "新闻(" + str(i + 1) + ")的标题触发词(" + title_trigger_i[k] + ")相对于新闻(" + str(j + 1) + ")的标题触发词最大余弦值为:" + str(max_cos)
                title_trigger_words_cos = 0
                for p in range(len(title_trigger_words_similarity)):
                    title_trigger_words_cos = title_trigger_words_cos + title_trigger_words_similarity[p]
                print "新闻(" + str(i + 1) + ")的标题触发词与新闻(" + str(j + 1) + ")的标题触发词的相似度为" + str(title_trigger_words_cos / len(title_trigger_words_similarity))
                result_i.append(title_trigger_words_cos / (len(title_trigger_i)*len(title_trigger_j)))
            elif len(title_trigger_i) == 0 or len(title_trigger_j) == 0:
                print "\n新闻(" + str(i + 1) + ")或新闻(" + str(j + 1) + ")均不具备标题触发词"

            if len(title_argument_i) > 0 and len(title_argument_j) > 0:
                title_argument_words_similarity = []
                for k in range(len(title_argument_i)):
                    max_cos = 0.0
                    if chineseWordVecs.has_key(title_argument_i[k]):
                        word2vec_i = chineseWordVecs[title_argument_i[k]]
                    else:
                        word2vec_i = []
                        for m in range(100):
                            word2vec_i.append(random.uniform(-2, 6))
                    if title_argument_i[k] in title_argument_j:
                        title_argument_words_similarity.append(5)
                        print "(" + title_argument_i[k] + ")同时存在于两个标题事件元素列表中,该词的余弦相似度直接设置为5"
                    else:
                        for l in range(len(title_argument_j)):
                            if chineseWordVecs.has_key(title_argument_j[l]):
                                word2vec_j = chineseWordVecs[title_argument_j[l]]
                            else:
                                word2vec_j = []
                                for m in range(100):
                                    word2vec_j.append(random.uniform(-2, 6))
                            mn = 0
                            mm = 0
                            nn = 0
                            cos_value = 0.0
                            for o in range(100):
                                mn += word2vec_i[o] * word2vec_j[o]
                                mm += word2vec_i[o] * word2vec_i[o]
                                nn += word2vec_j[o] * word2vec_j[o]
                            cos_value = mn / (math.sqrt(mm) * math.sqrt(nn))
                            print "词语(" + title_argument_i[k] + ")与词语(" + title_argument_j[l] + ")的余弦相似度为:" + str(cos_value)
                            if cos_value > max_cos:
                                print "新闻(" + str(i + 1) + ")的标题事件元素(" + title_argument_i[k] + ")相对于新闻(" + str(j + 1) + ")的标题事件元素最大余弦值更新为:" + str(cos_value)
                                max_cos = cos_value
                            else:
                                print "新闻(" + str(i + 1) + ")的标题事件元素(" + title_argument_i[k] + ")相对于新闻(" + str(j + 1) + ")的标题事件元素最大余弦值保持不变,为:" + str(max_cos)
                        title_argument_words_similarity.append(max_cos)
                        print "新闻(" + str(i + 1) + ")的标题事件元素(" + title_argument_i[k] + ")相对于新闻(" + str(j + 1) + ")的标题事件元素最大余弦值为:" + str(max_cos)
                title_argument_words_cos = 0
                for p in range(len(title_argument_words_similarity)):
                    title_argument_words_cos = title_argument_words_cos + title_argument_words_similarity[p]
                print "新闻(" + str(i + 1) + ")的标题事件元素与新闻(" + str(j + 1) + ")的标题事件元素的相似度为" + str(title_argument_words_cos / len(title_argument_words_similarity))
                result_i.append(title_argument_words_cos / len(title_argument_words_similarity))
            elif len(title_argument_i) == 0 or len(title_argument_j) == 0:
                print "\n新闻(" + str(i + 1) + ")和新闻(" + str(j + 1) + ")均不具备标题事件元素"

            if len(article_ner_i) > 0 and len(article_ner_j) > 0:
                article_ner_words_similarity = []
                for k in range(len(article_ner_i)):
                    max_cos = 0.0
                    if chineseWordVecs.has_key(article_ner_i[k]):
                        word2vec_i = chineseWordVecs[article_ner_i[k]]
                    else:
                        word2vec_i = []
                        for m in range(100):
                            word2vec_i.append(random.uniform(-2, 6))
                    if article_ner_i[k] in article_ner_j:
                        article_ner_words_similarity.append(5)
                        print "(" + article_ner_i[k] + ")同时存在于两个新闻名实体列表中,该词的余弦相似度直接设置为5"
                    else:
                        for l in range(len(article_ner_j)):
                            if chineseWordVecs.has_key(article_ner_j[l]):
                                word2vec_j = chineseWordVecs[article_ner_j[l]]
                            else:
                                word2vec_j = []
                                for m in range(100):
                                    word2vec_j.append(random.uniform(-2, 6))
                            mn = 0
                            mm = 0
                            nn = 0
                            cos_value = 0.0
                            for o in range(100):
                                mn += word2vec_i[o] * word2vec_j[o]
                                mm += word2vec_i[o] * word2vec_i[o]
                                nn += word2vec_j[o] * word2vec_j[o]
                            cos_value = mn / (math.sqrt(mm) * math.sqrt(nn))
                            print "词语(" + article_ner_i[k] + ")与词语(" + article_ner_j[l] + ")的余弦相似度为:" + str(cos_value)
                            if cos_value > max_cos:
                                print "新闻(" + str(i + 1) + ")的新闻名实体(" + article_ner_i[k] + ")相对于新闻(" + str(j + 1) + ")的新闻名实体最大余弦值更新为:" + str(cos_value)
                                max_cos = cos_value
                            else:
                                print "新闻(" + str(i + 1) + ")的新闻名实体(" + article_ner_i[k] + ")相对于新闻(" + str(j + 1) + ")的新闻名实体最大余弦值保持不变,为:" + str(max_cos)
                        article_ner_words_similarity.append(max_cos)
                        print "新闻(" + str(i + 1) + ")的新闻名实体(" + article_ner_i[k] + ")相对于新闻(" + str(j + 1) + ")的新闻名实体最大余弦值为:" + str(max_cos)
                article_ner_words_cos = 0
                for p in range(len(article_ner_words_similarity)):
                    article_ner_words_cos = article_ner_words_cos + article_ner_words_similarity[p]
                print "新闻(" + str(i + 1) + ")的新闻名实体与新闻(" + str(j + 1) + ")的新闻名实体的相似度为" + str(article_ner_words_cos / len(article_ner_words_similarity))
                result_i.append(article_ner_words_cos / len(article_ner_words_similarity))
            elif len(article_ner_i) == 0 or len(article_ner_j) == 0:
                print "\n新闻(" + str(i + 1) + ")或新闻(" + str(j + 1) + ")均不具备新闻名实体"

            if len(article_trigger_i) > 0 and len(article_trigger_j) > 0:
                article_trigger_words_similarity = []
                for k in range(len(article_trigger_i)):
                    max_cos = 0.0
                    if chineseWordVecs.has_key(article_trigger_i[k]):
                        word2vec_i = chineseWordVecs[article_trigger_i[k]]
                    else:
                        word2vec_i = []
                        for m in range(100):
                            word2vec_i.append(random.uniform(-2, 6))
                    if article_trigger_i[k] in article_trigger_j:
                        article_trigger_words_similarity.append(5)
                        print "(" + article_trigger_i[k] + ")同时存在于两个新闻触发词列表中,该词的余弦相似度直接设置为5"
                    else:
                        for l in range(len(article_trigger_j)):
                            if chineseWordVecs.has_key(article_trigger_j[l]):
                                word2vec_j = chineseWordVecs[article_trigger_j[l]]
                            else:
                                word2vec_j = []
                                for m in range(100):
                                    word2vec_j.append(random.uniform(-2, 6))
                            mn = 0
                            mm = 0
                            nn = 0
                            cos_value = 0.0
                            for o in range(100):
                                mn += word2vec_i[o] * word2vec_j[o]
                                mm += word2vec_i[o] * word2vec_i[o]
                                nn += word2vec_j[o] * word2vec_j[o]
                            cos_value = mn / (math.sqrt(mm) * math.sqrt(nn))
                            print "词语(" + article_trigger_i[k] + ")与词语(" + article_trigger_j[l] + ")的余弦相似度为:" + str(cos_value)
                            if cos_value > max_cos:
                                print "新闻(" + str(i + 1) + ")的新闻触发词(" + article_trigger_i[k] + ")相对于新闻(" + str(j + 1) + ")的新闻触发词最大余弦值更新为:" + str(cos_value)
                                max_cos = cos_value
                            else:
                                print "新闻(" + str(i + 1) + ")的新闻触发词(" + article_trigger_i[k] + ")相对于新闻(" + str(j + 1) + ")的新闻触发词最大余弦值保持不变,为:" + str(max_cos)
                        article_trigger_words_similarity.append(max_cos)
                        print "新闻(" + str(i + 1) + ")的新闻触发词(" + article_trigger_i[k] + ")相对于新闻(" + str(j + 1) + ")的新闻触发词最大余弦值为:" + str(max_cos)
                article_trigger_words_cos = 0
                for p in range(len(article_trigger_words_similarity)):
                    article_trigger_words_cos = article_trigger_words_cos + article_trigger_words_similarity[p]
                print "新闻(" + str(i + 1) + ")的新闻触发词与新闻(" + str(j + 1) + ")的新闻触发词的相似度为" + str(article_trigger_words_cos / len(article_trigger_words_similarity))
                result_i.append(article_trigger_words_cos / (len(article_trigger_i)*len(article_trigger_j)))
            elif len(article_trigger_i) == 0 or len(article_trigger_j) == 0:
                print "\n新闻(" + str(i + 1) + ")和新闻(" + str(j + 1) + ")均不具备新闻触发词"

            if len(article_argument_i) > 0 and len(article_argument_j) > 0:
                article_argument_words_similarity = []
                for k in range(len(article_argument_i)):
                    max_cos = 0.0
                    if chineseWordVecs.has_key(article_argument_i[k]):
                        word2vec_i = chineseWordVecs[article_argument_i[k]]
                    else:
                        word2vec_i = []
                        for m in range(100):
                            word2vec_i.append(random.uniform(-2, 6))
                    if article_argument_i[k] in article_argument_j:
                        article_argument_words_similarity.append(5)
                        print "(" + article_argument_i[k] + ")同时存在于两个新闻事件元素列表中,该词的余弦相似度直接设置为5"
                    else:
                        for l in range(len(article_argument_j)):
                            if chineseWordVecs.has_key(article_argument_j[l]):
                                word2vec_j = chineseWordVecs[article_argument_j[l]]
                            else:
                                word2vec_j = []
                                for m in range(100):
                                    word2vec_j.append(random.uniform(-2, 6))
                            mn = 0
                            mm = 0
                            nn = 0
                            cos_value = 0.0
                            for o in range(100):
                                mn += word2vec_i[o] * word2vec_j[o]
                                mm += word2vec_i[o] * word2vec_i[o]
                                nn += word2vec_j[o] * word2vec_j[o]
                            cos_value = mn / (math.sqrt(mm) * math.sqrt(nn))
                            print "词语(" + article_argument_i[k] + ")与词语(" + article_argument_j[l] + ")的余弦相似度为:" + str(cos_value)
                            if cos_value > max_cos:
                                print "新闻(" + str(i + 1) + ")的新闻事件元素(" + article_argument_i[k] + ")相对于新闻(" + str(j + 1) + ")的新闻事件元素最大余弦值更新为:" + str(cos_value)
                                max_cos = cos_value
                            else:
                                print "新闻(" + str(i + 1) + ")的新闻事件元素(" + article_argument_i[k] + ")相对于新闻(" + str(j + 1) + ")的新闻事件元素最大余弦值保持不变,为:" + str(max_cos)
                        article_argument_words_similarity.append(max_cos)
                        print "新闻(" + str(i + 1) + ")的新闻事件元素(" + article_argument_i[k] + ")相对于新闻(" + str(j + 1) + ")的新闻事件元素最大余弦值为:" + str(max_cos)
                article_argument_words_cos = 0
                for p in range(len(article_argument_words_similarity)):
                    article_argument_words_cos = article_argument_words_cos + article_argument_words_similarity[p]
                print "新闻(" + str(i + 1) + ")的新闻事件元素与新闻(" + str(j + 1) + ")的新闻事件元素的相似度为" + str(article_argument_words_cos / len(article_argument_words_similarity))
                result_i.append(article_argument_words_cos / len(article_argument_words_similarity))
            elif len(article_argument_i) == 0 or len(article_argument_j) == 0:
                print "\n新闻(" + str(i + 1) + ")和新闻(" + str(j + 1) + ")均不具备新闻事件元素"
            sum_i=0
            for s in range(len(result_i)):
                sum_i=sum_i+result_i[s]
            result_news.append(sum_i/len(result_i))
    result_list.append(result_news)
for i in range(len(result_list)):
    print("\n")
    for j in range(len(result_list[i])):
        print str(i+1)+"--"+str(j+1)+":"+str(result_list[i][j])