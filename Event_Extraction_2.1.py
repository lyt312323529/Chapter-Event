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

#计算联通图的PR值的类
class PRIterator:
    _doc_=''''z计算一张图的PR值'''
    #类的初始化函数
    def __init__(self,dg):
        self.damping_factor=0.85#阻尼系数
        self.max_iterations=100#最大迭代次数
        self.min_delta=0.00001#确定迭代是否结束的参数，即结果精度
        self.graph=dg
    #计算联通图PR值的函数
    def page_rank(self):
        #将图中没有出链的节点改为对所有节点有出链
        for node in self.graph.nodes():
            if len(self.graph.neighbors(node))==0:
                for node2 in self.graph.nodes():
                    graph.add_edge(self.graph,(node,node2))
        nodes=self.graph.nodes()
        graph_size=len(nodes)
        if graph_size==0:
            return{}
        page_rank=dict.fromkeys(nodes,1.0/graph_size)#给每一个顶点赋予初始的PR值
        damping_value=(1.0-self.damping_factor)/graph_size#公式中的(1-a)/N部分
        flag=False
        for i in range(self.max_iterations):
            change=0
            for node in nodes:
                rank=0
                for incident_page in self.graph.neighbors(node):
                    rank+=self.damping_factor*(page_rank[incident_page]/len(self.graph.neighbors(incident_page)))
                rank+=damping_value
                change+=abs(page_rank[node]-rank)#绝对值
                page_rank[node]=rank
            print("This is No.%s iteration"%(i+1))
            print(page_rank)
            if change<self.min_delta:
                flag=True
                break
        if flag:
            print("finished in %s iterations!"%node)
        else:
            print("finished out of 100 iterations!")
        return page_rank

chineseWordVecs = dict()
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

MODELDIR = "/media/lyt312323529/c4175817-9d97-490b-95c6-636149e75a87/Graph_Generate/ltp_data"
print("正在加载LTP模型...")
segmentor = Segmentor()
p = os.path.join(MODELDIR, "cws.model")
segmentor.load(p)
postagger = Postagger()
postagger.load(os.path.join(MODELDIR, "pos.model"))
recognizer = NamedEntityRecognizer()
recognizer.load(os.path.join(MODELDIR, "ner.model"))
print("加载完毕")

for news_index_ in range(30):
    news_index = news_index_+1

    DOMTree = xml.dom.minidom.parse("relation" + str(news_index) + ".xml")
    doc = DOMTree.documentElement

    events_code_list = []
    events_word2vec_list = []
    relations_list = []
    words_list = []
    page_ranks_list = []
    ner_list = []
    sentence_list = []
    sentences = doc.getElementsByTagName("sentence")
    for sentence in sentences:
        print ("\nThe id of sentence is:" + sentence.getAttribute("id"))
        origin_texts = sentence.getElementsByTagName("origin_text")
        for origin_text in origin_texts:
            print("这句话为:" + origin_text.firstChild.data.encode("utf-8"))
            sentence_list.append(origin_text.firstChild.data.encode("utf-8"))
        relations = sentence.getElementsByTagName("relation")
        print("提取出来的关系对数目为：" + str(len(relations)))
        for relation in relations:
            print("\n提取relation")
            arguments = relation.getElementsByTagName("argument")
            arguments_list = []
            trigger_list = []
            trigger_list.append(relation.getAttribute("pred").encode("utf-8").replace('(','').replace('z','').replace('Z',')').replace(')',''))
            print("加入的关系元素为(" + relation.getAttribute("pred").replace('(','').replace('z','').replace('Z',')').replace(')','').encode("utf-8") + ")")
            for argument in arguments:
                arguments_list.append(argument.getAttribute("content").replace('(','').replace('z','').replace('Z',')').replace(')','').encode("utf-8"))
                print("加入的元素为(" + argument.getAttribute("content").replace('(','').replace('z','').replace('Z',')').replace(')','').encode("utf-8") + ")")
            relations_item = []
            relations_item.append(trigger_list)
            relations_item.append(arguments_list)
            relations_item.append(len(sentence_list) - 1)
            print("关系对对应的句子id为:" + str(relations_item[2]))
            relations_list.append(relations_item)
    print ("一共提取到(" + str(len(relations_list)) + ")组事件对")

    print "\n处理新闻标题"
    f = open("/media/lyt312323529/c4175817-9d97-490b-95c6-636149e75a87/Graph_Generate/title.txt", "r")
    lines = f.readlines()
    title_trigger = []
    title_ner = []
    title_argument = []
    line = lines[news_index - 1]
    words = segmentor.segment(line)
    postags = postagger.postag(words)
    for i in range(len(postags)):
        pt = postags[i][0]
        if pt == "v":
            if (not (words[i] in title_trigger)) and (len(words[i])>3):
                print "词语(" + words[i] + ")词性为" + pt + "为触发词，放入标题触发词列表中"
                title_trigger.append(words[i])
            else:
                print "词语(" + words[i] + ")词性为" + pt + "为触发词，已经存在于标题触发词列表中,不需重复放入"
        elif (pt == "b") or (pt == "j") or (pt == "n"):
            if (pt == "n") and (len(postags[i]) > 1):
                if (postags[i][1] != "t") and (postags[i][1] != "d"):
                    if not (words[i] in title_argument):
                        print("词语(" + words[i] + ")的标签为(" + pt + "),符合词性要求，放入标题事件元素列表")
                        title_argument.append(words[i])
                    else:
                        print("词语(" + words[i] + ")的标签为(" + pt + "),符合词性要求，已经存在于标题事件元素列表中，不重复放入")
                else:
                    print("词语(" + words[i] + ")的标签为(" + pt + postags[i][1] + "),不符合词性要求，丢弃")
            else:
                if not (words[i] in title_argument):
                    print("词语(" + words[i] + ")的标签为(" + pt + "),符合词性要求，放入标题事件元素列表")
                    title_argument.append(words[i])
                else:
                    print("词语(" + words[i] + ")的标签为(" + pt + "),符合词性要求，已经存在于标题事件元素列表中，不重复放入")
        else:
            print("词语(" + words[i] + ")的标签为(" + pt + "),不符合词性要求，丢弃")
    if len(words) > 0:
        netags = recognizer.recognize(words, postags)
        print("\t".join(netags))
        wordStr = ""
        for k in range(len(words)):
            # print(postags[k][0])
            if (netags[k][0] == "B") and len(argument_words) > 1:
                l = k
                while True:
                    wordStr += argument_words[l]
                    if ((netags[l][0] == "E") or (l == len(argument_words) - 1)):
                        break
                    else:
                        l += 1
                if not (wordStr in title_ner):
                    title_ner.append(wordStr)
                    print("名实体(" + wordStr + ")加入名实体列表")
                wordStr = ""
            elif (netags[k][0] == "I") or (netags[k][0] == "E"):
                continue
            else:
                if netags[k][0] == "S":
                    if not (words[k] in title_ner):
                        title_ner.append(words[k])
                        print("名实体(" + words[k] + ")加入名实体列表")

    f.close()

    events_list = []
    for i in range(len(relations_list)):
        event = []
        trigger = []
        print("\n\n处理触发词结果如下")
        if (relations_list[i][0][0] != "is") and (relations_list[i][0][0] != "de"):
            words = segmentor.segment(relations_list[i][0][0])
            # wordStr = "\t".join(words)
            # print(wordStr)
            postags = postagger.postag(words)
            # print("\t".join(postags))
            for k in range(len(words)):
                pt = postags[k][0]
                # if (pt=="a")or(pt=="b")or(pt=="i")or(pt=="j")or(pt=="m")or(pt=="n")or(pt=="q")or(pt=="r")or(pt=="v"):
                if pt == "v":
                    print("词语(" + words[k] + ")的标签为(" + pt + "),符合词性要求，放入触发词列表")
                    trigger.append(words[k])
                else:
                    print("词语(" + words[k] + ")的标签为(" + pt + "),不符合词性要求，丢弃")
            if len(trigger) > 0:
                print("触发词有效，放入事件中")
                event.append(trigger)
                print("\n处理事件元素结果如下")
                arguments_item = []
                for l in range(len(relations_list[i][1])):
                    words = segmentor.segment(relations_list[i][1][l])
                    # wordStr = "\t".join(words)
                    # print(wordStr)
                    postags = postagger.postag(words)
                    # print("\t".join(postags))
                    argument_item = []
                    argument_words = []
                    argument_postags = []
                    for m in range(len(words)):
                        pt = postags[m][0]
                        if (pt == "b") or (pt == "j") or (pt == "n"):
                            if (pt == "n") and (len(postags[m]) > 1):
                                if (postags[m][1] != "t") and (postags[m][1] != "d"):
                                    print("词语(" + words[m] + ")的标签为(" + pt + postags[m][1] + "),符合词性要求，放入事件词汇列表")
                                    argument_words.append(words[m])
                                    argument_postags.append(postags[m])
                                else:
                                    print("词语(" + words[m] + ")的标签为(" + pt + postags[m][1] + "),不符合词性要求，丢弃")
                            else:
                                print("词语(" + words[m] + ")的标签为(" + pt + "),符合词性要求，放入事件词汇列表")
                                argument_words.append(words[m])
                                argument_postags.append(postags[m])
                        else:
                            print("词语(" + words[m] + ")的标签为(" + pt + "),不符合词性要求，丢弃")
                    print("词表长度为" + str(len(words)))
                    if len(argument_words) > 0:
                        netags = recognizer.recognize(argument_words, argument_postags)
                        print("\t".join(netags))
                        wordStr = ""
                        for k in range(len(argument_words)):
                            # print(postags[k][0])
                            if (netags[k][0] == "B") and len(argument_words) > 1:
                                l = k
                                while True:
                                    wordStr += argument_words[l]
                                    if ((netags[l][0] == "E") or (l == len(argument_words) - 1)):
                                        break
                                    else:
                                        l += 1
                                print("识别出名实体(" + wordStr + ")并加入事件")
                                arguments_item.append(wordStr)
                                if not (wordStr in ner_list):
                                    ner_list.append(wordStr)
                                    print("名实体(" + wordStr + ")加入名实体列表")
                                wordStr = ""
                            elif (netags[k][0] == "I") or (netags[k][0] == "E"):
                                continue
                            else:
                                arguments_item.append(argument_words[k])
                                print("词语(" + arguments_item[len(arguments_item) - 1] + ")放入事件")
                                if netags[k][0] == "S":
                                    if not (argument_words[k] in ner_list):
                                        ner_list.append(argument_words[k])
                                        print("名实体(" + argument_words[k] + ")加入名实体列表")
                if len(arguments_item) > 0:
                    event.append(arguments_item)
                    event.append(relations_list[i][2])
                    event.append(i)
                    events_list.append(event)
                    print(
                    "事件元素列表不为空，将事件(" + str(len(events_list) - 1) + ")放入事件列表中,该事件对应的句子id为:" + str(relations_list[i][2]))
                else:
                    print("事件元素列表为空,不符合事件条件，丢弃事件")
            else:
                print("触发词无效，事件丢弃")
    print("共提取(" + str(len(events_list)) + ")个事件")
    # print str(len(events_list))
    # print str(len(events_list[0]))
    # print str(len(events_list[0][0]))
    # print str(len(events_list[0][0][0]))
    # print events_list[0][0][0][0]
    # print events_list[0][1][0][0]

    print("\n生成单词列表")
    words_sentence_id_list = []
    arguments_list = []
    triggers_list = []
    for i in range(len(events_list)):
        for j in range(len(events_list[i][0])):
            word_exist_state = False
            for k in range(len(words_list)):
                if words_list[k] == events_list[i][0][j]:
                    sentence_exist_state = False
                    # print "k:"+str(k)+"  len(words_list):"+str(len(words_list))+"  len(words_sentence_id_list):"+str(len(words_sentence_id_list))
                    for n in range(len(words_sentence_id_list[k])):
                        if words_sentence_id_list[k][n] == events_list[i][2]:
                            sentence_exist_state = True
                            print "\n句子(" + str(words_sentence_id_list[k][n]) + ")已经存在,不重复加入"
                    if not sentence_exist_state:
                        words_sentence_id_list[k].append(events_list[i][2])
                        print "\n句子(" + str(events_list[i][2]) + ")不存在于列表中,加入列表"
                    word_exist_state = True
            if word_exist_state:
                print("事件(" + str(i) + ")中事件触发词(" + events_list[i][0][j] + ")已经存在于单词列表中,不重复加入,对应的句子id为:(" + str(
                    events_list[i][2]) + ")")
            else:
                word_sentence_id = []
                words_list.append(events_list[i][0][j])
                word_sentence_id.append(events_list[i][2])
                words_sentence_id_list.append(word_sentence_id)
                print("\n事件(" + str(i) + ")中将事件触发词(" + events_list[i][0][j] + ")加入单词列表,对应的句子id为:(" + str(
                    events_list[i][2]) + ")")
            trigger_exist_state = False
            for a in range(len(triggers_list)):
                if triggers_list[a] == events_list[i][0][j]:
                    trigger_exist_state = True
                    print "触发词(" + triggers_list[a] + ")已经存在于触发词列表中，不需要重复加入"
            if not trigger_exist_state:
                triggers_list.append(events_list[i][0][j])
                print "触发词(" + events_list[i][0][j] + ")不存在于触发词列表中，加入触发词列表"
        for l in range(len(events_list[i][1])):
            word_exist_state = False
            for m in range(len(words_list)):
                if words_list[m] == events_list[i][1][l]:
                    sentence_exist_state = False
                    for o in range(len(words_sentence_id_list[m])):
                        if words_sentence_id_list[m][o] == events_list[i][2]:
                            sentence_exist_state = True
                            print "\n句子(" + str(events_list[i][2]) + ")已经存在,不重复加入"
                    if not sentence_exist_state:
                        words_sentence_id_list[m].append(events_list[i][2])
                        print "\n句子(" + str(events_list[i][2]) + ")不存在于列表中,加入列表"
                    word_exist_state = True
            if word_exist_state:
                print("事件(" + str(i) + ")中事件元素(" + events_list[i][1][l] + ")已经存在于单词列表中,不重复加入,对应的句子id为:(" + str(
                    events_list[i][2]) + ")")
            else:
                word_sentence_id = []
                word_sentence_id.append(events_list[i][2])
                words_list.append(events_list[i][1][l])
                words_sentence_id_list.append(word_sentence_id)
                print("\n事件(" + str(i) + ")中将事件元素(" + events_list[i][1][l] + ")加入单词列表,对应的句子id为:(" + str(
                    events_list[i][2]) + ")")
            argument_exist_state = False
            for b in range(len(arguments_list)):
                if arguments_list[b] == events_list[i][1][l]:
                    argument_exist_state = True
                    print("事件元素(" + arguments_list[b] + ")已经存在于事件元素列表中，不重复加入")
            if not argument_exist_state:
                arguments_list.append(events_list[i][1][l])
                print "事件元素(" + events_list[i][1][l] + ")不存在于事件元素列表中，加入事件元素列表"

    print("\n生成联通图,将节点插入联通图")
    # dg=digraph()
    dg = graph()
    nodes_list = range(len(words_list))
    dg.add_nodes(nodes_list)

    print("\n将同一事件中的节点连接起来")
    for i in range(len(events_list)):
        print("\n将事件(" + str(i) + ")中的关系词串联起来")
        if len(events_list[i][0]) > 1:
            for j in range(len(events_list[i][0]) - 1):
                relation1_index = -1
                relation2_index = -1
                for k in range(len(words_list)):
                    if words_list[k] == events_list[i][0][j]:
                        relation1_index = k
                for l in range(len(words_list)):
                    if words_list[l] == events_list[i][0][j + 1]:
                        relation2_index = l
                if (not dg.has_edge((relation1_index, relation2_index))):
                    dg.add_edge((relation1_index, relation2_index))
                    print("事件(" + str(i) + ")中的关系词(" + str(events_list[i][0][j]) + ")与关系词(" + str(
                        events_list[i][0][j + 1]) + ")建立一条边")
                else:
                    print("事件(" + str(i) + ")中的关系词(" + str(events_list[i][0][j]) + ")与关系词(" + str(
                        events_list[i][0][j + 1]) + ")已经存在一条边，不需要重复建立")
        else:
            print("事件(" + str(i) + ")的关系词个数为(" + str(len(events_list[i][0])) + ")，不需要进行关系词的连接")

        print("将事件(" + str(i) + ")中的事件元素串联起来")
        if len(events_list[i][1]) > 1:
            for j in range(len(events_list[i][1]) - 1):
                relation1_index = -1
                relation2_index = -1
                for k in range(len(words_list)):
                    if words_list[k] == events_list[i][1][j]:
                        relation1_index = k
                for l in range(len(words_list)):
                    if words_list[l] == events_list[i][1][j + 1]:
                        relation2_index = l
                if (not dg.has_edge((relation1_index, relation2_index))):
                    dg.add_edge((relation1_index, relation2_index))
                    print("事件(" + str(i) + ")中的事件元素(" + str(events_list[i][1][j]) + ")与事件元素(" + str(
                        events_list[i][1][j + 1]) + ")建立一条边")
                else:
                    print("事件(" + str(i) + ")中的事件元素(" + str(events_list[i][1][j]) + ")与事件元素(" + str(
                        events_list[i][1][j + 1]) + ")已经存在一条边，不需要重复建立")
        else:
            print("事件(" + str(i) + ")的事件元素个数为(" + str(len(events_list[i][0])) + ")，不需要进行事件元素的连接")

        print("将事件(" + str(i) + ")的关系词和事件元素连接起来")
        for j in range(len(events_list[i][0])):
            for k in range(len(events_list[i][1])):
                relation1_index = -1
                relation2_index = -1
                for l in range(len(words_list)):
                    if words_list[l] == events_list[i][0][j]:
                        relation1_index = l
                    if words_list[l] == events_list[i][1][k]:
                        relation2_index = l
                if (not dg.has_edge((relation1_index, relation2_index))):
                    dg.add_edge((relation1_index, relation2_index))
                    print("事件(" + str(i) + ")中的关系词(" + str(events_list[i][0][j]) + ")与事件元素(" + str(
                        events_list[i][1][k]) + ")建立一条边")
                else:
                    print("事件(" + str(i) + ")中的关系词(" + str(events_list[i][0][j]) + ")与事件元素(" + str(
                        events_list[i][1][k]) + ")已经存在一条边，不需要重复建立")

    # print("\n提取单词的词向量")
    # for i in range(len(events_list)):
    #     events_word2vec_item=[]
    #     triggers_word2vec_list=[]
    #     for j in range(len(events_list[i][0])):
    #         if chineseWordVecs.has_key(events_list[i][0][j]):
    #             print("\n事件"+str(i)+"中的触发词词语("+events_list[i][0][j]+")具有词向量")
    #             print("该词向量为:"+str(chineseWordVecs[events_list[i][0][j]]))
    #             triggers_word2vec_list.append(chineseWordVecs[events_list[i][0][j]])
    #         else:
    #             print("\n事件" + str(i) + "中的触发词词语(" + events_list[i][0][j] + ")不具有词向量")
    #             zero_list = []
    #             for m in range(100):
    #                 zero_list.append(0)
    #             triggers_word2vec_list.append(zero_list)
    #     arguments_word2vec_list=[]
    #     for k in range(len(events_list[i][1])):
    #         if chineseWordVecs.has_key(events_list[i][1][k]):
    #             print("\n事件" + str(i) + "中的事件元素词语(" + events_list[i][1][k] + ")具有词向量")
    #             print("该词向量为:" + str(chineseWordVecs[events_list[i][1][k]]))
    #             arguments_word2vec_list.append(chineseWordVecs[events_list[i][1][k]])
    #         else:
    #             print("\n事件" + str(i) + "中的事件元素词语(" + events_list[i][1][k]+ ")不具有词向量")
    #             zero_list = []
    #             for m in range(100):
    #                 zero_list.append(0)
    #             arguments_word2vec_list.append(zero_list)
    #     events_word2vec_item.append(triggers_word2vec_list)
    #     events_word2vec_item.append(arguments_word2vec_list)
    #     events_word2vec_list.append(events_word2vec_item)
    # print("\n词向量读取完毕，共有("+str(len(events_word2vec_list))+")个事件")

    words_word2vec_list = []
    print("提取单词的词向量")
    for i in range(len(words_list)):
        if chineseWordVecs.has_key(words_list[i]):
            print("\n词语(" + str(words_list[i]) + ")具有词向量")
            print("该词向量为：" + str(chineseWordVecs[words_list[i]]))
            words_word2vec_list.append(chineseWordVecs[words_list[i]])
        else:
            print("\n词语" + str(words_list[i]) + "不具有词向量")
            zero_list = []
            for m in range(100):
                zero_list.append(random.uniform(-2, 6))
            words_word2vec_list.append(zero_list)
            print("向列表中加入随机向量:" + str(zero_list))

    print("根据余弦相似度将节点连接起来")
    for i in range(len(words_word2vec_list)):
        for j in range(len(words_word2vec_list)):
            if i != j:
                mn = 0
                mm = 0
                nn = 0
                cos_value = 0
                for k in range(100):
                    mn += words_word2vec_list[i][k] * words_word2vec_list[j][k]
                    mm += words_word2vec_list[i][k] * words_word2vec_list[i][k]
                    nn += words_word2vec_list[j][k] * words_word2vec_list[j][k]
                cos_value = mn / (math.sqrt(mm) * math.sqrt(nn))
                # print "词语("+str(words_list[i])+")与词语("+str(words_list[j])+")的余弦相似度为"+str(cos_value)
                if cos_value > 0.7:
                    if (not dg.has_edge((i, j))):
                        dg.add_edge((i, j))
                    print "词语(" + str(words_list[i]) + ")与词语(" + str(words_list[j]) + ")的余弦相似度为" + str(cos_value)
                    print("余弦相似度超过阀值，建立一条边")
                    # else:
                    # print("余弦相似度低于阀值，不做处理")

    print("\n输出所有的连接边:")
    edges = dg.edges()
    for i in range(len(edges)):
        print str(edges[i])
    edges_word = []
    for i in range(len(edges)):
        tup = (words_list[edges[i][0]].decode('utf8'), words_list[edges[i][1]].decode('utf8'))
        print("\n" + tup[0] + "---" + tup[1])
        edges_word.append(tup)
    print("\n为节点赋予标签")
    labels = dict()
    for i in range(len(words_list)):
        labels[i] = words_list[i]
    print("\n使用networkx插件建立新的联通图")
    G = nx.Graph()
    G.add_nodes_from(words_list)
    G.add_edges_from(edges_word)
    # pos=nx.circular_layout(G)
    # pos=nx.spectral_layout(G)
    pos = nx.spring_layout(G)
    # pos=nx.shell_layout(G)
    # nx.draw(G,pos,node_shape='.',node_size=40)
    # nx.draw_networkx_nodes(G,pos,range(len(words_list)),node_shape='.',node_size=40)
    # nx.draw_networkx_labels(G,pos,labels,node_shape='.',node_size=40)
    plt.subplot(2, 2, 1)
    nx.draw_networkx(G, pos, node_shape='.', node_size=40)

    pr = PRIterator(dg)
    page_ranks = pr.page_rank()
    print("提取page_rangk字典中的值")
    for i in range(len(page_ranks)):
        page_rank = []
        page_rank.append(i)
        page_rank.append(page_ranks[i])
        page_ranks_list.append(page_rank)
    print(str(page_ranks_list))

    print("按照PR值对词组进行排序")
    for i in range(len(page_ranks_list) - 1):
        epoch = i + 1
        for j in range(len(page_ranks_list) - epoch):
            page_rank = []
            if page_ranks_list[j][1] > page_ranks_list[j + 1][1]:
                page_rank = page_ranks_list[j]
                page_ranks_list[j] = page_ranks_list[j + 1]
                page_ranks_list[j + 1] = page_rank
    print(str(page_ranks_list))

    nodes_word = []
    exist_nodes = []
    node_state = False
    print "寻找PR值最大的名实体节点"
    for i in range(len(page_ranks_list) - 1):
        PR_index = page_ranks_list[len(page_ranks_list) - i - 1][0]
        if words_list[PR_index] in ner_list:
            node = PR_index
            node_state = True
            print("开始节点为名实体(" + words_list[PR_index] + ")")
            break
    if not (node_state):
        for i in range(len(page_ranks_list) - 1):
            if words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]] in arguments_list:
                node = page_ranks_list[len(page_ranks_list) - i - 1][0]
                break

    max_ner_node = node
    exist_nodes.append(node)
    nodes_word.append(words_list[node])
    while True:
        neighbors = dg.neighbors(node)
        finish_state = True
        for i in range(len(neighbors)):
            if neighbors[i] in exist_nodes:
                finish_state = (finish_state and True)
            else:
                finish_state = (finish_state and False)
        if finish_state:
            break
        for i in range(len(neighbors)):
            if neighbors[i] in exist_nodes:
                continue
            else:
                max_neighbor = neighbors[i]
                print("\n最开始的最大邻接点是(" + words_list[max_neighbor] + ")")
                print("这个节点的PR值为：" + str(page_ranks[max_neighbor]))
                break
        for i in range(len(neighbors)):
            if not (neighbors[i] in exist_nodes):
                if (page_ranks[neighbors[i]] > page_ranks[max_neighbor]):
                    max_neighbor = neighbors[i]
                    print("\n最大邻接点更换为(" + words_list[max_neighbor] + ")")
                    print("这个节点的PR值为：" + str(page_ranks[max_neighbor]))
        node = max_neighbor
        nodes_word.append(words_list[max_neighbor])
        exist_nodes.append(node)
        print("加入词语(" + words_list[node] + ")")

    print("\n权值最大的词语为:(" + words_list[page_ranks_list[len(page_ranks_list) - 1][0]] + "),对应的句子如下:")
    max_word_sentence_id = words_sentence_id_list[page_ranks_list[len(page_ranks_list) - 1][0]]
    for i in range(len(max_word_sentence_id)):
        print str(max_word_sentence_id[i]) + ":  " + sentence_list[max_word_sentence_id[i]]

    article_trigger = []
    article_ner = []
    article_argument = []

    count = 0
    print "\n"
    for i in range(len(page_ranks_list)):
        if words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]] in ner_list:
            count = count + 1
            print "权值排名第(" + str(count) + ")的名实体为:" + words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]]
            article_ner.append(words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]])
            if count == 3 or count == len(ner_list):
                break

    count = 0
    print "\n"
    for i in range(len(page_ranks_list)):
        if (words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]] in triggers_list) and (len(words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]])>3):
            count = count + 1
            print "权值排名第(" + str(count) + ")的触发词为:" + words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]]
            article_trigger.append(words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]])
            if count == 3 or count == len(triggers_list):
                break

    count = 0
    print "\n"
    for i in range(len(page_ranks_list)):
        if words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]] in arguments_list:
            count = count + 1
            print "权值排名第(" + str(count) + ")的事件元素为:" + words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]]
            article_argument.append(words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]])
            if count == 3 or count == len(arguments_list):
                break

    title_argument_word2vec = []
    title_trigger_word2vec = []
    title_ner_word2vec = []
    article_ner_word2vec = []
    article_trigger_word2vec = []
    article_argument_word2vec = []

    f = open("/media/lyt312323529/c4175817-9d97-490b-95c6-636149e75a87/Graph_Generate/news_" + str(
        news_index) + "_result.txt", "w")

    zero_list = []
    for m in range(100):
        zero_list.append(random.uniform(-2, 6))

    for i in range(len(title_argument)):
        if chineseWordVecs.has_key(title_argument[i]):
            print "\n标题事件元素(" + title_argument[i] + ")具有词向量，该词向量为:"
            print str(chineseWordVecs[title_argument[i]])
            title_argument_word2vec.append(chineseWordVecs[title_argument[i]])
        else:
            print("\n词语(" + str(title_argument[i]) + ")不具有词向量")
            title_argument_word2vec.append(zero_list)
            print("向标题事件元素列表中加入随机向量:" + str(zero_list))

    for i in range(len(title_trigger)):
        if chineseWordVecs.has_key(title_trigger[i]):
            print "\n标题触发词(" + title_trigger[i] + ")具有词向量，该词向量为:"
            print str(chineseWordVecs[title_trigger[i]])
            title_trigger_word2vec.append(chineseWordVecs[title_trigger[i]])
        else:
            print("\n词语(" + str(title_trigger[i]) + ")不具有词向量")
            title_trigger_word2vec.append(zero_list)
            print("向标题触发词列表中加入随机向量:" + str(zero_list))

    for i in range(len(title_ner)):
        if chineseWordVecs.has_key(title_ner[i]):
            print "\n标题名实体(" + title_ner[i] + ")具有词向量，该词向量为:"
            print str(chineseWordVecs[title_ner[i]])
            title_ner_word2vec.append(chineseWordVecs[title_ner[i]])
        else:
            print("\n词语(" + str(title_ner[i]) + ")不具有词向量")
            title_ner_word2vec.append(zero_list)
            print("向标题名实体列表列表中加入随机向量:" + str(zero_list))

    for i in range(len(article_ner)):
        if chineseWordVecs.has_key(article_ner[i]):
            print "\n新闻名实体(" + article_ner[i] + ")具有词向量，该词向量为:"
            print str(chineseWordVecs[article_ner[i]])
            article_ner_word2vec.append(chineseWordVecs[article_ner[i]])
        else:
            print("\n词语(" + str(article_ner[i]) + ")不具有词向量")
            article_ner_word2vec.append(zero_list)
            print("向新闻名实体列表中加入随机向量:" + str(zero_list))

    for i in range(len(article_trigger)):
        if chineseWordVecs.has_key(article_trigger[i]):
            print "\n新闻触发词(" + article_trigger[i] + ")具有词向量，该词向量为:"
            print str(chineseWordVecs[article_trigger[i]])
            article_trigger_word2vec.append(chineseWordVecs[article_trigger[i]])
        else:
            print("\n词语(" + str(article_trigger[i]) + ")不具有词向量")
            article_trigger_word2vec.append(zero_list)
            print("向新闻触发词列表中加入随机向量:" + str(zero_list))

    for i in range(len(article_argument)):
        if chineseWordVecs.has_key(article_argument[i]):
            print "\n新闻事件元素(" + article_argument[i] + ")具有词向量，该词向量为:"
            print str(chineseWordVecs[article_argument[i]])
            article_argument_word2vec.append(chineseWordVecs[article_argument[i]])
        else:
            print("\n词语(" + str(article_argument[i]) + ")不具有词向量")
            article_argument_word2vec.append(zero_list)
            print("向新闻事件元素列表中加入随机向量:" + str(zero_list))
    title_ner_str = ""
    for i in range(len(title_ner)):
        title_ner_str = title_ner_str + title_ner[i] + " "
    title_ner_str = title_ner_str + "\n"

    title_trigger_str = ""
    for i in range(len(title_trigger)):
        title_trigger_str = title_trigger_str + title_trigger[i] + " "
    title_trigger_str = title_trigger_str + "\n"

    title_argument_str = ""
    for i in range(len(title_argument)):
        title_argument_str = title_argument_str + title_argument[i] + " "
    title_argument_str = title_argument_str + "\n"

    article_ner_str = ""
    for i in range(len(article_ner)):
        article_ner_str = article_ner_str + article_ner[i] + " "
    article_ner_str = article_ner_str + "\n"

    article_trigger_str = ""
    for i in range(len(article_trigger)):
        article_trigger_str = article_trigger_str + article_trigger[i] + " "
    article_trigger_str = article_trigger_str + "\n"

    article_argument_str = ""
    for i in range(len(article_argument)):
        article_argument_str = article_argument_str + article_argument[i] + " "
    article_argument_str = article_argument_str + "\n"

    insert_str = title_ner_str + title_trigger_str + title_argument_str + article_ner_str + article_trigger_str + article_argument_str
    f.write(insert_str)

    f.close()

    print("\n权值最大的名实体为:(" + words_list[max_ner_node] + "),对应的句子如下:")
    max_word_sentence_id = words_sentence_id_list[max_ner_node]
    for i in range(len(max_word_sentence_id)):
        print str(max_word_sentence_id[i]) + ":  " + sentence_list[max_word_sentence_id[i]]
    neighbors = dg.neighbors(max_ner_node)
    neighbors_trigger_list = []
    for i in range(len(neighbors)):
        if words_list[neighbors[i]] in triggers_list:
            print "相邻节点(" + words_list[neighbors[i]] + ")是触发词,加入相邻触发词列表"
            neighbors_trigger_list.append(neighbors[i])
    max_trigger_page_rank = page_ranks[neighbors_trigger_list[0]]
    max_trigger = neighbors_trigger_list[0]
    for i in range(len(neighbors_trigger_list) - 1):
        if page_ranks[neighbors_trigger_list[i + 1]] > max_trigger_page_rank:
            max_trigger_page_rank = page_ranks[neighbors_trigger_list[i + 1]]
            max_trigger = neighbors_trigger_list[i + 1]
    print "名实体(" + words_list[max_ner_node] + ")的相邻的权值最大的触发词为(" + words_list[max_trigger] + ")"
    print "权值最大的名实体以及它权值最大的邻接点触发词对应的句子如下:"
    for i in range(len(events_list)):
        if (words_list[max_trigger] in events_list[i][0]) and (words_list[max_ner_node] in events_list[i][1]):
            print "\n" + sentence_list[events_list[i][2]]
            print "\n触发词为:"
            for j in range(len(relations_list[events_list[i][3]][0])):
                print relations_list[events_list[i][3]][0][j]
            print "事件元素为:"
            for k in range(len(relations_list[events_list[i][3]][1])):
                print relations_list[events_list[i][3]][1][k]

    for i in range(len(page_ranks_list) - 1):
        if words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]] in triggers_list:
            print "\n权值最大的触发词为:(" + words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]] + "),对应句子如下:"
            max_word_sentence_id = words_sentence_id_list[page_ranks_list[len(page_ranks_list) - i - 1][0]]
            for j in range(len(max_word_sentence_id)):
                print str(max_word_sentence_id[j]) + ":  " + sentence_list[max_word_sentence_id[j]]
            break

    # print "\n触发词列表如下"
    # for i in range(len(arguments_list)):
    #     print arguments_list[i]

    for i in range(len(page_ranks_list) - 1):
        if words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]] in arguments_list:
            print "\n权值最大的事件元素为:(" + words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]] + "),对应的句子如下:"
            max_word_sentence_id = words_sentence_id_list[page_ranks_list[len(page_ranks_list) - i - 1][0]]
            for j in range(len(max_word_sentence_id)):
                print str(max_word_sentence_id[j]) + ":  " + sentence_list[max_word_sentence_id[j]]
            break

    print("\n绘制最大权重路径")
    G2 = nx.Graph()
    edges_word = []
    for i in range(len(nodes_word) - 1):
        tup = (nodes_word[i].decode('utf-8'), nodes_word[i + 1].decode('utf-8'))
        edges_word.append(tup)
    G2.add_nodes_from(nodes_word)
    G2.add_edges_from(edges_word)
    pos = nx.spring_layout(G2)
    plt.subplot(2, 2, 4)
    nx.draw_networkx(G2, pos, node_shape='.', node_size=40)

    plt.savefig("graph"+str(news_index)+".png")
    #plt.show()