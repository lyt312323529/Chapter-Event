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

DOMTree=xml.dom.minidom.parse("relation.xml")
doc=DOMTree.documentElement

events_code_list=[]
events_word2vec_list=[]
relations_list=[]
chineseWordVecs=dict()
words_list=[]
page_ranks_list=[]

sentences=doc.getElementsByTagName("sentence")
for sentence in sentences:
    print ("\nThe id of sentence is:"+sentence.getAttribute("id"))
    relations=sentence.getElementsByTagName("relation")
    event=[]
    print("提取出来的关系对数目为："+str(len(relations)))
    for relation in relations:
        print("\n提取relation")
        arguments=relation.getElementsByTagName("argument")
        arguments_list=[]
        trigger_list=[]
        trigger_list.append(relation.getAttribute("pred").encode("utf-8"))
        print("加入的关系元素为("+relation.getAttribute("pred").encode("utf-8")+")")
        for argument in arguments:
            arguments_list.append(argument.getAttribute("content").encode("utf-8"))
            print("加入的元素为("+argument.getAttribute("content").encode("utf-8")+")")
        relations_item=[]
        relations_item.append(trigger_list)
        relations_item.append(arguments_list)
        relations_list.append(relations_item)
print ("一共提取到("+str(len(relations_list))+")组事件对")
arguments_list=[]

MODELDIR="/users4/ytliu/Graph_Generate/ltp_data"
print("正在加载LTP模型...")
segmentor=Segmentor()
p=os.path.join(MODELDIR,"cws.model")
segmentor.load(p)
postagger=Postagger()
postagger.load(os.path.join(MODELDIR,"pos.model"))
recognizer=NamedEntityRecognizer()
recognizer.load(os.path.join(MODELDIR,"ner.model"))
print("加载完毕")

events_list=[]
for i in range(len(relations_list)):
    event = []
    trigger = []
    print("\n\n处理触发词结果如下")
    if (relations_list[i][0][0]!="is")and(relations_list[i][0][0]!="de"):
        words=segmentor.segment(relations_list[i][0][0])
        #wordStr = "\t".join(words)
        #print(wordStr)
        postags=postagger.postag(words)
        #print("\t".join(postags))
        for k in range(len(words)):
            pt=postags[k][0]
            if (pt=="a")or(pt=="b")or(pt=="i")or(pt=="j")or(pt=="m")or(pt=="n")or(pt=="q")or(pt=="r")or(pt=="v"):
                print("词语("+words[k]+")的标签为("+pt+"),符合词性要求，放入触发词列表")
                trigger.append(words[k])
            else:
                print("词语(" + words[k] + ")的标签为(" + pt + "),不符合词性要求，丢弃")
        if len(trigger)>0:
            print("触发词有效，放入事件中")
            event.append(trigger)
            print("\n处理事件元素结果如下")
            arguments_item=[]
            for l in range(len(relations_list[i][1])):
                words = segmentor.segment(relations_list[i][1][l])
                #wordStr = "\t".join(words)
                #print(wordStr)
                postags = postagger.postag(words)
                #print("\t".join(postags))
                argument_item=[]
                argument_words=[]
                argument_postags=[]
                for m in range(len(words)):
                    pt=postags[m][0]
                    if (pt == "a") or (pt == "b") or (pt == "i") or (pt == "j") or (pt == "m") or (pt == "n") or (pt == "q") or (pt == "r") or (pt == "v"):
                        print("词语(" + words[m] + ")的标签为(" + pt + "),符合词性要求，放入事件词汇列表")
                        argument_words.append(words[m])
                        argument_postags.append(postags[m])
                    else:
                        print("词语(" + words[m] + ")的标签为(" + pt + "),不符合词性要求，丢弃")
                netags = recognizer.recognize(argument_words, argument_postags)
                print("\t".join(netags))
                wordStr = ""
                for k in range(len(argument_words)):
                    # print(postags[k][0])
                    if (netags[k][0] == "B") and len(argument_words) > 1:
                        l = k
                        while True:
                            wordStr += argument_words[l]
                            if (netags[l][0] == "E"):
                                break
                            else:
                                l += 1
                        print("识别出名实体(" + wordStr + ")")
                        arguments_item.append(wordStr)
                        print("名实体加入事件")
                        wordStr=""
                    elif (netags[k][0] == "I") or (netags[k][0] == "E"):
                        continue
                    else:
                        arguments_item.append(argument_words[k])
                        print("词语(" + arguments_item[len(arguments_item)-1]+ ")放入事件")
            if len(arguments_item)>0:
                event.append(arguments_item)
                events_list.append(event)
                print("事件元素列表不为空，将事件("+str(len(events_list)-1)+")放入事件列表中")
            else:
                print("事件元素列表为空,不符合事件条件，丢弃事件")
        else:
            print("触发词无效，事件丢弃")
print("共提取("+str(len(events_list))+")个事件")
# print str(len(events_list))
# print str(len(events_list[0]))
# print str(len(events_list[0][0]))
# print str(len(events_list[0][0][0]))
# print events_list[0][0][0][0]
# print events_list[0][1][0][0]

print("\n生成联通图,将节点插入联通图")
# dg=digraph()
dg = graph()
nodes_list = range(len(events_list))
dg.add_nodes(nodes_list)


print("read CHINESE_WORD2VEC_FILENAME")
word2vec = open("/users4/ytliu/Graph_Generate/chinese-wiki-20160305.word2vec", "r")
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


print("\n提取单词的词向量")
for i in range(len(events_list)):
    events_word2vec_item=[]
    triggers_word2vec_list=[]
    for j in range(len(events_list[i][0])):
        if chineseWordVecs.has_key(events_list[i][0][j]):
            print("\n事件"+str(i)+"中的触发词词语("+events_list[i][0][j]+")具有词向量")
            print("该词向量为:"+str(chineseWordVecs[events_list[i][0][j]]))
            triggers_word2vec_list.append(chineseWordVecs[events_list[i][0][j]])
        else:
            print("\n事件" + str(i) + "中的触发词词语(" + events_list[i][0][j] + ")不具有词向量")
            zero_list = []
            for m in range(100):
                zero_list.append(0)
            triggers_word2vec_list.append(zero_list)
    arguments_word2vec_list=[]
    for k in range(len(events_list[i][1])):
        if chineseWordVecs.has_key(events_list[i][1][k]):
            print("\n事件" + str(i) + "中的事件元素词语(" + events_list[i][1][k] + ")具有词向量")
            print("该词向量为:" + str(chineseWordVecs[events_list[i][1][k]]))
            arguments_word2vec_list.append(chineseWordVecs[events_list[i][1][k]])
        else:
            print("\n事件" + str(i) + "中的事件元素词语(" + events_list[i][1][k]+ ")不具有词向量")
            zero_list = []
            for m in range(100):
                zero_list.append(0)
            arguments_word2vec_list.append(zero_list)
    events_word2vec_item.append(triggers_word2vec_list)
    events_word2vec_item.append(arguments_word2vec_list)
    events_word2vec_list.append(events_word2vec_item)
print("\n词向量读取完毕，共有("+str(len(events_word2vec_list))+")个事件")

#print(events_word2vec_list)
for i in range(len(events_word2vec_list)):
    max_cos_trigger=0
    max_cos_argument=0
    for j in range(len(events_word2vec_list)):
        if i==j:
            continue
        else:
            max_cos_trigger=0
            max_cos_argument=0
            for k in range(len(events_word2vec_list[i][0])):
                for l in range(len(events_word2vec_list[j][0])):
                    mn=0
                    mm=0
                    nn=0
                    for o in range(100):
                        mn+=(events_word2vec_list[i][0][k][o]*events_word2vec_list[j][0][l][o])
                        mm+=(events_word2vec_list[i][0][k][o]*events_word2vec_list[i][0][k][o])
                        nn+=(events_word2vec_list[j][0][l][o]*events_word2vec_list[j][0][l][o])
                    if(mn==0)or(nn==0)or(mm==0):
                        cos_trigger=0
                    else:
                        cos_trigger=mn/(math.sqrt(mm)*math.sqrt(nn))
                   #print("触发词("+events_list[i][0][k]+")与触发词("+events_list[j][0][l]+")的余弦相似度为("+str(cos_trigger)+")")
                    if cos_trigger>max_cos_trigger:
                        max_cos_trigger=cos_trigger
            print("触发词相似度为:("+str(max_cos_trigger)+")")
            for k in range(len(events_word2vec_list[i][1])):
                for l in range(len(events_word2vec_list[j][1])):
                    mn=0
                    mm=0
                    nn=0
                    for o in range(100):
                        mn+=(events_word2vec_list[i][1][k][o]*events_word2vec_list[j][1][l][o])
                        mm+=(events_word2vec_list[i][1][k][o]*events_word2vec_list[i][1][k][o])
                        nn+=(events_word2vec_list[j][1][l][o]*events_word2vec_list[j][1][l][o])
                    if(mn==0)or(nn==0)or(mm==0):
                        cos_argument=0
                    else:
                        cos_argument=mn/(math.sqrt(mm)*math.sqrt(nn))
                    #if i==16 and events_list[i][1][k]=="伊拉克":
                        #print str(events_word2vec_list[i][1][k])
                        #print str(events_word2vec_list[j][1][l])
                        #print("事件元素("+events_list[i][1][k]+")与事件元素("+events_list[j][1][l]+")的余弦相似度为("+str(cos_argument)+")")
                    if cos_argument>max_cos_argument:
                        max_cos_argument=cos_argument
            print("事件元素相似度为:("+str(max_cos_argument)+")")
            if (max_cos_trigger+max_cos_argument)>1:
                if (not dg.has_edge((i,j))):
                    dg.add_edge((i,j))
                    print("事件("+str(i)+")事件("+str(j)+")的事件相似度为("+str(max_cos_trigger+max_cos_argument)+"),大于阀值，建立一条边\n")
                else:
                    print("事件("+str(i)+")事件("+str(j)+")的事件相似度为("+str(max_cos_trigger+max_cos_argument)+"),大于阀值，事件之间的边已经存在，不需重复建立\n")
            else:
                print("事件("+str(i)+")事件("+str(j)+")的事件相似度为("+str(max_cos_trigger+max_cos_argument)+"),小于阀值\n")
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
for i in range(3):
    print("\n重要性排在第"+str(i+1)+"的事件为")
    index=page_ranks_list[len(page_ranks_list)-i-1][0]
    for j in range(len(events_list[index][0])):
        print("事件触发词为:"+events_list[index][0][j])
    for j in range(len(events_list[index][1])):
        print("事件元素为:"+events_list[index][1][j])
# print("按照PR值对词组进行排序")
# for i in range(len(page_ranks_list) - 1):
#     epoch = i + 1
#     for j in range(len(page_ranks_list) - epoch):
#         page_rank = []
#         if page_ranks_list[j][1] > page_ranks_list[j + 1][1]:
#             page_rank = page_ranks_list[j]
#             page_ranks_list[j] = page_ranks_list[j + 1]
#             page_ranks_list[j + 1] = page_rank
# print(str(page_ranks_list))
#
# print("\n输出所有的连接边:")
# edges = dg.edges()
# for i in range(len(edges)):
#     print str(edges[i])
#
# print("\n将编码替换为词组")
# nodes_word = []
# for i in range(len(words_list)):
#     nodes_word.append(words_list[i].decode('utf8'))
# edges_word = []
# for i in range(len(edges)):
#     tup = (words_list[edges[i][0]].decode('utf8'), words_list[edges[i][1]].decode('utf8'))
#     # print("\n"+tup[0]+"---"+tup[1])
#     edges_word.append(tup)
#
# print("\n为节点赋予标签")
# labels = dict()
# for i in range(len(words_list)):
#     labels[i] = words_list[i]
#
# print("\n使用networkx插件建立新的联通图")
# G = nx.Graph()
# G.add_nodes_from(nodes_word)
# G.add_edges_from(edges_word)
# # pos=nx.circular_layout(G)
# # pos=nx.spectral_layout(G)
# pos = nx.spring_layout(G)
# # pos=nx.shell_layout(G)
# # nx.draw(G,pos,node_shape='.',node_size=40)
# # nx.draw_networkx_nodes(G,pos,range(len(words_list)),node_shape='.',node_size=40)
# # nx.draw_networkx_labels(G,pos,labels,node_shape='.',node_size=40)
# plt.subplot(2, 2, 1)
# nx.draw_networkx(G, pos, node_shape='.', node_size=40)
#
# print("\n寻找权值最大的路径")
# nodes_word = []
# exist_nodes = []
# node = ""
# # print("\n寻找权值最大的触发词")
# # for i in range(len(page_ranks_list)):
# #     if words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]] in triggers_list:
# #         print(words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]] + ":" + str(
# #             page_ranks_list[len(page_ranks_list) - i - 1][1]))
# # for i in range(len(page_ranks_list)):
# #     if words_list[page_ranks_list[len(page_ranks_list) - i - 1][0]] in triggers_list:
# #         # print(words_list[page_ranks_list[len(page_ranks_list)-i-1][0]]+":"+str(page_ranks_list[len(page_ranks_list)-i-1][1]))
# #         node = page_ranks_list[len(page_ranks_list) - i - 1][0]
# #         nodes_word.append(words_list[node])
# #         exist_nodes.append(node)
# #         print("\n加入词语(" + words_list[node] + ")")
# #         break
# node=page_ranks_list[len(page_ranks_list) - i - 1][0]
# while True:
#     neighbors = dg.neighbors(node)
#     finish_state = True
#     for i in range(len(neighbors)):
#         if neighbors[i] in exist_nodes:
#             finish_state = (finish_state and True)
#         else:
#             finish_state = (finish_state and False)
#     if finish_state:
#         break
#     for i in range(len(neighbors)):
#         if neighbors[i] in exist_nodes:
#             continue
#         else:
#             max_neighbor = neighbors[i]
#             print("\n最开始的最大邻接点是(" + words_list[max_neighbor] + ")")
#             print("这个节点的PR值为：" + str(page_ranks[max_neighbor]))
#             break
#     for i in range(len(neighbors)):
#         if not (neighbors[i] in exist_nodes):
#             if (page_ranks[neighbors[i]] > page_ranks[max_neighbor]):
#                 max_neighbor = neighbors[i]
#                 print("\n最大邻接点更换为(" + words_list[max_neighbor] + ")")
#                 print("这个节点的PR值为：" + str(page_ranks[max_neighbor]))
#     node = max_neighbor
#     nodes_word.append(words_list[max_neighbor])
#     exist_nodes.append(node)
#     print("加入词语(" + words_list[node] + ")")
#
# print("\n绘制最大权重路径")
# G2 = nx.Graph()
# edges_word = []
# for i in range(len(nodes_word) - 1):
#     tup = (nodes_word[i].decode('utf-8'), nodes_word[i + 1].decode('utf-8'))
#     edges_word.append(tup)
# G2.add_nodes_from(nodes_word)
# G2.add_edges_from(edges_word)
# pos = nx.spring_layout(G2)
# plt.subplot(2, 2, 4)
# nx.draw_networkx(G2, pos, node_shape='.', node_size=40)
#
# plt.savefig("graph.png")
# plt.show()
