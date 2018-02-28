#coding=utf-8
import sys
import string
import os
import math
from pygraph.classes.graph import graph
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import subprocess
from matplotlib.font_manager import FontManager


events_list=[]
events_code_list=[]
events_word2vec_list=[]
triggers_list=[]
arguments_list=[]
words_list=[]
page_ranks_list=[]
chineseWordVecs=dict()

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
                    digraph.add_edge(self.graph,(node,node2))
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

def main():
    if len(sys.argv)!=4:
        print (str(len(sys.argv)))
    else:
        for i in range(len(sys.argv)):
            print sys.argv[i]

    print("read triggers list")
    triggers_input=open(sys.argv[1],"r")
    for line in triggers_input:
        if line.count("\t")<6:
            print(str(line.count("\t")))
            continue
        else:
            tokens=line.strip().split("\t")
            print(tokens[4]+":"+tokens[3])
            begin_index=tokens[3].strip().split(",")[0]
            print("begin index is"+begin_index)
            trigger=[]
            trigger.append(tokens[4])
            trigger.append(begin_index)
            triggers_list.append(trigger)
    print("length of triggers list is "+str(len(triggers_list)))

    print("read arguments list")
    arguments_input=open(sys.argv[2],"r")
    for line in arguments_input:
        if line.count("\t")<11:
            print(str(line.count("\t")))
            continue
        else:
            tokens=line.strip().split("\t")
            tokens[4]=tokens[4].replace(',','').replace('，','')
            tokens[4]=tokens[4].replace('。','').replace('.','')
            tokens[4]=tokens[4].replace('[','').replace(']','')
            tokens[4]=tokens[4].replace('；','').replace(';','')
            tokens[4]=tokens[4].replace('《','').replace('》','')
            tokens[4]=tokens[4].replace('￥','').replace('$','') 
            tokens[4]=tokens[4].replace('、','').replace('\\','')
            tokens[4]=tokens[4].strip()           
            print(tokens[4]+":"+tokens[11])
            argument=[]
            argument.append(tokens[4])
            argument.append(tokens[11])
            arguments_list.append(argument)
    print("length of arguments list is "+str(len(arguments_list)))
    
    print("generate event")
    for i in range(len(arguments_list)):
        if len(events_list)==0:
            for j in range(len(triggers_list)):
                if arguments_list[i][1]==triggers_list[j][1]:
                    print("插入第一个事件")
                    event=[]
                    event.append(arguments_list[i])
                    event.append(triggers_list[j])
                    events_list.append(event)
                    print("插入结果为：")
                    print(arguments_list[i][0]+":"+arguments_list[i][1])
                    print(triggers_list[j][0]+":"+triggers_list[j][1])
                    print(events_list[0][0][0]+":"+events_list[0][0][1])
                    print(events_list[0][1][0]+":"+events_list[0][1][1])
                    break
        else:
            exist_state=False
            for k in range(len(events_list)):
                if arguments_list[i][1]==events_list[k][0][1]:
                    events_list[k].append(arguments_list[i])
                    print("该事件已经存在于事件列表中，现向列表中添加事件元素")
                    print("添加的事件元素为："+arguments_list[i][0]+":"+arguments_list[i][1])
                    print("该事件为：")
                    for l in range(len(events_list[k])):
                        print(events_list[k][l][0]+":"+events_list[k][l][1])
                    exist_state=True
                    break
            if not exist_state:
                for j in range(len(triggers_list)):
                    if arguments_list[i][1]==triggers_list[j][1]:
                        print("插入新的事件如下：")
                        event=[]
                        event.append(arguments_list[i])
                        event.append(triggers_list[j])
                        events_list.append(event)
                        print("插入结果为：")
                        print(arguments_list[i][0]+":"+arguments_list[i][1])
                        print(triggers_list[j][0]+":"+triggers_list[j][1])
                        print(events_list[len(events_list)-1][0][0]+":"+events_list[len(events_list)-1][0][1])
                        print(events_list[len(events_list)-1][1][0]+":"+events_list[len(events_list)-1][1][1])
    print("事件提取完毕")

    print("\n生成词表")
    for i in range(len(events_list)):
        for j in range(len(events_list[i])):
            if len(words_list)==0:
                words_list.append(events_list[i][j][0])
                print("向词表中添加第一个词,这个词是："+words_list[0])
            else:
                exist_state=False
                for k in range(len(words_list)):
                    if events_list[i][j][0]==words_list[k]:
                        print("词语("+events_list[i][j][0]+")已经存在，不需要重复加入，这个存在的词语是:"+words_list[k])
                        exist_state=True
                        break
                if not exist_state:
                    words_list.append(events_list[i][j][0])
                    print("将一个新词("+events_list[i][j][0]+")添加进词表中，这个词是:"+words_list[len(words_list)-1])

    print("\n将事件触发词和事件元素替换为词典编码")
    for i in range(len(events_list)):
        event=[]
        for j in range(len(events_list[i])):
            for k in range(len(words_list)):
                if words_list[k]==events_list[i][j][0]:
                    event.append(k)
                    break
        events_code_list.append(event)
        for l in range(len(events_list[i])):
            print("词语("+events_list[i][l][0]+")替换为("+str(events_code_list[len(events_code_list)-1][l])+")")
        print("\n")

    print("\n生成联通图")
    #dg=digraph()
    dg=graph()
    nodes_list=range(len(words_list))
    dg.add_nodes(nodes_list)
    print("将句子级的事件元素以及事件触发词串联起来")
    for i in range(len(events_code_list)):
        for j in range(len(events_code_list[i])):
            for k in range(len(events_code_list[i])):
                if not dg.has_edge((events_code_list[i][j],events_code_list[i][k])) and (j!=k):
                    dg.add_edge((events_code_list[i][j],events_code_list[i][k]))
                    print("事件"+str(i)+"中的("+words_list[events_code_list[i][j]]+")与("+words_list[events_code_list[i][k]]+")之间建立了一条边")

    print("\n显示可用字体")
    fm=FontManager()
    mat_fonts=set(f.name for f in fm.ttflist)
    output=subprocess.check_output('fc-list :lang=zh -f "%{family}\n"',shell=True)
    zh_fonts=set(f.split(',',1)[0] for f in output.split('\n'))
    available=mat_fonts
    print '*'*10,'可用的字体','*'*10
    for f in available:
        print f

    print("read CHINESE_WORD2VEC_FILENAME")
    word2vec=open(sys.argv[3],"r")
    for line in word2vec:
        if line.count(" ")<5:
            continue
        else:
            index=line.find(" ")
            curWord=line[:index]
            rest=line[index+1:]
            tokens=rest.strip().split(" ")
            numTokens=[]
            for tok in tokens:
                numTokens.append(float(tok))
            chineseWordVecs[curWord]=numTokens
    word2vec.close()
    print("load CHINESE_WORD2VEC_FILENAME successfully")
    
    print("\n提取单词的词向量")
    for i in range(len(events_code_list)):
        event_word2vec=[]
        for j in range(len(events_code_list[i])):
            event_word_word2vec=[]
            words=words_list[events_code_list[i][j]].split(" ")
            for l in range(len(words)):
                if chineseWordVecs.has_key(words[l]):
                    #print("\n事件"+str(i)+"中的词语("+words[l]+")具有词向量")
                    #print("该词向量为:"+str(chineseWordVecs[words[l]]))
                    event_word_word2vec.append(chineseWordVecs[words[l]])
                else:
                    #print("\n事件"+str(i)+"中的词语("+words[l]+")不具有词向量")
                    zero_list=[]
                    for m in range(100):
                        zero_list.append(0)
                    event_word_word2vec.append(zero_list)
            event_word2vec.append(event_word_word2vec)
        events_word2vec_list.append(event_word2vec)
    #for i in range(len(events_code_list)):
        #for j in range(len(events_code_list[i])):
            #words=words_list[events_code_list[i][j]].split(" ")
            #for l in range(len(words)):
                #if chineseWordVecs.has_key(words[l]):
                    #print("\n事件"+str(i)+"中的词语("+words[l]+")具有词向量")
                    #print("该词向量为:"+str(chineseWordVecs[words[l]]))
                    #print("提取出的词向量为:"+str(events_word2vec_list[i][j][l]))
                    #print("提取出的词向量长度为:"+str(len(events_word2vec_list[i][j][l])))
                #else:
                    #print("\n事件"+str(i)+"中的词语("+words[l]+")不具有词向量")
                    #print("提取出的词向量为:"+str(events_word2vec_list[i][j][l]))
                    #print("提取出的词向量长度为:"+str(len(events_word2vec_list[i][j][l])))

    print("根据余弦相似度，将词组用边连接起来")
    for i in range(len(events_code_list)):
        for j in range(len(events_code_list)):
            if i==j:
                continue
            else:
                #print("将事件("+str(i)+")与事件("+str(j)+")做比较")
                for k in range(len(events_code_list[i])):
                    for l in range(len(events_code_list[j])):
                        #print("将事件("+str(i)+")的第("+str(k)+")个词组与事件("+str(j)+")的第("+str(l)+")个词组做比较")
                        max_cos=0
                        for m in range(len(events_word2vec_list[i][k])):
                            for n in range(len(events_word2vec_list[j][l])):
                                mn=0
                                mm=0
                                nn=0
                                for o in range(100):
                                    mn+=(events_word2vec_list[i][k][m][o]*events_word2vec_list[j][l][n][o])
                                    mm+=(events_word2vec_list[i][k][m][o]*events_word2vec_list[i][k][m][o])
                                    nn+=(events_word2vec_list[j][l][n][o]*events_word2vec_list[j][l][n][o])
                                if (mn==0)or(nn==0)or(mm==0):
                                    cos=0
                                else:
                                    cos=mn/(math.sqrt(mm)*math.sqrt(nn))
                                if cos>max_cos:
                                    max_cos=cos
                        #print("词组("+words_list[events_code_list[i][k]]+")与词组("+words_list[events_code_list[j][l]]+")的余弦相似度为:"+str(max_cos))
                        if max_cos>0.7:
                            if not dg.has_edge((events_code_list[i][k],events_code_list[j][l])):
                                dg.add_edge((events_code_list[i][k],events_code_list[j][l]))
                                print("词组("+words_list[events_code_list[i][k]]+")与词组("+words_list[events_code_list[j][l]]+")的余弦相似度为("+str(max_cos)+")超过阀值，建立一条边")
    pr=PRIterator(dg)
    page_ranks=pr.page_rank()
    print("词表长度为:"+str(len(words_list)))
    
    print("提取page_rangk字典中的值")
    for i in range(len(words_list)):
        page_rank=[]
        page_rank.append(i)
        page_rank.append(page_ranks[i])
        page_ranks_list.append(page_rank)
    print(str(page_ranks_list))
    
    print("按照PR值对词组进行排序")
    for i in range(len(page_ranks_list)-1):
        epoch=i+1
        for j in range(len(page_ranks_list)-epoch):
            page_rank=[]
            if page_ranks_list[j][1]>page_ranks_list[j+1][1]:
                page_rank=page_ranks_list[j]
                page_ranks_list[j]=page_ranks_list[j+1]
                page_ranks_list[j+1]=page_rank
    print(str(page_ranks_list))
    
    #print("提取出的篇章级事件为:")
    #for i in range(len(page_ranks_list)):
        #print("\n核心短语为:"+words_list[page_ranks_list[len(page_ranks_list)-i-1][0]])
        #neighbors=dg.neighbors(page_ranks_list[len(page_ranks_list)-i-1][0])
        #for j in range(len(neighbors)):
            #print(words_list[neighbors[j]])
    print("\n输出所有的连接边:")
    edges=dg.edges()
    for i in range(len(edges)):
        print str(edges[i])

    print("\n将编码替换为词组")
    nodes_word=[]
    for i in range(len(words_list)):
        nodes_word.append(words_list[i].decode('utf8'))
    edges_word=[]
    for i in range(len(edges)):
        tup=(words_list[edges[i][0]].decode('utf8'),words_list[edges[i][1]].decode('utf8'))
        #print("\n"+tup[0]+"---"+tup[1])
        edges_word.append(tup)

    print("\n为节点赋予标签")
    labels=dict()
    for i in range(len(words_list)):
        labels[i]=words_list[i]

    print("\n使用networkx插件建立新的联通图")
    G=nx.Graph()
    G.add_nodes_from(nodes_word)
    G.add_edges_from(edges_word)
    #pos=nx.circular_layout(G)
    #pos=nx.spectral_layout(G)
    pos=nx.spring_layout(G)
    #pos=nx.shell_layout(G)
    #nx.draw(G,pos,node_shape='.',node_size=40)
    #nx.draw_networkx_nodes(G,pos,range(len(words_list)),node_shape='.',node_size=40)
    #nx.draw_networkx_labels(G,pos,labels,node_shape='.',node_size=40)
    plt.subplot(2,2,1)
    nx.draw_networkx(G,pos,node_shape='.',node_size=40)


    # print("\n寻找权值最大的路径并绘制出来")
    # dis=[]
    # path=[]
    # print("初始化路径矩阵")
    # for i in range(len(words_list)):
    #     path_i=[]
    #     for j in range(len(words_list)):
    #         path_i.append(j)
    #     path.append(path_i)
    # print("初始化邻接矩阵")
    # FLOAT_MAX=0.0
    # for i in range(len(words_list)):
    #     dis_i=[]
    #     for j in range(len(words_list)):
    #         if dg.has_edge((i,j)):
    #             dis_i.append(0.0-page_ranks[i]-page_ranks[j])
    #             print("\n节点("+words_list[i]+")与节点("+words_list[j]+")之间存在一条边，这条边的权值为:"+str(dis_i[len(dis_i)-1]))
    #         else:
    #             dis_i.append(FLOAT_MAX)
    #             print(
    #             "\n节点(" + words_list[i] + ")与节点(" + words_list[j] + ")之间不存在边，将权值矩阵此处的值初始化为:" + str(dis_i[len(dis_i) - 1]))
    #     dis.append(dis_i)
    # print("\n开始寻找任意两个节点之间权值最大的路径")
    # for temp in range(len(words_list)):
    #     for row in range(len(words_list)):
    #         for col in range(len(words_list)):
    #             if dg.has_edge((temp,row)) and dg.has_edge((temp,col)):
    #                 select=dis[row][temp]+dis[temp][col]
    #                 if dis[row][col]>select:
    #                     dis[row][col]=select
    #                     path[row][col]=path[row][temp]
    #                     print("\n节点("+words_list[row]+")与节点("+words_list[col]+"之间插入了节点("+wodrs_list[temp]+")")
    print("\n寻找权值最大的路径")
    nodes_word=[]
    exist_nodes=[]
    node=""
    triggers_only_list=[]
    for i in range(len(triggers_list)):
        triggers_only_list.append(triggers_list[i][0])
    for i in range(len(page_ranks_list)):
        if words_list[page_ranks_list[len(page_ranks_list)-i-1][0]] in triggers_only_list:
            print(words_list[page_ranks_list[len(page_ranks_list)-i-1][0]]+":"+str(page_ranks_list[len(page_ranks_list)-i-1][1]))
    for i in range(len(page_ranks_list)):
        if words_list[page_ranks_list[len(page_ranks_list)-i-1][0]] in triggers_only_list:
            #print(words_list[page_ranks_list[len(page_ranks_list)-i-1][0]]+":"+str(page_ranks_list[len(page_ranks_list)-i-1][1]))
            node = page_ranks_list[len(page_ranks_list) - i - 1][0]
            nodes_word.append(words_list[node])
            exist_nodes.append(node)
            print("\n加入词语(" + words_list[node] + ")")
            break
    while True:
        neighbors = dg.neighbors(node)
        finish_state=True
        for i in range(len(neighbors)):
            if neighbors[i] in exist_nodes:
                finish_state=(finish_state and True)
            else:
                finish_state=(finish_state and False)
        if finish_state:
            break
        for i in range(len(neighbors)):
            if neighbors[i] in exist_nodes:
                continue
            else:
                max_neighbor = neighbors[i]
                print("\n最开始的最大邻接点是("+words_list[max_neighbor]+")")
                print("这个节点的PR值为："+str(page_ranks[max_neighbor]))
                break
        for i in range(len(neighbors)):
            if not(neighbors[i] in exist_nodes):
                if (page_ranks[neighbors[i]]>page_ranks[max_neighbor]):
                    max_neighbor = neighbors[i]
                    print("\n最大邻接点更换为(" + words_list[max_neighbor] + ")")
                    print("这个节点的PR值为：" + str(page_ranks[max_neighbor]))
        node=max_neighbor
        nodes_word.append(words_list[max_neighbor])
        exist_nodes.append(node)
        print("加入词语(" + words_list[node] + ")")

    print("\n绘制最大权重路径")
    G2=nx.Graph()
    edges_word=[]
    for i in range(len(nodes_word)-1):
        tup=(nodes_word[i].decode('utf-8'),nodes_word[i+1].decode('utf-8'))
        edges_word.append(tup)
    G2.add_nodes_from(nodes_word)
    G2.add_edges_from(edges_word)
    pos = nx.spring_layout(G2)
    plt.subplot(2, 2, 4)
    nx.draw_networkx(G2, pos, node_shape='.', node_size=40)

    plt.savefig("graph.png")
    plt.show()

if __name__=="__main__":
    main()
