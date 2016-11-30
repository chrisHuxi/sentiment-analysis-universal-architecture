# -*- coding: UTF-8 -*-

#=====================加载要用的模块======================#
from __future__ import print_function
#保证python3.x的兼容性
#import os
import numpy as np
#import random
#import string
import tensorflow as tf
#导入画图模块pyplot
import matplotlib.pyplot as plt
#=====================加载要用的模块======================#

#========================读入数据=======================#
####将句子的索引形式读入为np.array的形式####
#输入：句子索引的文件名（路径+文件名），句子的最大长度（int），句子最大长度使能（bool）
#输出：扩充或剪掉的句子索引数列（np.array）
def readSentence2Array(sentenceFileName,inputMaxLength = 128 , inputMaxLengthFlag = True):
    file = open(sentenceFileName,'r')
    lines = file.readlines()
    sentenceList = []
    for line in lines:
        splitList = line.strip().split(' ')
        sentenceList.append(splitList)
    maxLength = len(sentenceList[0])
    for row in sentenceList:
        if maxLength <= len(row):
            maxLength = len(row)
    if inputMaxLengthFlag == False: 
        for row in sentenceList:
            for i in range(len(row),maxLength):
                row.append('0')
                
    elif inputMaxLengthFlag == True:
        temp_list = []
        for row in sentenceList:
            if len(row) < inputMaxLength:
                for i in range(len(row),inputMaxLength):
                    row.append('0')
                temp_list.append(row)
            elif len(row) >= inputMaxLength:
                temp_list.append(row[0:inputMaxLength])
        sentenceList = temp_list
        
    numSentenceList = strList2numList2D(sentenceList)
    print (np.array(numSentenceList))
    return np.array(numSentenceList)

####将字符类型的list转换为数字类型的list（2D）####
#输入：字符类型的list
#输出：对应的数字类型的list
def strList2numList2D(strlist):
    copyList = []
    for sentence in strlist:
        temp_list = []
        for number in sentence:
            temp_list.append(int(number))
        copyList.append(temp_list)
    return copyList

####读入label文件####
#输入：label文件的文件名
#输出：label数列（np.array）
def readLabelFile2Array(LabelFileName):
    file = open(LabelFileName,'r')
    lines = file.readlines()
    labelList = []
    for line in lines:
        splitList = line.strip()
        labelList.append(splitList)
    return np.array(labelList)
#========================读入数据end====================#

#加载embeddings
savedArrayFile = r'/media/huxi/新加卷1/NLP_exLab/pythonCode/lstm/embedding_and_data/embeddingVocabulary.npy'
wordEmbedding = np.load(savedArrayFile)

lengthEmbedding = np.shape(wordEmbedding)[1]#词向量长度
lengthVocabulary = np.shape(wordEmbedding)[0]#词典长度


sentenceFileName = r'/media/huxi/新加卷1/NLP_exLab/pythonCode/lstm/embedding_and_data/sentence_index.txt'
sentenceArray = readSentence2Array(sentenceFileName)
num_array = np.shape(sentenceArray)[0]#句子条数
sentenceLength = np.shape(sentenceArray)[1]#句子长度



#加载label文件
LabelFileName =  r'/media/huxi/新加卷1/NLP_exLab/pythonCode/lstm/embedding_and_data/shuffledLabel.txt'
labels = readLabelFile2Array(LabelFileName)

#将dataset和label分成8：1：1的比例
train_dataset = sentenceArray[0:int(num_array*0.8)]
valid_dataset = sentenceArray[int(num_array*0.8):int(num_array*0.8)+int(num_array*0.1)]
test_dataset = sentenceArray[int(num_array*0.8)+int(num_array*0.1):]
train_labels = labels[0:int(num_array*0.8)]
valid_labels = labels[int(num_array*0.8):int(num_array*0.8)+int(num_array*0.1)]
test_labels = labels[int(num_array*0.8)+int(num_array*0.1):]

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


#===================一些辅助函数=======================#
####重新构造dataset和labels的形式####（对于CNN需要将data重构为多个通道，而lstm模型不需要）
#输入：直接读出的dataset数列，直接读出的label数列
#输出：重构后的dataset数列（np.array），label数列（np.array）
def reformat(dataset,labels):
  kindOfLabels = list(set(labels))
  reformedLabels = []
  for label in labels:
        reformedLabels.append(kindOfLabels.index(label))
  reformedLabels = np.array(reformedLabels)
  return dataset,reformedLabels
    
#运行函数
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


####根据模型的预测结果与labels比较，得出准确率####
#输入：预测结果（one hot array），labels（一维 array）
#输出：准确率
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == labels)
          /predictions.shape[0])
          
          
          
####计算混淆矩阵####（这个需要根据具体问题定制，多分类和二分类问题不同）
#输入：预测结果（one hot array），labels（一维 array）
#输出：混淆矩阵的四个分量列表（list×4）
def calcuConfusionMatrix(predictions, labels):
    truePositiveList = []
    trueNegativeList = []
    falsePositiveList = []
    falseNegativeList = []
    print (labels.shape)
    for i in range(0,labels.shape[0]):
        if np.argmax(predictions, 1)[i] == 1 and labels[i] == 1:
            #true positive
            truePositiveList.append(i)
        elif np.argmax(predictions, 1)[i] == 0 and labels[i] == 0:
            trueNegativeList.append(i)
        elif np.argmax(predictions, 1)[i] == 1 and labels[i] == 0:
            falsePositiveList.append(i)
        elif np.argmax(predictions, 1)[i] == 0 and labels[i] == 1:
            falseNegativeList.append(i)
        else:
            print ('something wrong!')
    confusionMatrixRow1= [len(truePositiveList),len(falseNegativeList)]
    confusionMatrixRow2 = [len(falsePositiveList),len(trueNegativeList)]
    confusionMatrix = [(confusionMatrixRow1),(confusionMatrixRow2)]
    print ('the Confusion Matrix:')
    print (np.mat(confusionMatrix))
    return  truePositiveList,trueNegativeList,falsePositiveList,falseNegativeList

####计算F1值####
#输入：混淆矩阵的四个列表（list×4）
#输出：F1值  
def calcuF1Score(truePositiveList,trueNegativeList,falsePositiveList,falseNegativeList):
        precisionScore = len(truePositiveList)/(0.0 + len(truePositiveList) + len(falsePositiveList))
        recallScore = len(truePositiveList)/(0.0 + len(truePositiveList) + len(falseNegativeList))
        F1Score = 2*len(truePositiveList)/(0.0 + 2*len(truePositiveList) + len(falsePositiveList) + len(falseNegativeList))
        return precisionScore,recallScore,F1Score

        
#===================一些辅助函数end=====================#

#====================计算模型的构建=====================#
####LSTM模型####
#在每一个lstm cell中的hidden unit的数量
num_nodes = 64

#label的类别数，也是分类数
kindNumLabels = len(list(set(labels)))

#设为最大句子长度
num_unrollings = 128

#一个batch的大小
batch_size = 32

'''
wordEmbedding#所有词向量
lengthEmbedding#词向量长度
lengthVocabulary#词典长度

sentenceArray#句子向量
num_array#句子条数
sentenceLength#句子长度
'''
embedding_size = lengthEmbedding#词向量维数

#定义计算图
graph = tf.Graph()
with graph.as_default():
  #词向量初始化为pre-train过的词向量
  embeddings = tf.Variable(wordEmbedding)
  
  # Parameters:
  # Input gate: input, previous output, and bias.
  ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes]))
  # Forget gate: input, previous output, and bias.
  fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  fb = tf.Variable(tf.zeros([1, num_nodes]))
  # Memory cell: input, state and bias.                             
  cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  cb = tf.Variable(tf.zeros([1, num_nodes]))
  # Output gate: input, previous output, and bias.
  ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
  om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ob = tf.Variable(tf.zeros([1, num_nodes]))
  # Variables saving state across unrollings.
  train_saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  train_saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    
  valid_saved_output = tf.Variable(tf.zeros([valid_dataset.shape[0], num_nodes]), trainable=False)
  valid_saved_state = tf.Variable(tf.zeros([valid_dataset.shape[0], num_nodes]), trainable=False)
    
  test_saved_output = tf.Variable(tf.zeros([test_dataset.shape[0], num_nodes]), trainable=False)
  test_saved_state = tf.Variable(tf.zeros([test_dataset.shape[0], num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, kindNumLabels], -0.1, 0.1))
  b = tf.Variable(tf.zeros([kindNumLabels]))  
    
  # Definition of the cell computation.
  def lstm_cell(i, o, state):

    embed = tf.nn.embedding_lookup(embeddings, i)#1×embedding_size 这里的i需要是一个int32或int64
    input_gate = tf.sigmoid(tf.matmul(embed, ix) + tf.matmul(o, im) + ib)#i为输入的x,o为上一个cell中传来的h
    forget_gate = tf.sigmoid(tf.matmul(embed, fx) + tf.matmul(o, fm) + fb)
    update = tf.matmul(embed, cx) + tf.matmul(o, cm) + cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(tf.matmul(embed, ox) + tf.matmul(o, om) + ob)
    return output_gate * tf.tanh(state), state
  '''
  #==========试试能不能写个model出来=============
  '''
  def model(data,chooseState="train"):
      # Unrolled LSTM loop.
        
      outputs = list()
      if chooseState=="train":
            saved_output = train_saved_output
            saved_state = train_saved_state
      elif chooseState=="valid":
            saved_output = valid_saved_output
            saved_state = valid_saved_state
      elif chooseState=="test":
            saved_output = test_saved_output
            saved_state = test_saved_state
            
      output = saved_output
      state = saved_state
      for i in data:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)
        
      pooling = tf.reduce_mean(tf.pack(outputs), 0)
      #pooling之后应该是（batch_size，num_nodes）
      if chooseState=="train":
          # State saving across unrollings.
          with tf.control_dependencies([train_saved_output.assign(output),
                                        train_saved_state.assign(state)]):#使得这个语句块中的语句在saved_output.assign(output),saved_state.assign(state)执行之后执行
            # Classifier.
            #logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
            logits = tf.nn.xw_plus_b(pooling, w, b)
      elif chooseState=="valid":
          # State saving across unrollings.
          with tf.control_dependencies([valid_saved_output.assign(output),
                                        valid_saved_state.assign(state)]):#使得这个语句块中的语句在saved_output.assign(output),saved_state.assign(state)执行之后执行
            # Classifier.
            #logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
            logits = tf.nn.xw_plus_b(pooling, w, b)
      elif chooseState=="test":
          # State saving across unrollings.
          with tf.control_dependencies([test_saved_output.assign(output),
                                        test_saved_state.assign(state)]):#使得这个语句块中的语句在saved_output.assign(output),saved_state.assign(state)执行之后执行
            # Classifier.
            #logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
            logits = tf.nn.xw_plus_b(pooling, w, b)
      return logits
  '''
  ==============================================
  '''
    
    
  tf_train_inputs = list()
  for _ in range(num_unrollings):
    tf_train_inputs.append(tf.placeholder(tf.int64, shape=[batch_size]))#[batch_size,vocabulary_size]))
    tf_train_inputs
  tf_train_labels = tf.placeholder(tf.int64, shape=[batch_size])
    
  #train_inputs = train_data[:num_unrollings]
  #train_labels = train_data[1:]  # labels are inputs shifted by one time step.
  #tf.cast(train_labels,tf.float64)
    
  tf_train_labels_float = []
  for i,_ in enumerate(labels):
      #print (type(train_labels[i]))
      tf_train_labels_float.append(tf.cast(tf_train_labels[i],tf.float32))
        
#===============开发集数据输入==============#
  tf_valid_dataset = list()
  for i in range(num_unrollings):
        #tf_valid_dataset.append(tf.constant(valid_dataset[:,i]))
        tf_valid_dataset.append(tf.placeholder(tf.int64, shape=[valid_dataset.shape[0]]))
#========================================#            
        
#===============测试集数据输入==============#
  tf_test_dataset = list()
  for i in range(num_unrollings):
        tf_test_dataset.append(tf.placeholder(tf.int64, shape=[test_dataset.shape[0]]))#[batch_size,vocabulary_size]))
#========================================#                    

        
  logits = model(tf_train_inputs)
  # State saving across unrollings.
  #with tf.control_dependencies([saved_output.assign(output),
  #                                 saved_state.assign(state)]):#使得这个语句块中的语句在saved_output.assign(output),saved_state.assign(state)执行之后执行
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        logits, tf.one_hot(indices = tf.concat(0, tf_train_labels),depth = kindNumLabels,on_value = 1.0,off_value = 0.0)))
  
  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)#这下面的几行干啥用的？
  gradients, v = zip(*optimizer.compute_gradients(loss))#这里类似于将不同值的梯度值全部unpack，然后分别放到gradients和v两个列表里面
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)#这里是增加正则项
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)
    
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset,chooseState = "valid"))
  test_prediction = tf.nn.softmax(model(tf_test_dataset,chooseState = "test"))       

#====================计算模型的构建end==================#

 

#=====================运行计算模型=====================# 
train_plot_x = []
train_plot_y = []
valid_plot_y = []

num_steps = 1501


with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size)]
    feed_dict = dict()
    feed_dict[tf_train_labels] = batch_labels

    
    for i in range(num_unrollings):
      feed_dict[tf_train_inputs[i]] = batch_data[:,i]   

      feed_dict[tf_valid_dataset[i]] = valid_dataset[:,i]           
      feed_dict[tf_test_dataset[i]] = test_dataset[:,i]   
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    


    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      train_plot_x.append(step)
      train_plot_y.append(accuracy(predictions, batch_labels))
       
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(feed_dict = feed_dict), valid_labels))
      valid_plot_y.append(accuracy(valid_prediction.eval(feed_dict = feed_dict), valid_labels))
       
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(feed_dict = feed_dict), test_labels))
  truePositiveList,trueNegativeList,falsePositiveList,falseNegativeList = calcuConfusionMatrix(test_prediction.eval(feed_dict = feed_dict), test_labels)
  print('precision score: %.1f%%    recall score: %.1f%%    F1 score: %.1f%%' % calcuF1Score( truePositiveList,trueNegativeList,falsePositiveList,falseNegativeList ))
  plt.plot(train_plot_x, train_plot_y,'b-',label = "train accuracy")
  plt.plot(train_plot_x, valid_plot_y,'r-',label = "valid accuracy")
  plt.xlabel("step")
  plt.ylabel("accuracy")
  plt.show()
  
#=====================运行计算模型end==================# 
''''''  