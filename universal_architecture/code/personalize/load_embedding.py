# -*- coding: utf-8 -*-
from gensim.models import Word2Vec

#=============���ز�ͬ������=============#
'''     �ɶ���      '''
####����word embedding����####
#���룺��
#��������غõ�ģ�ͣ���ͨ��model['s']�����ʴ�������ÿһ��������Ϊ60ά��ndarray
def loadEmbedding():
	
    model = Word2Vec.load(r'D:\NLP_exLab\pythonCode\CNN\wordEmbedding\Word60.model')	#3���ļ�����һ��Word60.model   Word60.model.syn0.npy   Word60.model.syn1neg.npy
    return model
    
