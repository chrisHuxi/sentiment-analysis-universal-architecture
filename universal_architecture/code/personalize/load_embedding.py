# -*- coding: utf-8 -*-
from gensim.models import Word2Vec

#=============加载不同词向量=============#
'''     可定制      '''
####加载word embedding函数####
#输入：无
#输出：加载好的模型，可通过model['s']来访问词向量，每一个词向量为60维的ndarray
def loadEmbedding():
	
    model = Word2Vec.load(r'D:\NLP_exLab\pythonCode\CNN\wordEmbedding\Word60.model')	#3个文件放在一起：Word60.model   Word60.model.syn0.npy   Word60.model.syn1neg.npy
    return model
    
