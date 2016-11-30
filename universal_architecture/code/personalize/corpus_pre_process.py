# -*- coding: UTF-8 -*-
import pynlpir
from copy import *

#=============根据不同语料进行预处理=============#
'''     可定制      '''

####读入文件进行分词####
#输入：文件名（绝对路径）（string）
#输出：分词结果 （list）
def separateWordFromFile(fileName):
	pynlpir.open()
	file = open(fileName,'r')
	lines = file.readlines()
	i = 0
	allSegmentResult = []
	#print type(s)
	label = []
	for line in lines:
		i = i+1
		textsegment = line
		if textsegment == "\n":
			print "skip"
			continue
		##note:
		'''   gbk 转 utf-8时，    
		   gbk --> unicode --> utf-8
           分解为两个步骤，
                   1.    gbk --> unicode
                             python 语法：你的字符串.decode("gbk")
                   2.    unicode --> utf-8
                            python 语法：你的字符串.decode("gbk").encode("utf-8")
		'''

		segmentResult = pynlpir.segment(textsegment,pos_tagging=True)
		newSegmentResult = removePunctuation(segmentResult)
		allSegmentResult.append(newSegmentResult)

	print len(allSegmentResult)
	file.close()
	pynlpir.close()
	#print label
	return allSegmentResult
    
 
####去除分词工具分词后留下的词性信息####
#输入：分词结果 （list）
#输出：去除词性后的分词结果 （list）
def removePunctuation(segmentResult):
	NewSegmentResult = deepcopy(segmentResult)
	for everyone in NewSegmentResult:
		if (everyone[1]==u'punctuation mark' or everyone[1]== None):
			NewSegmentResult.remove(everyone)
	newSegmentResult = [everyeum[0] for everyeum in NewSegmentResult]
	return newSegmentResult
	
#=============根据不同语料进行预处理=============#    
    
    
if __name__ == '__main__':
    #测试代码
    pass