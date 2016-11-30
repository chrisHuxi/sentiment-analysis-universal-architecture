# -*- coding: UTF-8 -*-
import pynlpir
from copy import *

#=============���ݲ�ͬ���Ͻ���Ԥ����=============#
'''     �ɶ���      '''

####�����ļ����зִ�####
#���룺�ļ���������·������string��
#������ִʽ�� ��list��
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
		'''   gbk ת utf-8ʱ��    
		   gbk --> unicode --> utf-8
           �ֽ�Ϊ�������裬
                   1.    gbk --> unicode
                             python �﷨������ַ���.decode("gbk")
                   2.    unicode --> utf-8
                            python �﷨������ַ���.decode("gbk").encode("utf-8")
		'''

		segmentResult = pynlpir.segment(textsegment,pos_tagging=True)
		newSegmentResult = removePunctuation(segmentResult)
		allSegmentResult.append(newSegmentResult)

	print len(allSegmentResult)
	file.close()
	pynlpir.close()
	#print label
	return allSegmentResult
    
 
####ȥ���ִʹ��߷ִʺ����µĴ�����Ϣ####
#���룺�ִʽ�� ��list��
#�����ȥ�����Ժ�ķִʽ�� ��list��
def removePunctuation(segmentResult):
	NewSegmentResult = deepcopy(segmentResult)
	for everyone in NewSegmentResult:
		if (everyone[1]==u'punctuation mark' or everyone[1]== None):
			NewSegmentResult.remove(everyone)
	newSegmentResult = [everyeum[0] for everyeum in NewSegmentResult]
	return newSegmentResult
	
#=============���ݲ�ͬ���Ͻ���Ԥ����=============#    
    
    
if __name__ == '__main__':
    #���Դ���
    pass