 # -*- coding: UTF-8 -*-

#=================���ļ�================#
 
####���ļ��ж���1άlist��utf8��ʽ��####
#���룺�ļ�����string��
#�����1άlist
def readFormFile1DList(fromFileName):	
	file = open(fromFileName,'r')
	lines = file.readlines()
	resultList = []
	for line in lines:
		resultList.append(line.strip())
	return resultList
    
####���ļ��ж���2άlist��utf8��ʽ��####
#���룺�ļ�����string��
#�����2άlist
def readFormFile1DList(fromFileName):	
    file = open(fromFileName,'r')
    lines = file.readlines()
    resultList = []
    for line in lines:
        rowList = line.strip().split(' ')
        resultList.append(rowList)
    return resultList
    
#=====================================#   
    
    
#=================д�ļ�================#   
    
####��1άlist������ļ���utf8��ʽ��####
#���룺�ļ�����string���� 1άlist��list��
#�������
def output2File1DList(toFileName,list1D):
	f =  open(toFileName,'w')
	WriteText = []
	for everyone in list1D:
		WriteText.append((str(everyone.encode('utf-8'))))
		WriteText.append('\n')
	f.writelines(WriteText)
	f.close()
    
    
####��2άlist������ļ���utf8��ʽ��####
#���룺�ļ�����string����2άlist ��list��
#�������    
def output2File2DList(toFileName,list2D):	
	f =  open(toFileName,'w')
	WriteText = []
	for everyrow in list2D:
		for everycolumn in everyrow:
			WriteText.append((str(everycolumn).encode('utf-8')+' '))
		WriteText.append('\n')
	f.writelines(WriteText)
	f.close()
    
#=====================================#       
    
    
    
if __name__ == '__main__':
    #���Դ���
    pass