# -*- coding: UTF-8 -*-
import sys
sys.path.append("..")
import text_read_wirte as Textrw
reload(Textrw)


#=============根据不同label标签进行预处理=============#
'''     可定制      '''

####从文件中读入01label####
#输入：文件名
#输出：labelList
def readLabel(fileName):
    return Textrw.readFormFile1DList(fileName)
        
#=============根据不同label标签进行预处理=============#
        
        
        
if __name__ == '__main__':
    #测试代码
    pass