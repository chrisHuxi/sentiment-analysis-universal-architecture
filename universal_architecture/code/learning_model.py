# -*- coding: UTF-8 -*-
#===========================����Ҫ�õ��ĸ���ģ��=========================#
#�����ļ���дģ��
import text_read_write as Textrw
reload(text_read_write)

#������Ի������ı�ģ��
import personalize.corpus_pre_process as Cprepro
reload(personalize.corpus_pre_process)

#������Ի������ǩģ��
import personalize.label_read as Labelr
reload(personalize.label_read)

#������ظ��Ի�������ģ��
import personalize.load_embedding as Loadwv
reload(personalize.load_embedding)

#����ѧϰģ��
import choose_model.lstm.lstm_model as lstm_model
reload(choose_model.lstm.lstm_model)
#=================================================================#


#����ѡ��model����
class LearingModel:
    