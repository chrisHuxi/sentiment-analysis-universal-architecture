# -*- coding: UTF-8 -*-
#===========================加载要用到的各类模块=========================#
#引入文件读写模块
import text_read_write as Textrw
reload(text_read_write)

#引入个性化处理文本模块
import personalize.corpus_pre_process as Cprepro
reload(personalize.corpus_pre_process)

#引入个性化处理标签模块
import personalize.label_read as Labelr
reload(personalize.label_read)

#引入加载个性化词向量模块
import personalize.load_embedding as Loadwv
reload(personalize.load_embedding)

#引入学习模型
import choose_model.lstm.lstm_model as lstm_model
reload(choose_model.lstm.lstm_model)
#=================================================================#


#用来选择model的类
class LearingModel:
    