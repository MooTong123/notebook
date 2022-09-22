# -- coding: utf-8 --

def execute(conn, inputs, params, outputs, reportfileName):
    # <editable>
    '''
    载入模块
    '''
    import pandas as pd
    import numpy as np
    import db_utils

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])

    '''
    新增自增序列
    '''
    new = np.arange(1, data_in.index.size + 1)
    new = pd.DataFrame({params['new_name']: new})
    data_out = pd.concat([new, data_in], axis=1)

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    # </editable>