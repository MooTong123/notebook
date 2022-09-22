# -- coding: utf-8 --


def execute(conn, inputs, params, outputs, reportFileName):
    # <editable>
    '''
    载入模块
    '''
    import pandas as pd
    import db_utils

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])

    '''
    相关性分析
    '''
    data_out = data_in.corr(method=params['method'])
    ind = pd.DataFrame({'ind': data_out.index})
    ind.index = data_out.index
    data_out = pd.concat([ind, data_out], axis=1)
    data_out = data_out.round(3)
    '''
    将结果写出
    '''

    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)

    # </editable>