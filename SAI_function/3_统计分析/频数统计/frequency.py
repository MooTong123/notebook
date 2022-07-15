# -- coding: utf-8 --

def execute(conn, inputs, params, outputs, reportFileName):
    '''
    载入模块
    '''
    import pandas as pd
    import db_utils
    from collections import Counter

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])

    '''
    频数统计
    '''
    data = data_in[params['columns']]
    data_out = pd.DataFrame.from_dict(Counter(data), orient='index').reset_index()
    data_out.columns = [params['columns'], 'count']

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
