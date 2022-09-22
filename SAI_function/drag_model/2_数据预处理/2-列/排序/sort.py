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
    排序
    '''
    data_out = data_in.sort_values(by=params['by'],
                                   ascending=eval(params['ascending']),
                                   na_position=params['na_position'])
    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    # </editable>