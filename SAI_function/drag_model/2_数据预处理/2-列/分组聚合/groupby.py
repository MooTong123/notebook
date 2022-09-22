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
    data_in = db_utils.query(conn,'select ' + params['columns'] + ',' + params['by'] + ' from ' + inputs['data_in'])

    '''
    分组聚合
    '''
    b = params['by'].split(',')
    if params['method'] == 'count':
        data_out = data_in.groupby(b).count().reset_index()
    elif params['method'] == 'max':
        data_out = data_in.groupby(b).max().reset_index()
    elif params['method'] == 'mean':
        data_out = data_in.groupby(b).mean().reset_index()
    elif params['method'] == 'median':
        data_out = data_in.groupby(b).median().reset_index()
    elif params['method'] == 'size':
        data_out = data_in.groupby(b).size().reset_index()
    elif params['method'] == 'min':
        data_out = data_in.groupby(b).min().reset_index()
    elif params['method'] == 'std':
        data_out = data_in.groupby(b).std().reset_index()
    else:
        data_out = data_in.groupby(b).sum().reset_index()

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    # </editable>