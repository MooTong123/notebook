# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
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
    全表统计
    '''
    data_out = data_in.describe().T
    data_out.columns = ['count', 'mean', 'std', 'min', 'upper_quartile', 'median', 'lower_quartile', 'max']
    index = pd.DataFrame(data_out.index, columns=['col'], index=data_out.index)
    data_out = data_out.apply(lambda x: round(x, 2), axis=1)
    data_out = pd.concat([index, data_out], axis=1)
    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)