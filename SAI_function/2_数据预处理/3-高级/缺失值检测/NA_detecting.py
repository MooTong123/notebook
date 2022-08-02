# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    import pyh
    import db_utils
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])

    '''
    缺失值
    '''

    data_out = pd.DataFrame(data_in.isnull().sum(), columns=['count'], index=data_in.columns).reset_index()

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)