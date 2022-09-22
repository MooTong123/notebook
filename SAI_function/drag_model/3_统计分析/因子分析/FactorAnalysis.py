# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    '''
    载入模块
    '''
    from sklearn.decomposition import FactorAnalysis
    import numpy as np
    import db_utils
    import pandas as pd

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])

    '''
    因子分析
    '''
    data_in = data_in.select_dtypes(include=['number'])
    fit = FactorAnalysis(n_components=int(params['n_components']), max_iter=int(params['max_iter'])).fit_transform(
        data_in)
    data_out = pd.DataFrame(fit)
    data_out = np.around(data_out, decimals=4)
    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)