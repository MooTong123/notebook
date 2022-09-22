# -- coding: utf-8 --

def execute(conn, inputs, params, outputs, reportFileName):
    '''
    载入模块
    '''
    from sklearn.decomposition import PCA
    import numpy as np
    import pandas as pd
    import db_utils

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])

    '''
    主成分分析
    '''
    data_in = data_in.select_dtypes(include=['number'])  # 筛选数值型数据
    n_samples, n_features = data_in.shape
    if not 1 <= int(params['n_components']) <= n_features:
        raise ValueError("\n降维后的维数为%r,该值必须要在[1,%r]之间." % (int(params['n_components']), n_features))

    pca_model = PCA(n_components=int(params['n_components']))
    pca_model.fit(data_in)
    pca_model.explained_variance_ratio_

    # 执行降维
    data_out = pca_model.transform(data_in)
    columns = list(range(1, int(params['n_components']) + 1))
    columns = ['comp_' + str(i) for i in columns]
    data_out = pd.DataFrame(data_out, columns=columns)
    data_out = np.around(data_out, decimals=4)

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)