# -- coding: utf-8 --



def execute(conn, inputs, params, outputs, reportfileName):
    #<editable>
    '''
    载入模块
    '''
    import db_utils
    import pandas as pd
    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])
    features = data_in[params['features'].split(',')]

    '''
    缺失值处理
    '''
    data_out = data_in.copy()

    if params['method'] == 'drop':
        data_out = data_out.dropna(subset=params['features'].split(','))

    else:
        if params['method'] == 'Median_interpolation':
            features = features.select_dtypes(include=['number'])
            features = features.fillna(features.median())
        elif params['method'] == 'Mode_interpolation':
            features = features.select_dtypes(include=['number'])
            features = features.fillna(features.mode())
        elif params['method'] == 'slinear':
            features = features.select_dtypes(include=['number'])
            features = features.interpolate(method='slinear')
        elif params['method'] == 'quadratic':
            features = features.select_dtypes(include=['number'])
            features = features.interpolate(method='quadratic')
        else:
            features = features.select_dtypes(include=['number'])
            features = features.fillna(features.mean())

        for feature in features.columns:
            data_out[feature] = features[feature]

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    #</editable>

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    data = {'a': [0, 1,np.nan ], 'b': [2, np.nan, 4], 'c': [4, np.nan, 6]}
    a = pd.DataFrame(data)
    print(a)

    c = a.copy()
    c[['a','b']].dropna()
    print(c)

