# -- coding: utf-8 --

def execute(conn, inputs, params, outputs, reportFileName):
    # <editable>
    '''
    载入模块
    '''

    import db_utils
    import joblib
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])

    features = data_in[params['features'].split(',')]  # features选取的特征需要在columns里
    features = features.select_dtypes(include=['number'])

    '''
    标准化
    '''

    if params['method'] == 'MaxAbsScaler':
        model = MaxAbsScaler()
    elif params['method'] == 'MinMaxScaler':
        model = MinMaxScaler()
    else:
        model = StandardScaler()

    fit_features = model.fit_transform(features)
    fit_features = pd.DataFrame(fit_features, columns=features.columns)

    data_out = data_in.copy()

    for feature in fit_features.columns:
        data_out[feature] = fit_features[feature]

    '''
    保存模型
    '''
    model_file = 'model.pkl'
    joblib.dump(model, model_file)

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    return model
    # </editable>


if __name__ == '__main__':
    import pandas as pd

    data = {'a': [0, 1, 2], 'b': [2, 3, 4], 'c': [4, 5, 6]}
    a = pd.DataFrame(data)
    print(a)

    features = a[['a', 'b']]
    # print(features)

    features += 1
    # print(features)

    for feature in features.columns:
        a[feature] = features[feature]

    print(a)
