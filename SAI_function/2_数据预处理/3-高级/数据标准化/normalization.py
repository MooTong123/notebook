# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    # <editable>
    '''
    载入模块
    '''
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
    import pandas as pd
    import db_utils
    import pickle
    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])

    '''
    标准化
    '''
    data_in = data_in.select_dtypes(include=['number'])  # 筛选数值型数据

    if (params['method'] == 'MaxAbsScaler'):
        model = MaxAbsScaler()
    elif (params['method'] == 'MinMaxScaler'):
        model = MinMaxScaler()
    else:
        model = StandardScaler()

    data_out = model.fit_transform(data_in)
    data_out = pd.DataFrame(data_out, columns=data_in.columns)

    '''
    保存模型
    '''
    pickle.dump(model, open('model.pkl', 'wb'))

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    return model
    # </editable>