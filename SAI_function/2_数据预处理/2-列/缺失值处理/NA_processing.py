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
    data_in = db_utils.query(conn, 'select '+ params['columns'] + ' from ' + inputs['data_in'])

    '''
    缺失值处理
    '''
    if (params['method'] == 'drop'):
        data_out = data_in.dropna()
    elif (params['method'] == 'Median_interpolation'):
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.fillna(data_in.median())
    elif (params['method'] == 'Mode_interpolation'):
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.fillna(data_in.mode())
    elif (params['method'] == 'slinear'):
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.interpolate(method = 'slinear')
    elif (params['method'] == 'quadratic'):
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.interpolate(method = 'quadratic')
    else :
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.fillna(data_in.mean())

    data_out = pd.DataFrame(data_out)

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    #</editable>