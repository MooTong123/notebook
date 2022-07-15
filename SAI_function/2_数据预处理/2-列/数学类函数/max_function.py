# -- coding: utf-8 --


def execute(conn, inputs, params, outputs, reportFileName):
    # <editable>
    '''
    导入模块
    '''
    import math
    import db_utils

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])
    '''
    使用数学类函数
    '''
    fun = eval(params['method'])
    if data_in[params['label']].dtypes == 'float64' or data_in[params['label']].dtypes == 'int':
        data_in[params['label']] = data_in[params['label']].apply(lambda x: fun(x))
        data_out = data_in
    else:
        raise ValueError('请选择数值型数据！')

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    # </editable>