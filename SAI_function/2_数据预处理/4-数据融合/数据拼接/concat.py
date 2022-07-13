

def execute(conn, inputs, params, outputs, reportfileName):
    #<editable>
    '''
    载入模块
    '''
    import pandas as pd
    import db_utils

    '''
    选择目标数据
    '''

    if params['columns_left'] == '':
        left = db_utils.query(conn, 'select * from ' + inputs['left'])
    else:
        left = db_utils.query(conn, 'select '+ params['columns_left'] + ' from '+ inputs['left'])

    if params['columns_right'] == '':
        right = db_utils.query(conn, 'select * from '+ inputs['right'])
    else:
        right = db_utils.query(conn, 'select '+ params['columns_right'] + ' from '+ inputs['right'])

    '''
    拼接数据
    '''
    data_out = pd.concat([left, right], axis=int(params['axis']), join=params['join'])

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)

    #</editable>


