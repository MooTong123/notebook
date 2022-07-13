# -- coding: utf-8 --

def execute(conn, inputs, params, outputs, reportfileName):
    #<editable>
    '''
    载入模块
    '''
    import sqlalchemy
    import db_utils

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select '+ params['columns'] + ' from ' + inputs['data_in'])

    '''
    列名为键，类型为值的字典
    '''
    modifi_type = params['modifi_type']
    type_dict = {}
    for i in eval(modifi_type):
        if i['targetType'] == 'numeric':
            type_dict[i['origName']] = eval(('sqlalchemy.types.NUMERIC' + '(255,' + str(i['otherParam']) + ')'))
        elif i['targetType'] == 'text':
            type_dict[i['origName']] = sqlalchemy.types.Text
        elif i['targetType'] == 'date':
            type_dict[i['origName']] = sqlalchemy.types.Date
        else:
            type_dict[i['origName']] = sqlalchemy.types.TIMESTAMP

    # 写出数据
    db_utils.modifyType(conn, outputs['data_out'], data_in, type_dict)
    #</editable>