# -- coding: utf-8 --

def execute(conn, inputs, params, outputs, reportfileName):
    #<editable>
    '''
	载入模块
    '''
    import db_utils

    '''
    选择目标数据
	'''
    data_in = db_utils.query(conn, 'select '+ params['columns'] + ' from ' + inputs['data_in'])

    '''
	去除重复
	'''
    data_in.drop_duplicates(inplace = True)
    data_out = data_in
    '''
	将结果写出
	'''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    #</editable>