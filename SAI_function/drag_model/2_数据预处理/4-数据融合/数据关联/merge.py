# -- coding: utf-8 --

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
	left = db_utils.query(conn, 'select '+ params['left_columns'] + ' from ' + inputs['left'])
	right = db_utils.query(conn, 'select '+ params['right_columns'] + ' from ' + inputs['right'])
	'''
	表连接
	'''
	data_out = pd.merge(left, right,
						left_on=params['left_on'].split(","), right_on=params['right_on'].split(","),
	                    how=params['how'], left_index=bool(params['left_index']), right_index=bool(params['right_index']),
						sort=params['sort'])
	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    #</editable>


