# -- coding: utf-8 --

def execute(conn, inputs, params, outputs, reportfileName):
	# <editable>
	'''
    载入模块
    '''
	import db_utils
	from sklearn.model_selection import train_test_split

	'''
    选择目标数据
    '''
	data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])

	'''
    数据拆分
    '''
	if params['stratify'] == 'None':
		train, test = train_test_split(data_in, test_size=float(params['test_size']),
									   random_state=int(params['random_state']), shuffle=bool(params['shuffle']),
									   stratify=None)
	else:
		train, test = train_test_split(data_in, test_size=float(params['test_size']),
									   random_state=int(params['random_state']), shuffle=bool(params['shuffle']),
									   stratify=data_in[params['stratify']])
	'''
    将结果写出
    '''
	db_utils.dbWriteTable(conn, outputs['train'], train)
	db_utils.dbWriteTable(conn, outputs['test'], test)
# </editable>