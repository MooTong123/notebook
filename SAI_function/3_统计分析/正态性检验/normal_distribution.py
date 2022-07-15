# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    # <editable>
    '''
    载入模块
    '''
    from scipy.stats import normaltest
    import pandas as pd
    import db_utils
    import numpy as np
    import report_utils
    report = report_utils.Report()

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])

    '''
    正态性检验
    '''
    data_in = data_in.select_dtypes(include=['number'])  # 筛选数值型数据
    p = normaltest(data_in, nan_policy=params['nan_policy'])[1]

    report.h1('正态性检验结果')
    report.h3('检验结果，当p<0.05时，可以证明数据不服从正态分布')
    p = pd.DataFrame(p, index=data_in.columns,columns=['p值'])
    p = p.reset_index()

    report.table(np.around(p, decimals=4))

    '''
    生成报告
    '''
    report.writeToHtml(reportFileName)
    # </editable>