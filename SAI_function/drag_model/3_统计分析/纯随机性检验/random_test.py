# -- coding: utf-8 --

def execute(conn, inputs, params, outputs, reportFileName):
    '''
    载入模块
    '''
    from statsmodels.stats.diagnostic import acorr_ljungbox
    import pyh
    import report_utils
    import db_utils
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")
    report = report_utils.Report()
    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['sequence'] + ' from ' + inputs['data_in'])

    '''
    纯随机性检验
    '''
    data_in = data_in.dropna()
    sequence = data_in[params['sequence']]
    p = acorr_ljungbox(sequence)[1]
    data_out = pd.DataFrame({'lags': range(1, len(p) + 1), 'pvalue': p})

    report.p("如果p值小于0.05时,可以证明通过白噪声检验！")
    report.table(data_out)

    '''
    生成报告
    '''
    report.writeToHtml(reportFileName)
    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    # </editable>