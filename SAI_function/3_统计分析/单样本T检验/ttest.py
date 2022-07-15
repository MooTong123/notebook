# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    # <editable>
    '''
    载入模块
    '''
    from scipy.stats import ttest_1samp
    import pyh
    import report_utils
    import db_utils
    report = report_utils.Report()

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['sequence'] + ' from ' + inputs['data_in'])

    '''
    单样本t检验
    '''
    sequence = data_in[params['sequence']]
    p = ttest_1samp(sequence, float(params['expectation']))[1]
    if (p < 0.05):
        report.h1('单样本t检验结果')
        report.h3('检验结果')
        report.p("p值为：" + str(p) + ",可以证明有统计学意义(小于0.01有显著差异性)")
    else:
        report.h1('单样本t检验结果')
        report.h3('检验结果')
        report.p("p值为：" + str(p) + ",无充分证据证明有统计学意义")

    '''
    生成报告
    '''
    report.writeToHtml(reportFileName)
    # </editable>