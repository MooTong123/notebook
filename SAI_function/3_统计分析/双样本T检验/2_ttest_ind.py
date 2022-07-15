# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    '''
    载入模块
    '''
    from scipy.stats import ttest_ind
    import pyh
    import report_utils
    import db_utils
    report = report_utils.Report()

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['col1'] + ',' + params['col2'] + ' from ' + inputs['data_in'])

    '''
    双样本t检验
    '''
    col1 = data_in[params['col1']]
    col2 = data_in[params['col2']]
    p = ttest_ind(col1, col2)[1]
    if (p < 0.05):
        report.h1('双样本t检验结果')
        report.h3('检验结果')
        report.p("p值为：" + str(p) + ",认为两者总体均值不同")
    else:
        report.h1('双样本t检验结果')
        report.h3('检验结果')
        report.p("p值为：" + str(p) + ",无充分证据证明两者总体均值不同")

    '''
    生成报告
    '''
    report.writeToHtml(reportFileName)