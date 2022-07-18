# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    '''
    载入模块
    '''
    from scipy.stats import chisquare
    import pyh
    import report_utils
    import db_utils
    report = report_utils.Report()

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['sequence'] + ' from ' + inputs['data_in'])

    '''
    卡方检验
    '''
    sequence = data_in[params['sequence']]
    p = chisquare(sequence)[1]
    if (p < 0.05):
        report.h1('卡方检验结果')
        report.p("p值为：" + str(p) + ",可以证明检验结果显著")
    else:
        report.h1('卡方检验结果')
        report.p("p值为：" + str(p) + ",无充分证据证明检验结果显著")
    # report.h1('卡方检验结果')
    # report.h3('检验结果')
    # report.p(str(p))

    '''
    生成报告
    '''
    report.writeToHtml(reportFileName)