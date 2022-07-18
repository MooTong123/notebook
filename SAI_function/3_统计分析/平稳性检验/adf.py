# -- coding: utf-8 --

def execute(conn, inputs, params, outputs, reportFileName):
    '''
    载入模块
    '''
    from statsmodels.graphics.tsaplots import plot_acf  # 绘制自相关图
    from statsmodels.tsa.stattools import adfuller as ADF  # 单位根检验
    import matplotlib.pyplot as plt
    import pyh
    import report_utils
    import db_utils
    import warnings
    warnings.filterwarnings("ignore")
    report = report_utils.Report()

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['sequence'] + ' from ' + inputs['data_in'])
    data_in = data_in.dropna()
    '''
    平稳性检验
    '''
    sequence = data_in[params['sequence']]
    adf_result = ADF(sequence)
    test_statistic = adf_result[0]
    p_value = adf_result[1]
    use_lag = adf_result[2]
    nobs = adf_result[3]
    critical_1 = adf_result[4]['5%']
    critical_5 = adf_result[4]['1%']
    critical_10 = adf_result[4]['10%']
    report.h1('平稳性检验结果')
    report.h3('检验结果')
    report.p('Test statistic：' + str(test_statistic))
    report.p(' p-value：' + str(p_value))
    report.p('Number of lags used：' + str(use_lag))
    report.p('Number of observations used for the ADF regression and calculation of the critical values：' + str(nobs))
    report.p('Critical values for the test statistic at the 5 %：' + str(critical_1))
    report.p('Critical values for the test statistic at the 1 %：' + str(critical_5))
    report.p('Critical values for the test statistic at the 10 %：' + str(critical_10))

    '''
    自相关图
    '''
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(111)
    plot_acf(sequence, ax=ax1, fft=True)
    plt.savefig('acf.png')
    report.h3('ACF')
    report.image('acf.png')

    '''
    生成报告
    '''
    report.writeToHtml(reportFileName)