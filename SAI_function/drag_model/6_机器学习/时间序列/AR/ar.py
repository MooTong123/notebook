# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    '''
    载入模块
    '''
    import db_utils
    import report_utils
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.ar_model import AutoReg
    from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score

    warnings.filterwarnings('ignore')
    report = report_utils.Report()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    '''
    选择目标数据
    '''
    data = db_utils.query(conn, 'select ' + params['data'] + ' from ' + inputs['data_in'])
    time = db_utils.query(conn, 'select ' + params['time'] + ' from ' + inputs['data_in'])
    time[params['time']] = pd.to_datetime(time[params['time']])

    '''
    训练和拟合模型
    '''
    lags = [int(i) for i in params['lags'].split(',')]
    max_lag = max(lags)

    model = AutoReg(data, lags=lags).fit()

    '''
    模型拟合
    '''

    ts_fit = model.predict(start=1, end=len(time)).reset_index(drop=True)
    fit_value = pd.DataFrame(ts_fit, columns=[params['add_col']])

    '''
    输出预测值
    '''
    data_out = pd.concat([time, data, fit_value], axis=1)

    '''
    模型参数
    '''
    a = pd.DataFrame([params.keys(), params.values()]).T
    a.columns = ['参数名称', '参数值']
    report.h3('模型参数')
    report.p("需要配置的参数及其取值如下。")
    report.table(a)

    '''
    模型具体信息
    '''
    model_params = {}
    model_params['aic'] = model.aic
    model_params['hqic'] = model.hqic
    model_params['bic'] = model.bic
    model_params['ar_lags'] = model.ar_lags
    model_params['df_model'] = model.df_model

    a = pd.DataFrame([model_params.keys(), model_params.values()]).T
    a.columns = ['参数名称', '参数值']
    report.h3('模型属性')
    report.p("输出模型的属性信息。")
    report.table(a)

    '''
    模型检验图
    '''
    plt.figure(figsize=(6.0, 4.0))
    model.plot_diagnostics()
    plt.tight_layout()
    plt.savefig('diagnostics.png')
    report.h3('模型检验图')
    report.p('模型各项检验信息如下图。')
    report.image('diagnostics.png')

    '''
    模型指标
    '''

    report.h3('模型评价指标')
    report.p('模型拟合效果指标如下。')
    md_info = pd.DataFrame()
    md_info['重要指标'] = ['MSE', 'RMSE', 'MAE', 'EVS', 'R-Squared']
    md_info['值'] = ['%.4f' % mean_squared_error(data[max_lag:], ts_fit[max_lag:]),
                    '%.4f' % np.sqrt(mean_squared_error(data[max_lag:], ts_fit[max_lag:])),
                    '%.4f' % mean_absolute_error(data[max_lag:], ts_fit[max_lag:]),
                    '%.4f' % explained_variance_score(data[max_lag:], ts_fit[max_lag:]),
                    '%.4f' % r2_score(data[max_lag:], ts_fit[max_lag:])]
    report.table(md_info)

    '''
    模型拟合情况
    '''
    report.h3('模型拟合情况')
    report.p('建立AR模型,设置lags为：' + (params['lags']) + '，得到的预测值与实际值对比图如下图所示。')
    plt.figure(figsize=(6.0, 4.0))
    plt.style.use('ggplot')
    plt.plot(time, data, marker='*', label='origin')
    plt.plot(time, ts_fit[:len(time)], marker='.', alpha=0.7, label='fit')
    plt.legend()
    plt.title('Fitting of AR Model')
    plt.xlabel('date')
    plt.ylabel(data.columns.values[0])
    plt.tight_layout()
    plt.savefig('overview.png')
    report.image('overview.png')

    # '''
    # 模型预测情况
    # '''
    # pre_index = pd.date_range(start=time[params['time']][len(time) - 1], periods=int(params['periods']),
    #                           freq=time[params['time']][len(time) - 1] - time[params['time']][len(time) - 2])
    # report.h3('模型预测情况')
    # report.p('设置预测周期数为' + str(int(params['periods'])) + '，得到的预测值如下图所示。')
    # pre_data = pd.DataFrame({timeseries.columns.values[0] + '_preValue': ts_pred[len(time):]})
    # pre_data[timeseries.columns.values[0] + '_preValue'] = pre_data.apply(
    #     lambda x: '%.2f' % x[timeseries.columns.values[0] + '_preValue'], axis=1)
    # pre_data.index = pre_index
    # plt.figure(figsize=(6.0, 4.0))
    # plt.style.use('ggplot')
    # plt.title('Prediction of ARIMA Model')
    # plt.plot(time, timeseries, '.-', label='origin')
    # plt.plot(pre_index, ts_pred[len(time):], '.-', label='prediction')
    # plt.legend()
    # plt.xlabel('date')
    # plt.ylabel(timeseries.columns.values[0])
    # plt.tight_layout()
    # plt.savefig('pre_view.png')
    # report.image('pre_view.png')
    # fit_data = time
    # pre_data = timeseries

    '''
    生成报告
    '''
    report.writeToHtml(reportFileName)

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
