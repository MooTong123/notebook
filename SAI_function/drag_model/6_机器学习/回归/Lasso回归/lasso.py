# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    '''
    载入模块
    '''
    import pyh
    import db_utils
    import report_utils
    import joblib
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
    from sklearn.linear_model import Lasso

    warnings.filterwarnings("ignore")
    report = report_utils.Report()

    '''
    载入数据
    '''
    x_train = db_utils.query(conn, 'select ' + params['features'] + ' from ' + inputs['data_in'])
    y_train = db_utils.query(conn, 'select ' + params['label'] + ' from ' + inputs['data_in'])

    '''
    建立模型
    '''
    if type(params['random_state']) != int:
        random_state = None
    else:
        random_state = int(params['random_state'])

    model = Lasso(alpha=float(params['alpha']), fit_intercept=bool(params['fit_intercept']),
                  normalize=bool(params['normalize']), max_iter=int(params['max_iter']),
                  precompute=bool(params['precompute']), tol=float(params['tol']), random_state=random_state,
                  positive=bool(params['positive']), selection=params['selection'])

    '''
    模型训练与拟合
    '''
    model.fit(x_train, y_train)
    y_ov = model.predict(x_train)

    '''
    模型参数
    '''
    a = pd.DataFrame([params.keys(), params.values()]).T
    a.columns = ['参数名称', '参数值']
    report.h3('模型参数')
    report.p("需要配置的参数及其取值如下。")
    report.table(a)

    '''
    模型属性
    '''
    report.h3('模型属性')
    fol = params['label'] + '_pred = '
    intercept = (model.intercept_[0] if 'True' == params['fit_intercept'] else 0)
    for i in range(len(model.coef_)):
        if i > 0:
            if model.coef_[i] > 0:
                fol += ' + ' + str(model.coef_[i]) + '*' + x_train.columns[i]
            else:
                fol += ' - ' + str(abs(model.coef_[i])) + '*' + x_train.columns[i]
        elif model.coef_[0] > 0:
            fol += str(intercept) + ' + ' + str(model.coef_[0]) + '*' + x_train.columns[0]
        else:
            fol += str(intercept) + ' - ' + str(abs(model.coef_[0])) + '*' + x_train.columns[0]
    md_info1 = pd.DataFrame({'模型公式': [fol]})
    report.table(md_info1)

    '''
    模型指标
    '''
    report.h3('模型评价指标')
    report.p('模型拟合效果指标如下。')
    md_info = pd.DataFrame()
    md_info['重要指标'] = ['MSE', 'RMSE', 'MAE', 'EVS', 'R-Squared']
    md_info['值'] = ['%.2f' % mean_squared_error(y_train, y_ov), '%.2f' % np.sqrt(mean_squared_error(y_train, y_ov)),
                    '%.2f' % mean_absolute_error(y_train, y_ov), '%.2f' % explained_variance_score(y_train, y_ov),
                    '%.2f' % r2_score(y_train, y_ov)]
    report.table(md_info)

    '''
    模型拟合情况
    '''
    data_out = pd.concat([x_train, y_train, pd.DataFrame({params['add_col']: y_ov})], axis=1)
    data_out[params['add_col']] = data_out.apply(lambda x: '%.2f' % x[params['add_col']], axis=1)
    report.h3('模型拟合情况')
    report.p('取' + str(len(x_train)) + '条数据作为训练集，建立LASSO回归模型，得到的预测值与真实值对比图如下图所示。')
    X_label = [str(i) for i in x_train.index]
    plt.figure(figsize=(6.0, 4.0))
    plt.style.use('ggplot')
    plt.plot(X_label, y_train, marker='*', label='origin')
    plt.plot(X_label, y_ov, marker='.', alpha=0.7, label='prediction')
    if len(x_train) > 10 and (len(x_train) - 1) % 10 < 5:
        plt.xticks(np.linspace(0, np.ceil(len(x_train) / 5) * 5 - 1, 5))
    elif len(x_train) > 10 and (len(x_train) - 1) % 10 > 5:
        plt.xticks(np.linspace(0, np.ceil(len(x_train) / 10) * 10 - 1, 10))
    else:
        plt.xticks(np.linspace(0, len(x_train) - 1, len(x_train)))
    plt.legend()
    plt.title('Fitting of LASSO Regression Model')
    plt.xlabel('index')
    plt.ylabel(y_train.columns.values[0])
    plt.tight_layout()
    plt.savefig('overview.png')
    report.image('overview.png')

    # 保存模型
    model_file = 'model.pkl'
    joblib.dump(model, model_file)

    '''
    生成报告
    '''
    report.writeToHtml(reportFileName)

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)

    return model
