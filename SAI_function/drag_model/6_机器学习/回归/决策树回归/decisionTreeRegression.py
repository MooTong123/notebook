# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    '''
    载入模块
    '''
    import pyh
    import report_utils
    import db_utils
    import pydot
    import joblib
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from itertools import cycle
    from six import StringIO
    from sklearn import tree
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score

    warnings.filterwarnings("ignore")
    report = report_utils.Report()

    '''
    选择目标数据
    '''
    x_train = db_utils.query(conn, 'select ' + params['features'] + ' from ' + inputs['data_in'])
    y_train = db_utils.query(conn, 'select ' + params['label'] + ' from ' + inputs['data_in'])

    '''
    构建决策树模型
    '''
    if type(params['max_depth']) != int:
        max_depth = None
    else:
        max_depth = int(params['max_depth'])

    if params['max_features'] == 'None':
        max_features = None
    else:
        max_features = str(params['max_features'])

    if type(params['random_state']) != int:
        random_state = None
    else:
        random_state = int(params['random_state'])

    if type(params['max_leaf_nodes']) != int:
        max_leaf_nodes = None
    else:
        max_leaf_nodes = int(params['max_leaf_nodes'])

    model = DecisionTreeRegressor(criterion=params['criterion'], splitter=params['splitter'],
                                  max_depth=max_depth, min_samples_split=int(params['min_samples_split']),
                                  min_samples_leaf=int(params['min_samples_leaf']),
                                  min_weight_fraction_leaf=float(params['min_weight_fraction_leaf']),
                                  max_features=max_features, random_state=random_state,
                                  max_leaf_nodes=max_leaf_nodes, ccp_alpha=float(params['ccp_alpha']),
                                  min_impurity_decrease=float(params['min_impurity_decrease']))

    '''
    模型训练
    '''
    model.fit(x_train, y_train)

    '''
    模型预测
    '''
    # 用模型进行预测，返回预测值
    y_fit = model.predict(x_train)

    '''
    模型参数
    '''
    report.h1('决策树回归算法')
    a = pd.DataFrame([params.keys(), params.values()]).T
    a.columns = ['参数名称', '参数值']
    report.h3('模型参数')
    report.p("输出配置的参数以及参数的取值。")
    report.table(a)

    '''
    模型属性
    '''

    model_params = {}
    model_params['max_features_'] = model.max_features_
    model_params['n_features_'] = model.n_features_
    model_params['n_outputs_'] = model.n_outputs_
    a = pd.DataFrame([model_params.keys(), model_params.values()]).T
    a.columns = ['参数名称', '参数值']
    report.h3('模型属性')
    report.p("输出模型的属性信息。")
    report.table(a)

    a = pd.DataFrame([x_train.columns, np.around(model.feature_importances_, decimals=4)]).T
    a.columns = ['特征', 'feature importance']
    a = a.sort_values('feature importance', ascending=False)
    report.p("输出模型的特征的重要性信息：")
    report.table(a)

    a.index = [a['特征'].unique()]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)

    a.plot(kind='barh', figsize=(10, 6), ).get_figure()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig('bar')
    plt.show()
    report.image('bar.png')

    '''
    模型指标
    '''
    report.h3('模型评价指标')
    report.p('模型拟合效果指标如下。')
    md_info = pd.DataFrame()
    md_info['重要指标'] = ['MSE', 'RMSE', 'MAE', 'EVS', 'R-Squared']
    md_info['值'] = ['%.2f' % mean_squared_error(y_train, y_fit), '%.2f' % np.sqrt(mean_squared_error(y_train, y_fit)),
                    '%.2f' % mean_absolute_error(y_train, y_fit), '%.2f' % explained_variance_score(y_train, y_fit),
                    '%.2f' % r2_score(y_train, y_fit)]
    report.table(md_info)
    report.writeToHtml(reportFileName)

    '''
    模型拟合情况
    '''
    data_out = pd.concat([x_train, y_train, pd.DataFrame({params['add_col']: list(y_fit)})], axis=1)
    data_out[params['add_col']] = data_out.apply(lambda x: '%.2f' % x[params['add_col']], axis=1)
    report.h3('模型拟合情况')
    report.p('取' + str(len(x_train)) + '条数据作为训练集，建立CART数回归模型，得到的预测值与真实值对比图如下图所示。')

    x_label = [str(i) for i in x_train.index]
    plt.figure(figsize=(6.0, 4.0))
    plt.style.use('ggplot')
    plt.plot(x_label, y_train, marker='*', label='origin')
    plt.plot(x_label, y_fit, marker='.', alpha=0.7, label='prediction')
    if len(x_train) > 10 and (len(x_train) - 1) % 10 < 5:
        plt.xticks(np.linspace(0, np.ceil(len(x_train) / 5) * 5 - 1, 5))
    elif len(x_train) > 10 and (len(x_train) - 1) % 10 > 5:
        plt.xticks(np.linspace(0, np.ceil(len(x_train) / 10) * 10 - 1, 10))
    else:
        plt.xticks(np.linspace(0, len(x_train) - 1, len(x_train)))
    plt.legend()
    plt.title('Fitting of CART Tree Model')
    plt.xlabel('index')
    plt.ylabel(y_train.columns.values[0])
    plt.tight_layout()
    plt.savefig('overview.png')
    report.image('overview.png')

    '''
    CART决策树图
    '''
    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_dot('model.dot')
    graph[0].write_png('model.png')
    report.h3('CART决策树图')
    report.image('model.png')

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
