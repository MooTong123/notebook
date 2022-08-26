# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    '''
    载入模块
    '''
    import report_utils
    import db_utils
    import warnings
    import numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    warnings.filterwarnings('ignore')
    report = report_utils.Report()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])
    data_in = data_in.select_dtypes(include=['number'])

    '''
    密度聚类
    '''
    model = DBSCAN(eps=float(params['eps']), min_samples=int(params['min_samples']), algorithm=params['algorithm'],
                   leaf_size=int(params['leaf_size']))

    '''
    模型训练与拟合
    '''
    model.fit(data_in)

    '''
    报告
    '''
    report.h1('DBSCAN密度聚类')
    a = pd.DataFrame([params.keys(), params.values()]).T
    a.columns = ['参数名称', '参数值']
    report.h3('模型参数')
    report.p("输出配置的参数以及参数的取值。")
    report.table(a)

    '''
    输出聚类label
    '''
    fit_label = model.labels_ + 1
    columns = np.append(data_in.columns, params['add_col'])
    data_out = np.column_stack((data_in, fit_label))
    data_out = pd.DataFrame(data_out, columns=columns)

    '''
    饼图结果概况
    '''
    lable_res = pd.Series(fit_label)
    lable_res = lable_res.value_counts()
    lable_res = pd.DataFrame(lable_res)
    lable_res['group'] = lable_res.index
    lable_res['group'] = lable_res['group'].apply(lambda x: 'group' + str(x))
    lable_res = lable_res.reset_index(drop=True)
    fig = plt.figure(figsize=(8, 6))
    a = pd.DataFrame([lable_res['group'], lable_res[0]])
    b = []
    for i in range(len(lable_res)):
        b.append(0.1)
    labels = pd.Series(lable_res['group'])
    fracs = pd.Series(lable_res[0])

    plt.axes(aspect=1)
    plt.pie(x=fracs, labels=labels,
            explode=tuple(b),
            autopct='%3.1f %%',
            shadow=True, labeldistance=1.1,
            startangle=180,
            pctdistance=0.6,
            radius=2.5
            )
    plt.axis('equal')
    plt.title('pie')
    plt.legend(loc=0, bbox_to_anchor=(0.92, 1))
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=6)
    plt.savefig("pie.png")
    plt.show()

    report.h3('饼图结果概况')
    string = '由饼图可以看出，总共分为' + str(len(lable_res)) + '个聚类分群，分别是：'
    for i in range(len(lable_res)):
        if (i < len(lable_res) - 1):
            string = string + lable_res.loc[i][1] + "、"
        else:
            string = string + lable_res.loc[i][1] + "。"
    report.p(string)
    for i in range(len(lable_res)):
        report.p(lable_res.loc[i][1] + "的个数为" + str(lable_res.loc[i][0]) + "，占比为" + str(
            np.around(lable_res.loc[i][0] / lable_res[0].sum(), decimals=2)))
    report.image('pie.png')

    '''
    散点图示例
    '''
    out = data_in
    out['label'] = fit_label
    if len(data_in) < 2000:
        tsne = TSNE(n_components=2, learning_rate=100).fit_transform(out)
        fig = plt.figure()
        x = tsne[:, 0]
        y = tsne[:, 1]
        plt.scatter(x, y, c=out['label'])
        plt.title("scatter\n");
        plt.savefig('scatter.png')
        plt.show()
        report.h3('散点图示例')
        report.p('通过对数据进行降维，在二维空间中展示的聚类结果。')
        report.image('scatter.png')

    ''' 
    将结果写出
    '''
    report.writeToHtml(reportFileName)
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    return model
