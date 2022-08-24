# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    '''
    载入模块
    '''
    import os
    import pyh
    import report_utils
    import db_utils
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from collections import Counter
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    # mulu = os.getcwd()
    warnings.filterwarnings("ignore")
    report = report_utils.Report()

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])
    data_in = data_in.select_dtypes(include=['number'])

    '''
    建立模型
    '''
    if type(params['random_state']) != int:
        random_state = None
    else:
        random_state = int(params['random_state'])

    model = KMeans(n_clusters=int(params['n_clusters']), n_init=int(params['n_init']),
                          max_iter=int(params['max_iter']), init=params['init'], tol=float(params['tol']),
                          random_state=random_state, algorithm=params['algorithm'])

    '''
    模型训练与拟合
    '''
    model.fit(data_in)

    '''
    模型参数
    '''
    report.h1('K-Means算法')
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
    输出聚类中心
    '''
    cluster_centers = pd.DataFrame(np.around(model.cluster_centers_, decimals=6))
    cluster_centers.columns = data_in.columns
    row_name = pd.DataFrame(list(np.arange(1, int(params['n_clusters']) + 1)), columns=['cluster_id'])
    cluster_centers = pd.concat([row_name, cluster_centers], axis=1)
    report.h3('聚类中心坐标：')
    report.table(cluster_centers)

    '''
    饼图结果概况
    '''
    lable_res = pd.Series(fit_label).value_counts()
    lable_res = pd.DataFrame(lable_res)
    lable_res['group'] = lable_res.index
    lable_res['group'] = lable_res['group'].apply(lambda x: 'group' + str(x))
    lable_res = lable_res.reset_index(drop=True)
    fig = plt.figure(figsize=(8, 6))
    a = pd.DataFrame([lable_res['group'], lable_res[0]])
    b = []
    for i in range(len(lable_res)):
        b.append(0.1)
    labels = pd.Series(lable_res['group']).unique()
    fracs = list(Counter(lable_res[0]))

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

    # 设置legend的字体大小
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=6)
    plt.savefig("pie.png")
    plt.show()

    report.h3('饼图结果概况')
    string = '由饼图可以看出，总共分为' + str(len(lable_res)) + '个聚类分群，分别是：'
    for i in range(len(lable_res)):
        if i < len(lable_res) - 1:
            string = string + lable_res.loc[i][1] + "、"
        else:
            string = string + lable_res.loc[i][1] + "。"
    report.p(string)
    for i in range(len(lable_res)):
        report.p(lable_res.loc[i][1] + "的个数为" + str(lable_res.loc[i][0]) + "，占比为" + str(
            np.around(lable_res.loc[i][0] / lable_res[0].sum(), decimals=2)))
    report.image('pie.png')

    '''
    雷达图
    '''
    # data_len = len(cluster_centers)
    # a = cluster_centers.iloc[:, 1:]
    # data = np.array(a.iloc[0])
    # labels = a.columns
    # kinds = list(a.index)
    # b = []
    #
    # for kind in kinds:
    #     b.append('Cluster_grouping_' + str(kind))
    # fig = plt.figure(figsize=(8, 6))
    # for i in range(0, len(cluster_centers)):
    #     data = np.array(a.iloc[i])
    #     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    #     data = np.concatenate((data, [data[0]]))  # 闭合
    #     angles = np.concatenate((angles, [angles[0]]))  # 闭合
    #     ax = fig.add_subplot(111, polar=True)  # polar参数！！
    #     ax.plot(angles, data, linewidth=2.0, label=b[i])  # 画线
    #     ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
    #     ax.set_title("\nradar chart\n")
    #     ax.set_rlim(a.min().min(), a.max().max())  # 设置雷达图的范围
    #     ax.grid(True)
    #     plt.legend(loc='lower right')
    # plt.savefig("Rader.png")
    # plt.show()
    # report.h3('雷达图示例')
    # report.p('雷达图在每个属性上的大小反应的是每个分群中该特征的优势和劣势。')
    # report.image('Rader.png')

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
        plt.title("KMeans聚类\n")
        plt.savefig('scatter.png')
        plt.show()
        report.h3('散点图示例')
        report.p('通过对数据进行降维，在二维空间中展示的聚类结果。')
        report.image('scatter.png')

    '''
    将结果写出
    '''
    # os.chdir(mulu)
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    report.writeToHtml(reportFileName)
    return model
