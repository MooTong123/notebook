# -- coding: utf-8 --
# <editable>
# 在此处添加算法描述
# </editable>
# conn: 数据库连接
# inputs: 输入数据集合，数据类型:list， 存储组件输入节点对应的数据，
#         通过输入节点的key获取数据，例如配置的key为“input1”，
#         那么inputs$input1即为该节点对应的数据表
# params : 参数集合，数据类型:list， 存储，获取的规则与inputs一致。需要注意的是:
#          params中参数的值都是字符类型的，需要在代码中进行数据类型转换，比如:
#          as.integer(params$centers)
# outputs: 存诸规则参见inputs
# reportfileName: 算法运行报告文件的存诸路径
# 返回值(可选): 如果函数用于训练模型，则必须返回模型对象

def execute(conn, inputs, params, outputs, reportFileName):
    # <editable>
    # 载入模块
    import pyh
    import report_utils
    import db_utils
    import pydot
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    import warnings
    from sklearn.tree import DecisionTreeClassifier
    from six import StringIO
    from sklearn import tree
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, classification_report
    from sklearn.preprocessing import label_binarize, Binarizer
    from itertools import cycle

    warnings.filterwarnings("ignore")
    report = report_utils.Report()

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn,
                             'select ' + params['features'] + ',' + params['label'] + ' from ' + inputs['data_in'])
    x_train = data_in.drop(params['label'], 1)
    y_train = data_in[params['label']]
    y_train = y_train.astype(str)
    class_names = y_train.unique()
    # n_classes = len(class_names)
    # y_one_hot = label_binarize(y_train, classes =class_names)
    y_binarize = label_binarize(y_train, classes=class_names)

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

    if params['class_weight'] == 'None':
        class_weight = None
    else:
        class_weight = str(params['class_weight'])

    model = DecisionTreeClassifier(criterion=params['criterion'], splitter=params['splitter'],
                                   max_depth=max_depth, min_samples_split=int(params['min_samples_split']),
                                   min_samples_leaf=int(params['min_samples_leaf']),
                                   min_weight_fraction_leaf=float(params['min_weight_fraction_leaf']),
                                   max_features=max_features,random_state=random_state,
                                   max_leaf_nodes=max_leaf_nodes,class_weight=class_weight,
                                   min_impurity_decrease=float(params['min_impurity_decrease']),
                                   ccp_alpha=float(params['ccp_alpha']))

    '''
    模型训练
    '''
    model.fit(x_train, y_train)

    '''
    模型预测
    '''
    # 用模型进行预测，返回预测值
    y_fit = model.predict(x_train)
    fit_label = pd.DataFrame(y_fit, columns=[params['add_col']])

    # 返回一个数组，数组的元素依次是X预测为各个类别的概率值
    # y_score = model.predict_proba(x_train)

    '''
    输出预测值
    '''
    data_out = pd.concat([x_train, y_train, fit_label], axis=1)

    '''
    保存模型
    '''
    model_file = 'model.pkl'
    joblib.dump(model, model_file, compress=3)

    '''
    报告
    '''
    report.h1('决策树算法')
    a = pd.DataFrame([params.keys(), params.values()]).T
    a.columns = ['参数名称', '参数值']
    report.h3('模型参数')
    report.p("输出配置的参数以及参数的取值。")
    report.table(a)

    model_params = {'模型分类': model.n_classes_, '模型特征数': model.n_features_}
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
    report.writeToHtml(reportFileName)
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

    cm = confusion_matrix(y_train, fit_label)
    n_classes = len(cm)

    if n_classes == 2:
        # 混淆矩阵
        cm = confusion_matrix(y_train, fit_label)
        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]
        acc = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        model_params = dict()
        model_params['accuracy'] = np.around(acc, decimals=2)
        model_params['precision'] = np.around(precision, decimals=2)
        model_params['recall'] = np.around(recall, decimals=2)
        model_params['f1'] = np.around(f1, decimals=2)
        a = pd.DataFrame([model_params.keys(), model_params.values()]).T
        a.columns = ['指标', '值']
        report.h3('模型评价指标')
        report.table(a)
        # print(acc)
        # print(precision)
        # print(recall)
        # print(f1)

    if n_classes > 2:
        # binarizer = Binarizer(threshold=0.5)
        # y_score = # binarizer.transform(y_score)
        target_names = class_names
        a = classification_report(y_train, fit_label, target_names=target_names)
        b = a.split('\n')
        res = []
        for bb in b:
            if bb != '':
                z = []
                c = bb.split('  ')
                for cc in c:
                    if cc != '':
                        z.append(cc.strip())
                res.append(z)

        res_table = pd.DataFrame(res[1:])
        res_table.columns = ['index', 'precision', 'recall', 'f1-score', 'support']
        report.h3('模型评价指标')
        report.table(res_table)

    '''
    绘制混淆矩阵图
    '''
    cm = confusion_matrix(y_train, fit_label)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x),
                         size='large',
                         horizontalalignment='center',
                         verticalalignment='center')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predict Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('cm_img.png')
    plt.show()
    report.h3('混淆矩阵')
    report.p("如下所示混淆矩阵图：")
    report.image('cm_img.png')

    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_dot('model.dot')
    graph[0].write_png('model.png')
    report.h3('决策树图')
    report.image('model.png')

    '''
    绘制ROC曲线
    fpr：假正例率
    tpr：真正例率
    '''
    # colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    y_fit = label_binarize(y_fit, classes=class_names)
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_binarize.ravel(), y_fit.ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 4))
        lw = 2
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.fill_between(fpr, tpr, alpha=0.2, color='b')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC and AUC')
        plt.legend(loc="lower right")
        plt.savefig('roc.png')
        plt.show()
        report.h3('ROC图')
        report.p("如下图所示：AUC所占的面积是" + str(np.around(roc_auc, decimals=2)))
        report.image('roc.png')

    if n_classes == 2:
        fpr, tpr, _ = precision_recall_curve(y_binarize.ravel(), y_fit.ravel())
        # roc_auc = auc(fpr, tpr)
        fpr[0] = 0
        plt.figure(figsize=(8, 4))
        lw = 2
        plt.plot(fpr, tpr, label='PR')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.fill_between(fpr, tpr, alpha=0.2, color='b')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('precision')
        plt.ylabel('recall')
        plt.title('PR')
        plt.legend(loc="lower right")
        plt.savefig('pr.png')
        plt.show()
        report.h3('Precision-Recall图')
        report.image('pr.png')

    if n_classes > 2:
        fpr, tpr, thresholds = roc_curve(y_binarize.ravel(), y_fit.ravel())
        auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 4))
        lw = 2
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.fill_between(fpr, tpr, alpha=0.2, color='b')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC and AUC')
        plt.legend(loc="lower right")
        plt.savefig('roc.png')
        plt.show()
        report.h3('ROC图')
        report.p("如下图所示：AUC所占的面积是" + str(np.around(auc, decimals=2)))
        report.image('roc.png')
    report.writeToHtml(reportFileName)

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    return model
    #<editable>
