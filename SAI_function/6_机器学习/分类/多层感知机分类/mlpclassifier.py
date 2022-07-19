# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    '''
    	载入模块
    	'''
    import pyh
    import report_utils
    import db_utils
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    import warnings
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, classification_report
    from sklearn.preprocessing import label_binarize
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
    # y_one_hot = label_binarize(y_train, classes=class_names)
    y_binarize = label_binarize(y_train, classes=class_names)

    '''
    构造模型 
    '''
    # TODO
    if type(params['random_state']) != int:
        random_state = None
    else:
        random_state = int(params['random_state'])

    if params['early_stopping']:
        early_stopping = True
    else:
        early_stopping = False

    model = MLPClassifier(hidden_layer_sizes=(int(params['hidden_layer_sizes']),), activation=params['activation'],
                          solver=params['solver'], max_iter=int(params['max_iter']), random_state=random_state,
                          learning_rate_init=float(params['learning_rate_init']), alpha=float(params['alpha']),
                          learning_rate=params['learning_rate'], tol=float(params['tol']),
                          n_iter_no_change=int(params['n_iter_no_change']), early_stopping=early_stopping,
                          validation_fraction=float(params['validation_fraction']))

    '''
    模型训练
    '''
    model.fit(x_train, y_train)

    '''
    模型预测
    '''
    y_fit = model.predict(x_train)
    y_score = model.predict_proba(x_train)
    fit_label = pd.DataFrame(y_fit, columns=[params['add_col']])

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
    # TODO
    report.h1('多层感知机分类算法')
    model_params = {}
    model_params['隐藏层神经元个数'] = params['hidden_layer_sizes']
    model_params['隐层激活函数'] = params['activation']
    model_params['求解器'] = params['solver']
    model_params['最大迭代'] = params['max_iter']
    model_params['L2惩罚（正则化项）参数'] = params['alpha']
    model_params['权重更新的学习率'] = params['learning_rate']
    model_params['初始学习率'] = params['learning_rate_init']

    a = pd.DataFrame([model_params.keys(), model_params.values()]).T
    a.columns = ['参数名称', '参数值']
    report.h3('模型参数')
    report.p("输出配置的参数以及参数的取值。")
    report.table(a)
    report.writeToHtml(reportFileName)
    model_params = {}
    model_params['classes_'] = model.classes_
    model_params['loss_'] = np.around(model.loss_, decimals=6)
    model_params['t_'] = model.t_
    a = pd.DataFrame([model_params.keys(), model_params.values()]).T
    a.columns = ['参数名称', '参数值']
    report.h3('模型属性')
    report.p("输出模型的属性信息。")
    report.table(a)
    report.writeToHtml(reportFileName)

    cm = confusion_matrix(y_train, fit_label)  # 混淆矩阵
    n_classes = len(cm)
    n_classes

    if n_classes == 2:
        cm = confusion_matrix(y_train, fit_label)  # 混淆矩阵
        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]
        acc = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        model_params = {}
        model_params['accuracy'] = np.around(acc, decimals=2)
        model_params['precision'] = np.around(precision, decimals=2)
        model_params['recall'] = np.around(recall, decimals=2)
        model_params['f1'] = np.around(f1, decimals=2)

        a = pd.DataFrame([model_params.keys(), model_params.values()]).T
        a.columns = ['指标', '值']

        report.h3('模型评价指标')
        report.table(a)
        report.writeToHtml(reportFileName)
        print(acc)
        print(precision)
        print(recall)
        print(f1)

    if n_classes > 2:
        from sklearn import preprocessing
        import numpy as np
        binarizer = preprocessing.Binarizer(threshold=0.5)
        y_score = binarizer.transform(y_score)
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
    cm = confusion_matrix(y_train, fit_label)  # 混淆矩阵
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

    '''
    绘制ROC曲线
    fpr：假正例率
    tpr：真正例率
    '''
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
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
        roc_auc = auc(fpr, tpr)
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
        # report.p("如下图所示：AUC所占的面积是"+str(np.around(roc_auc,decimals=2)))
        report.image('pr.png')

    if n_classes > 2:
        # print('调用函数auc：', metrics.roc_auc_score(y_binarize, y_score, average='micro'))
        # 2、手动计算micro类型的AUC
        # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
        fpr, tpr, thresholds = roc_curve(y_binarize.ravel(), y_fit.ravel())
        auc = auc(fpr, tpr)
        print('手动计算auc：', auc)  # 绘图
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
