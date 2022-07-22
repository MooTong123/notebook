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
    import warnings
    import joblib
    from sklearn.svm import SVC
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
    if params['class_weight'] == 'None':
        class_weight = None
    else:
        class_weight = str(params['class_weight'])

    if type(params['random_state']) != int:
        random_state = None
    else:
        random_state = int(params['random_state'])

    model = SVC(C=float(params['C']), kernel=params['kernel'], degree=int(params['degree']), gamma=params['gamma'],
                tol=float(params['tol']), class_weight=class_weight, max_iter=int(params['max_iter']),
                random_state=random_state, decision_function_shape=params['decision_function_shape'])

    '''
    模型训练
    '''
    model.fit(x_train, y_train)

    '''
    模型预测
    '''
    y_fit = model.predict(x_train)
    # y_score = model.predict_proba(x_train)
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
    report.h1('SVM分类算法')
    model_params = {}
    model_params['C'] = params['C']
    model_params['kernel'] = params['kernel']
    model_params['degree'] = params['degree']
    model_params['gamma'] = params['gamma']
    model_params['decision_function_shape'] = params['decision_function_shape']
    model_params['tol'] = params['tol']
    model_params['class_weight'] = params['class_weight']
    model_params['max_iter'] = params['max_iter']
    model_params['random_state'] = params['random_state']

    a = pd.DataFrame([model_params.keys(), model_params.values()]).T
    a.columns = ['参数名称', '参数值']
    report.h3('模型参数')
    report.p("输出配置的参数以及参数的取值。")
    report.table(a)

    model_params = {}
    model_params['class_weight_'] = model.class_weight_
    model_params['classes_'] = model.classes_
    model_params['fit_status_'] = model.fit_status_
    model_params['intercept_'] = model.intercept_
    model_params['n_features_in_'] = model.n_features_in_
    model_params['support_'] = model.support_
    model_params['support_vectors_'] = model.support_vectors_
    model_params['n_support_'] = model.n_support_
    model_params['shape_fit_'] = model.shape_fit_

    a = pd.DataFrame([model_params.keys(), model_params.values()]).T
    a.columns = ['参数名称', '参数值']
    report.h3('模型属性')
    report.p("输出模型的属性信息。")
    report.table(a)

    cm = confusion_matrix(y_train, fit_label)
    n_classes = len(cm)

    if n_classes == 2:
        cm = confusion_matrix(y_train, fit_label)
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

    if n_classes > 2:
        # binarizer = Binarizer(threshold=0.5)
        # y_score = binarizer.transform(y_score)
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

    '''
    绘制ROC曲线
    fpr：假正例率
    tpr：真正例率
    '''
    # setup plot details
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
