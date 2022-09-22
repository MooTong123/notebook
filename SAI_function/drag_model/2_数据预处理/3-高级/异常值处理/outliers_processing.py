# -- coding: utf-8 --
def execute(conn, inputs, params, outputs, reportFileName):
    import pyh
    import report_utils
    import db_utils
    import warnings
    import pandas as pd

    warnings.filterwarnings('ignore')
    report = report_utils.Report()
    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])
    features = data_in[params['features'].split(',')]
    '''
    找出缺失值
    '''

    def outRange(Ser1):
        QL = Ser1.quantile(float(params['upper_quantile']))
        QU = Ser1.quantile(float(params['lower_quantile']))
        IQR = QU - QL
        Ser1.loc[Ser1 > (QU + 1.5 * IQR)] = None
        Ser1.loc[Ser1 < (QL - 1.5 * IQR)] = None
        return Ser1

    names = features.columns
    for j in names:
        features[j] = outRange(features[j])

    data_out = data_in.copy()

    for feature in features.columns:
        data_out[feature] = features[feature]

    '''
    报告
    '''
    report.h1('异常值处理报告')
    report.h3('各列数据的异常值的数量')
    data_sum = pd.DataFrame(features.isnull().sum()).reset_index()
    report.table(data_sum)

    '''
    对异常值处理
    '''
    if params['method'] == 'drop':
        data_out = data_out.dropna()
    elif params['method'] == 'Median_interpolation':
        data_out = data_out.select_dtypes(include=['number'])
        data_out = data_out.fillna(data_out.median())
    elif params['method'] == 'Mode_interpolation':
        data_out = data_out.select_dtypes(include=['number'])
        data_out = data_out.fillna(data_out.mode())
    elif params['method'] == 'slinear':
        data_out = data_out.select_dtypes(include=['number'])
        data_out = data_out.interpolate(method='slinear')
    elif params['method'] == 'quadratic':
        data_out = data_out.select_dtypes(include=['number'])
        data_out = data_out.interpolate(method='quadratic')
    else:
        data_out = data_out.select_dtypes(include=['number'])
        data_out = data_out.fillna(data_out.mean())

    '''
    将结果写出
    '''
    report.writeToHtml(reportFileName)
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
