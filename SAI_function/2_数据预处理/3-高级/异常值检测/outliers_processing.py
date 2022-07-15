# -- coding: utf-8 --


def execute(conn, inputs, params, outputs, reportFileName):
    # <editable>
    import pyh
    import report_utils
    import db_utils
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    report = report_utils.Report()
    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select *' + ' from ' + inputs['data_in'])
    data_name = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])
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


    names = data_name.columns
    for j in names:
        data_in[j] = outRange(data_in[j])
    '''
    报告
    '''
    report.h1('异常值处理报告')
    report.h3('各列数据的异常值的数量')
    data_sum = pd.DataFrame(data_in.isnull().sum()).reset_index()
    report.table(data_sum)

    '''
    对异常值处理
    '''
    if (params['method'] == 'drop'):
        data_out = data_in.dropna()
    elif (params['method'] == 'Median_interpolation'):
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.fillna(data_in.median())
    elif (params['method'] == 'Mode_interpolation'):
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.fillna(data_in.mode())
    elif (params['method'] == 'slinear'):
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.interpolate(method='slinear')
    elif (params['method'] == 'quadratic'):
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.interpolate(method='quadratic')
    else:
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.fillna(data_in.mean())
    data_out = pd.DataFrame(data_out)

    '''
    将结果写出
    '''
    report.writeToHtml(reportFileName)
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)

    # </editable>
