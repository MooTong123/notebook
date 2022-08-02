# -- coding: utf-8 --


def execute(conn, inputs, params, outputs, reportFileName):
    # <editable>
    import pyh
    import report_utils
    import db_utils
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select ' + params['columns'] + ' from ' + inputs['data_in'])

    '''
    异常值
    '''
    def outRange(Ser1):
        QL = Ser1.quantile(float(params['upper_quantile']))
        QU = Ser1.quantile(float(params['lower_quantile']))
        IQR = QU - QL
        Ser1.loc[Ser1 > (QU + 1.5 * IQR)] = None
        Ser1.loc[Ser1 < (QL - 1.5 * IQR)] = None
        return Ser1

    for j in data_in.columns:
        data_in[j] = outRange(data_in[j])

    data_out = pd.DataFrame(data_in.isnull().sum(), columns=['count'], index=data_in.columns).reset_index()

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)

    # </editable>

