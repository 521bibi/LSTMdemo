#!/env/Scripts python
# @File    : reader.py
# @Time    : 2019/4/4 15:14
# @Author  : Bb
# -*- coding: utf-8 -*-

import pandas as pd

class XLSReader(object):
    def __init__(self):
        file  = 'E:\pyProjects\LSTMdemo\data\LSTMdemo.xlsx'
        self.xls_read = pd.read_excel(file,header=None,parse_dates=[0])

    def read(self):
        xls_data = self.xls_read
        parse_data = self.parse_xls(xls_data)
        return parse_data

    def parse_xls(self,content):
        parse_data = content.iloc[1:,:4]
        # parse_data[0] = pd.to_datetime(parse_data[0], format="%Y/%m/%d %H:%M:%S")
        parse_data.set_index(0, inplace=True)
        return parse_data

if __name__ == '__main__':
    reader = XLSReader()
    datas = reader.read()
    print(datas)