# -*- coding:utf-8 -*-
__author__ = 'anty'

def loadData():
    #data format yyyymmdd latitude,longitude
    return ['19840216101010202020']

def fileFixData(record):
    '''Read weather data from reader r in fixed-width format
        the field widths are:
            4,2,2  YYYYMMDD(data)
            2,2,2   DDMMSS(latitude)
            2,2,2   DDMMSS(longitude)
    The result is a list of values(not tuples)'''

    fields = ((4,int),(2,int),(2,int),   #date
              (2,int),(2,int),(2,int),  #latitude
              (2,int),(2,int),(2,int))  #longitude

    result = []
    #for each record
    for line in record:
        start = 0
        record_t = []

        #for each field in the record
        for (width,target_type) in fields:
            text = line [start:start + width]
            field = target_type(text)
            record_t.append(field)
            start += width

        result.append(record_t)
    print('result is', result)
    return  result

fileFixData(loadData())
