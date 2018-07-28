#encoding=utf-8

import csv

#with open('sh300.csv') as csvfile:
#with open('600783.csv') as csvfile:
with open('HFT_XY_unselected.csv') as csvfile:
    reader = csv.reader(csvfile,delimiter=' ',quotechar='|')
    for row in reader:
           print(type(row))
           print len(row)
           print row
           print '-------------------------'
