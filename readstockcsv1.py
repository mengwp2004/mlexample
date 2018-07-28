#encoding=utf-8

import matplotlib.pyplot as plt
import csv
import numpy as np
#with open('sh300.csv') as csvfile:
#with open('600783.csv') as csvfile:
from  array import *

datalen = 1000

def getdata():
  X= np.zeros((datalen,334))
  y = array('f')
  i =0

  with open('HFT_XY_unselected.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row  in reader:
           
           #l= row.keys()
           #print(len(l))
           #print(l)
           #print(row['realY'])
           #print(row['predictY'])
           y.append(float(row['realY']))
           del row['realY']
           row1 = array('f')
           for unit in row.values():
               #print(type(unit))
               row1.append((float(unit)))
           #print(type(row1))
           #m = np.asarray(row1)
           #if (i == 1) :
           #   X= row1
           #   i =i+1
           #   continue
           #print("====")
           #print(type(row1))
           #print(type(X))
           #print(len(X))
           X[i] = np.asarray(row1)
           #np.vstack((y,float(row['realY'])))
           #print(type(X))
           #X.append(row1)
           #y.append(float(row['realY']))
           #print(row.values())
           #print(type(row))
           #print len(row)
           #print row
           i = i+1
           #print '-------------------------'
           if i>= datalen :
                break

  return X,np.asarray(y)

X,y=getdata()

print(X)
print(X.shape)
print(type(X))


print(y)
print(y.shape)
print(type(y))
#print(y.shape)


plt.figure()
plt.plot(y)
plt.show()
