#encoding=utf-8

import numpy as np
from scipy.stats import pearsonr
np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)


print( "Lower noise", pearsonr(x, x + np.random.normal(0, 1, size)))
print( "Higher noise", pearsonr(x, x + np.random.normal(0, 10, size)))

#结果如下,每一行第一个元素代表相关系数，第二个代表p-value,
#相关系数越大代表越相关
#p-value，　小于<0.05　　代表拒绝无效假设

#('Lower noise', (0.71824836862138408, 7.3240173129983507e-49))
#('Higher noise', (0.057964292079338155, 0.31700993885324752))

x = np.random.uniform(-1, 1, 100000)
print( pearsonr(x, x**2))


