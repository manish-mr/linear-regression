import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
from random import randint

# Generate test data
def genData(classType, value):
    xElement = []
    m0 = [[-0.132, 0.320, 1.672, 2.230,  1.217, -0.819,  3.629,  0.8210,  1.808, 0.1700],
          [-0.711, -1.726, 0.139, 1.151, -0.373, -1.573, -0.243, -0.5220, -0.511, 0.5330]]
    m1 = [[-1.169, 0.813, -0.859, -0.608, -0.832, 2.015, 0.173, 1.432,  0.743, 1.0328],
          [2.065, 2.441,  0.247, 1.806,  1.286, 0.928, 1.923, 0.1299, 1.847, -0.052]]
    m0 = np.array(m0)
    m1 = np.array(m1)
    for j in range(value):
        r = randint(0, 9)
        index = math.floor(r)
        index = int(index)
        if classType == 0:
            m = m0[:,[index]]
        else:
            m= m1[:,[index]]
        r1 = np.random.randn(2,1)
        r2 = r1/float(math.sqrt(5))
        element = m+r2
        xElement.append([element[0][0], element[1][0],classType])
    return xElement

train_data = pd.read_csv("classasgntrain1.dat",delim_whitespace=True, header=None)

class_0_data = pd.concat([train_data[0], train_data[1]], axis=1, keys=['x1', 'x2']) #class0
class_1_data = pd.concat([train_data[2], train_data[3]], axis=1, keys=['x1', 'x2']) #class1

class_0_data['class'] = 0
class_1_data['class'] = 1

x_vals = train_data[0].values.tolist() + train_data[2].values.tolist()
y_vals = train_data[1].values.tolist() + train_data[3].values.tolist()

xmin = min(x_vals)
xmax = max(x_vals)
ymin = min(y_vals)
ymax = max(y_vals)

data = class_0_data.append(class_1_data, ignore_index=True)
data['dummy'] = 1
N = len(data)     # 200 for train

X = np.array(data[['dummy','x1','x2']])
y = np.array(data[['class']])

Xt = X.T

Bhat = ((inv(Xt.dot(X))).dot(Xt)).dot(y)
Yhat = X.dot(Bhat)

Yhathard = Yhat > 0.5
Yhathard = np.array(Yhathard, dtype=np.int)
print(Yhathard)

errorCount = 0
for i in range(len(Yhathard)):
    if Yhathard[i][0] != y[i][0]:
        errorCount = errorCount + 1

errorrate_train = errorCount/float(N)
print("Error rate for training data: ",errorrate_train)

# Test data
class_0 = 0
class_1 = 1
no_of_samples = 5000

class_0_data_test = genData(class_0, no_of_samples)
class_1_data_test = genData(class_1, no_of_samples)

for i in range(no_of_samples):
    class_0_data_test[i].append(class_0)
    class_1_data_test[i].append(class_1)

test_data = class_0_data_test + class_1_data_test

x1 = []
x2 = []
y_test = []

for i in range(len(test_data)):
    x1.append(test_data[i][0])
    x2.append(test_data[i][1])
    y_test.append(test_data[i][2])

test_df = pd.DataFrame(x1, columns=['x1'])
test_df['x2'] = x2
test_df['dummy'] = 1
X_test = np.array(test_df[['dummy','x1', 'x2']])
Yhat_test = X_test.dot(Bhat)

#print "Yhat_test: ",Yhat_test

Yhathard_test = Yhat_test > 0.5
Yhathard_test = np.array(Yhathard_test, dtype=np.int)
print(Yhathard_test)

print(Yhathard_test)
errorCount = 0
for i in range(len(Yhathard_test)):
    if Yhathard_test[i][0] != y_test[i]:
        errorCount = errorCount + 1

errorrate_test = errorCount/float(no_of_samples*2)
print("Error rate for test data: ",errorrate_test)

r_0 = np.linspace(xmin, xmax, 100)
r_1 = np.linspace(ymin, ymax, 100)

redptsx = []
redptsy = []
greenptsx = []
greenptsy = []
rndRhat = []
for u in range(len(r_0)):
    for v in range(len(r_1)):
      r_arr = np.array([1,r_0[u],r_1[v]])
      h = r_arr.dot(Bhat)
      if h > .5:
          redptsx.append(r_0[u])
          redptsy.append(r_1[v])
      else:
          greenptsx.append(r_0[u])
          greenptsy.append(r_1[v])


rdf = pd.DataFrame()
gdf = pd.DataFrame()
rdf['rx'] = redptsx
rdf['gx'] = redptsy
gdf['rx'] = greenptsx
gdf['gx'] = greenptsy


plt.scatter(rdf['rx'], rdf['gx'], color='r', label='Class0')
plt.scatter(gdf['rx'], gdf['gx'], color='b', label='Class1')
plt.legend()
plt.show()