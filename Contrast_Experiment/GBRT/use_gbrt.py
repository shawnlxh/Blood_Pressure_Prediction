import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

f1 = open('desktop/train_data')
f2 = open('desktop/test_data')

x_train = []
y_train = []
for i in f1:
    i = i.split('\t')
    i = i[0:-1]
    i = [float(i2) for i2 in i]
    y_train.append(i[0])
    x_train.append(i[1:])
    

x_test= []
y_test = []
for i in f2:
    i = i.split('\t')
    i = i[0:-1]
    i = [float(i2) for i2 in i]
    y_test.append(i[0])
    x_test.append(i[1:]) 
    
    
est = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.005, max_depth=3, random_state=0, loss='ls').fit(x_train, y_train)
label = est.predict(x_test)

f3 = open('desktop/label', 'w')
for i in label:
    f3.write(str(i))
    f3.write('\n')
    
f1.close()
f2.close()
f3.close()