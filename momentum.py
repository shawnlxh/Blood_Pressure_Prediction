import numpy as np
import json
import time
import os

with open('Downloads/ALL_USER_HISTORY.json','r') as f:
    data = json.load(f)
    
count = len(data)

train_data = []
for i in range( int(count * 0.8) ):
    train_data.append(data[i])

test_data = []
while i < count:
    test_data.append(data[i])
    i += 1
        
u = np.random.randn(20, 10)*0.05
w = np.random.randn(10, 10)*0.05
r = np.random.randn(10, 1)*0.05
learning_rate = 0.001
lamda = 0.0005
mu = 0.5

def train():
    global u
    global w
    global r
    count = 0
    for num in train_data:
        count += 1
    n = 0
    v1 = 0
    v2 = 0
    v3 = 0
    while n < count:
        du = 0
        dw = 0
        dr = 0
        
        #t = 1
        h0 = np.zeros((1, 10)) 
        i1 = np.asarray(train_data[n][0])
        i1 = i1.reshape(1,20)
        h1 = np.dot(i1, u) + np.dot(h0, w)
        y = np.dot(h1, r)
        j = train_data[n][1][0]
        
        e = y - j
        dl_r = e * h1.T
        dr += dl_r
        dl_h1 = e * r.T
        
        dl_u = np.dot(i1.T, dl_h1) + lamda * u
        dl_w = np.dot(h0.T, dl_h1) + lamda * w
        du += dl_u
        dw += dl_w
        
        #t = 2
        i2 = np.asarray(train_data[n][1])
        i2 = i2.reshape(1,20)
        h2 = np.dot(i2, u) + np.dot(h1, w)
        y = np.dot(h2, r)
        j = train_data[n][2][0]
        
        e = y - j
        dl_r = e * h2.T
        dr += dl_r
        dl_h2 = e * r.T
        
        dl_u = np.dot(i2.T, dl_h2) + lamda * u
        dl_w = np.dot(h1.T, dl_h2) + lamda * w
        du += dl_u
        dw += dl_w
        
        dl_h1 = np.dot(dl_h2, w.T)
        dl_u = np.dot(i1.T, dl_h1) + lamda * u
        dl_w = np.dot(h0.T, dl_h1) + lamda * w
        du += dl_u
        dw += dl_w
        
        #t > 2
        t = 3
        it2 = i1
        it1 = i2
        ht3 = h0
        ht2 = h1
        ht1 = h2
        while t < 7:
            it = np.asarray(train_data[n][t-1])
            it = it.reshape(1,20)
            ht = np.dot(it, u) + np.dot(ht1, w)
            y = np.dot(ht, r)
            j = train_data[n][t][0]
            
            e = y - j
            dl_r = e * ht.T
            dr += dl_r
            dl_ht = e * r.T
            
            dl_u = np.dot(it.T, dl_ht) + lamda * u
            dl_w = np.dot(ht1.T, dl_ht) + lamda * w
            du += dl_u
            dw += dl_w
            
            dl_ht1 = np.dot(dl_ht, w.T)
            dl_u = np.dot(it1.T, dl_ht1) + lamda * u
            dl_w = np.dot(ht2.T, dl_ht1) + lamda * w
            du += dl_u
            dw += dl_w
            
            dl_ht2 = np.dot(dl_ht1, w.T)
            dl_u = np.dot(it2.T, dl_ht2) + lamda * u
            dl_w = np.dot(ht3.T, dl_ht2) + lamda * w
            du += dl_u
            dw += dl_w
            
            t += 1
            it2 = it1
            it1 = it
            ht3 = ht2
            ht2 = ht1
            ht1 = ht
        
        v1 = mu * v1 - learning_rate * du
        u += v1
        v2 = mu * v2 - learning_rate * dw
        w += v2
        v3 = mu * v3 - learning_rate * dr
        r += v3
        n += 1
        
        
def predict():
    count = 0
    for num in test_data:
        count += 1
    all_loss = 0
    n = 0
    while n < count:
        ht1 = np.zeros((1, 10))
        for m in range(6):      
            i = np.asarray(test_data[n][m])
            i = i.reshape(1,20)
            ht = np.dot(i, u) + np.dot(ht1, w)
            ht1 = ht
        y = np.dot(ht, r)
        j = test_data[n][6][0]
        y = y * 148 + 26
        j = j * 148 + 26
        one_loss = abs(y - j)
        all_loss += one_loss
        n += 1
    loss = all_loss / count
    return loss
    
for i in range(1000):
    train()
    
loss = predict()
print loss
os.system('say "Check."')

#5.07619077


    