import numpy as np
import json
import time

with open('Downloads/ALL_USER_HISTORY.json','r') as f:
    data = json.load(f)
    
count = len(data)

train_data = []
for i in range( int(count * 0.8) ):
    for k in range(6):
        data[i][k].append(1)
    train_data.append(data[i])

test_data = []
i += 1
while i < count:
    for k in range(6):
        data[i][k].append(1)
    test_data.append(data[i])
    i += 1
        
u = np.random.randn(21, 10)*0.05
w = np.random.randn(10, 10)*0.05
r = np.random.randn(10, 1)*0.05
learning_rate = 0.001
lamda = 0.0005
beta1 = 0.5
beta2 = 0.3
p = 0.5

def f(x):    #tanh
	output = ( np.exp(x) - np.exp(-x) ) /( np.exp(x) + np.exp(-x) )
	return output

def train():
    global u
    global w
    global r
    count = 0
    for num in train_data:
        count += 1
    n = 0
    m1 = 0
    v1 = 0
    m2 = 0
    v2 = 0
    m3 = 0
    v3 = 0
    while n < count:
        du = 0
        dw = 0
        dr = 0
        
        #t = 1
        h0 = np.zeros((1, 10)) 
        i1 = np.asarray(train_data[n][0])
        i1 = i1.reshape(1,21)
        a1 = np.dot(i1, u) + np.dot(h0, w)
        h1 = f(a1)
        u1 = (np.random.rand(*h1.shape) < p) / p
        h1 *= u1
        y = np.dot(h1, r)
        j = train_data[n][1][0]
        
        if j != 0:
            e = y - j
            dl_r = e * h1.T
            dr += dl_r
            dl_h1 = e * r.T
        
            mid = (dl_h1) * (1 - h1 * h1) * u1
            dl_u = np.dot(i1.T, mid) + lamda * u
            dl_w = np.dot(h0.T, mid) + lamda * w
            du += dl_u
            dw += dl_w
        
        #t = 2
        i2 = np.asarray(train_data[n][1])
        i2 = i2.reshape(1,21)
        a2 = np.dot(i2, u) + np.dot(h1, w)
        h2 = f(a2)
        h1 *= u1
        y = np.dot(h2, r)
        j = train_data[n][2][0]
        
        if j != 0:
            e = y - j
            dl_r = e * h2.T
            dr += dl_r
            dl_h2 = e * r.T
        
            mid = (dl_h2) * (1 - h2 * h2) * u1
            dl_u = np.dot(i2.T, mid) + lamda * u
            dl_w = np.dot(h1.T, mid) + lamda * w
            du += dl_u
            dw += dl_w
        
            dl_h1 = np.dot(dl_h2, w.T)
            mid = (dl_h1) * (1 - h1 * h1) * u1
            dl_u = np.dot(i1.T, mid) + lamda * u
            dl_w = np.dot(h0.T, mid) + lamda * w
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
            it = it.reshape(1,21)
            at = np.dot(it, u) + np.dot(ht1, w)
            ht = f(at)
            h1 *= u1
            y = np.dot(ht, r)
            j = train_data[n][t][0]
            
            if j != 0:
                e = y - j
                dl_r = e * ht.T
                dr += dl_r
                dl_ht = e * r.T
                
                mid = (dl_ht) * (1 - ht * ht) * u1
                dl_u = np.dot(it.T, dl_ht) + lamda * u
                dl_w = np.dot(ht1.T, dl_ht) + lamda * w
                du += dl_u
                dw += dl_w
            
                dl_ht1 = np.dot(dl_ht, w.T)
                mid = (dl_ht1) * (1 - ht1 * ht1) * u1
                dl_u = np.dot(it1.T, dl_ht1) + lamda * u
                dl_w = np.dot(ht2.T, dl_ht1) + lamda * w
                du += dl_u
                dw += dl_w
            
                dl_ht2 = np.dot(dl_ht1, w.T)
                mid = (dl_ht2) * (1 - ht2 * ht2) * u1
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
            
        u = u - learning_rate * du
        w = w - learning_rate * dw
        r = r - learning_rate * dr
        
        #m1 = beta1 * m1 + (1 - beta1) * du
        #v1 = beta2 * v1 + (1 - beta2) * (du ** 2)
        #u = u - learning_rate * m1 / (np.sqrt(v1) + 1e-7)
        
        #m2 = beta1 * m2 + (1 - beta1) * dw
        #v2 = beta2 * v2 + (1 - beta2) * (dw ** 2)
        #w = w - learning_rate * m2 / (np.sqrt(v2) + 1e-7)
        
        #m3 = beta1 * m3 + (1 - beta1) * dr
        #v3 = beta2 * v3 + (1 - beta2) * (dr ** 2)
        #r = r - learning_rate * m3 / (np.sqrt(v3) + 1e-7)
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
            i = i.reshape(1,21)
            at = np.dot(i, u) + np.dot(ht1, w)
            ht = f(at)
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

best_loss = 100    
for i in range(3000):
    train()
    loss = predict()
    print loss
    #if loss < best_loss:
        #best_loss = loss
        #print loss
    
loss = predict()
print loss

#learning_rate = 0.001    lamda = 0.0005 
   


    