import numpy as np
import json
import time

with open('Downloads/trim_ALL_USER_HISTORY_qain&3_15_interval(4).json','r') as f1:
    data = json.load(f1)
    
with open('Downloads/trim_new_total.json','r') as f2:
    data2 = json.load(f2)
    
count = len(data)

ob_data_train = []
user_data_train = []

for line in data2:
    count1 = len(line)
    ob = []
    user = []
    for k in range(count1):
        temp1 = line[k][0:17]
        temp1.append(1)
        ob.append(temp1)
        temp2 = line[k][17:21]
        temp2.append(1)
        user.append(temp2)
    ob_data_train.append(ob)
    user_data_train.append(user)

a = 0
for i in range( int(count * 0.8) ):
    a += 1

ob_data_test = []
user_data_test = []
i += 1
while i < count:
    count1 = len(data[i])
    ob = []
    user = []
    for k in range(count1):
        temp1 = data[i][k][0:17]
        temp1.append(1)
        ob.append(temp1)
        temp2 = data[i][k][17:21]
        temp2.append(1)
        user.append(temp2)
    ob_data_test.append(ob)
    user_data_test.append(user)
    i += 1
        
u = np.random.randn(18, 10)*0.05
w = np.random.randn(10, 10)*0.05
r = np.random.randn(10, 1)*0.05
v = np.random.randn(5,1)*0.05
h = np.random.randn(1,5)*0.05
learning_rate = 0.001
lamda = 0

def f(x):    #tanh
	output = ( np.exp(x) - np.exp(-x) ) /( np.exp(x) + np.exp(-x) )
	return output

def train():
    global u
    global w
    global r
    global v
    global h
    count = 0
    for num in ob_data_train:
        count += 1
    n = 0
    while n < count:
        count1 = len(ob_data_train[n])
        
        du = 0
        dw = 0
        dr = 0
        dv = 0
        dh = 0
        
        #t = 1
        h0 = np.zeros((1, 10)) 
        i1 = np.asarray(ob_data_train[n][0])
        i1 = i1.reshape(1,18)
        ii1 = np.asarray(user_data_train[n][0])
        ii1 = ii1.reshape(1,5)
        
        a1 = np.dot(i1, u) + np.dot(h0, w)
        h1 = f(a1)
        j = ob_data_train[n][1][0]
        if j != 0:
            if ii1[0][0] != None:
                y = np.dot(h1, r) + np.dot(ii1, v)
            
                e = y - j
                dl_r = e * h1.T
                dr += dl_r
                dl_h1 = e * r.T
            
                dl_v = e * ii1.T
                dv += dl_v
            else:
                y = np.dot(h1, r) + np.dot(h, v)
                
                e = y - j
                dl_r = e * h1.T
                dr += dl_r
                dl_h1 = e * r.T
            
                dl_v = e * h.T
                dv += dl_v
                
                dl_h = v.T * e
                dh += dl_h
            
            mid = (dl_h1) * (1 - h1 * h1)
            dl_u = np.dot(i1.T, mid) + lamda * u
            dl_w = np.dot(h0.T, mid) + lamda * w
            du += dl_u
            dw += dl_w
        
        if count1 > 2:
            #t = 2
            i2 = np.asarray(ob_data_train[n][1])
            i2 = i2.reshape(1,18)
            ii2 = np.asarray(user_data_train[n][1])
            ii2 = ii2.reshape(1,5)
            
            a2 = np.dot(i2, u) + np.dot(h1, w)
            h2 = f(a2)
            
            j = ob_data_train[n][2][0]
            if j != 0:
                if ii2[0][0] != None:
                    y = np.dot(h2, r) + np.dot(ii2, v)
            
                    e = y - j
                    dl_r = e * h2.T
                    dr += dl_r
                    dl_h2 = e * r.T
            
                    dl_v = e * ii2.T
                    dv += dl_v
                else:
                    y = np.dot(h2, r) + np.dot(h, v)
                
                    e = y - j
                    dl_r = e * h2.T
                    dr += dl_r
                    dl_h2 = e * r.T
            
                    dl_v = e * h.T
                    dv += dl_v
                
                    dl_h = v.T * e
                    dh += dl_h
            
                mid = (dl_h2) * (1 - h2 * h2)
                dl_u = np.dot(i2.T, mid) + lamda * u
                dl_w = np.dot(h1.T, mid) + lamda * w
                du += dl_u
                dw += dl_w
        
                dl_h1 = np.dot(dl_h2, w.T)
                mid = (dl_h1) * (1 - h1 * h1)
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
        while t < count1:
            it = np.asarray(ob_data_train[n][t-1])
            it = it.reshape(1,18)
            iit = np.asarray(user_data_train[n][t-1])
            iit = iit.reshape(1,5)
            
            at = np.dot(it, u) + np.dot(ht1, w)
            ht = f(at)
            j = ob_data_train[n][t][0]
            if j != 0:
                if iit[0][0] != None:
                    y = np.dot(ht, r) + np.dot(iit, v)
            
                    e = y - j
                    dl_r = e * ht.T
                    dr += dl_r
                    dl_ht = e * r.T
            
                    dl_v = e * iit.T
                    dv += dl_v
                else:
                    y = np.dot(ht, r) + np.dot(h, v)
                
                    e = y - j
                    dl_r = e * ht.T
                    dr += dl_r
                    dl_ht = e * r.T
            
                    dl_v = e * h.T
                    dv += dl_v
                
                    dl_h = v.T * e
                    dh += dl_h
                
                mid = (dl_ht) * (1 - ht * ht)
                dl_u = np.dot(it.T, mid) + lamda * u
                dl_w = np.dot(ht1.T, mid) + lamda * w
                du += dl_u
                dw += dl_w
            
                dl_ht1 = np.dot(dl_ht, w.T)
                mid = (dl_ht1) * (1 - ht1 * ht1)
                dl_u = np.dot(it1.T, mid) + lamda * u
                dl_w = np.dot(ht2.T, mid) + lamda * w
                du += dl_u
                dw += dl_w
            
                dl_ht2 = np.dot(dl_ht1, w.T)
                mid = (dl_ht2) * (1 - ht2 * ht2)
                dl_u = np.dot(it2.T, mid) + lamda * u
                dl_w = np.dot(ht3.T, mid) + lamda * w
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
        v = v - learning_rate * dv
        h = h - learning_rate * dh
        
        n += 1
        
        
def predict1():
    count = 0
    for num in ob_data_test:
        count += 1
    all_loss = 0
    n = 0
    while n < count:
        count1 = len(ob_data_test[n])
        ht1 = np.zeros((1, 10))
        for m in range(count1-1):      
            i = np.asarray(ob_data_test[n][m])
            i = i.reshape(1,18)
            at = np.dot(i, u) + np.dot(ht1, w)
            ht = f(at)
            ht1 = ht
        ii = np.asarray(user_data_test[n][0])
        ii = ii.reshape(1,5)
        y = np.dot(ht, r) + np.dot(ii, v)
        j = ob_data_test[n][count1-1][0]
        y = y * 127 + 29
        j = j * 127 + 29
        one_loss = abs(y - j)
        all_loss += one_loss
        n += 1
    loss = all_loss / count
    return loss


best_loss = 100    
for i in range(500):
    train()
    loss = predict1()
    print loss
    #if loss < best_loss:
        #best_loss = loss
        #print loss


#learning_rate = 0.001    lamda = 0.0005  15.4-16.3  4.90
#15.1-16.3  more than 2 times in one month and at least 5 months  3.56652106
#more than 4 times in one month  at least 3 months  3.23957863
# gap a month 3.75768535






    