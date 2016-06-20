import numpy as np
import json
import time

with open('Downloads/trim_ALL_USER_HISTORY_qain&3_15_interval(2).json','r') as f:
    data = json.load(f)
    
count = len(data)

train_data = []
for i in range( int(count * 0.8) ):
    count1 = len(data[i])
    for k in range(count1):
        data[i][k].append(1)
    train_data.append(data[i])

test_data = []
i += 1
while i < count:
    count1 = len(data[i])
    for k in range(count1):
        data[i][k].append(1)
    test_data.append(data[i])
    i += 1
        
u = np.random.randn(23, 10)*0.05
w = np.random.randn(10, 10)*0.05
r = np.random.randn(10, 5)*0.05
learning_rate = 0.001
lamda = 0.0005
beta1 = 0.5
beta2 = 0.3

def f(x):    #tanh
	output = ( np.exp(x) - np.exp(-x) ) /( np.exp(x) + np.exp(-x) )
	return output
	
def rank(pressure):
    ranking = None
    pressure = pressure * 127 + 29
    if pressure < 80:
	ranking = [1,0,0,0,0]
	ranking = np.asarray(ranking)
	ranking = ranking.reshape(1,5)
    elif pressure < 90:
	ranking = [0,1,0,0,0]
	ranking = np.asarray(ranking)
	ranking = ranking.reshape(1,5)
    elif pressure < 99:
	ranking = [0,0,1,0,0]
	ranking = np.asarray(ranking)
	ranking = ranking.reshape(1,5)
    elif pressure < 109: 
	ranking = [0,0,0,1,0]
	ranking = np.asarray(ranking)
	ranking = ranking.reshape(1,5)
    else:
	ranking = [0,0,0,0,1]
	ranking = np.asarray(ranking)
	ranking = ranking.reshape(1,5)
    if ranking==None:
	print pressure
    return ranking

def train():
    global u
    global w
    global r
    count = 0
    for num in train_data:
        count += 1
    n = 0
    while n < count:
        count1 = len(train_data[n])
        
        du = 0
        dw = 0
        dr = 0
        
        #t = 1
        h0 = np.zeros((1, 10)) 
        i1 = np.asarray(train_data[n][0])
        i1 = i1.reshape(1,23)
        a1 = np.dot(i1, u) + np.dot(h0, w)
        h1 = f(a1)
        y = np.dot(h1, r)
        j = train_data[n][1][0]
        
        if j != 0:
            j = rank(j)
            e = 0
            for i in range(5):
                e += np.exp(y[0][i])
            e = np.exp(y) / e
            dl_r = np.dot(h1.T, (e - j)) + lamda * r
            #print j
            #print e
            #time.sleep(2)
            dr += dl_r
            dl_h1 = np.dot((e - j), r.T)
        
            mid = (dl_h1) * (1 - h1 * h1)
            dl_u = np.dot(i1.T, mid) + lamda * u
            dl_w = np.dot(h0.T, mid) + lamda * w
            du += dl_u
            dw += dl_w
        
        if count1 > 2:
            #t = 2
            i2 = np.asarray(train_data[n][1])
            i2 = i2.reshape(1,23)
            a2 = np.dot(i2, u) + np.dot(h1, w)
            h2 = f(a2)
            y = np.dot(h2, r)
            j = train_data[n][2][0]
            
            if j != 0:
                j = rank(j)
                e = 0
                for i in range(5):
                    e += np.exp(y[0][i])
                e = np.exp(y) / e
                dl_r = np.dot(h2.T, (e - j)) + lamda * r
                dr += dl_r
                dl_h2 = np.dot((e - j), r.T)
        
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
            it = np.asarray(train_data[n][t-1])
            it = it.reshape(1,23)
            at = np.dot(it, u) + np.dot(ht1, w)
            ht = f(at)
            y = np.dot(ht, r)
            j = train_data[n][t][0]
            
            if j != 0:
                j = rank(j)
                e = 0
                for i in range(5):
                    e += np.exp(y[0][i])
                e = np.exp(y) / e
                dl_r = np.dot(ht.T, (e - j)) + lamda * r
                dr += dl_r
                dl_ht = np.dot((e - j), r.T)
                
                mid = (dl_ht) * (1 - ht * ht)
                dl_u = np.dot(it.T, dl_ht) + lamda * u
                dl_w = np.dot(ht1.T, dl_ht) + lamda * w
                du += dl_u
                dw += dl_w
            
                dl_ht1 = np.dot(dl_ht, w.T)
                mid = (dl_ht1) * (1 - ht1 * ht1)
                dl_u = np.dot(it1.T, dl_ht1) + lamda * u
                dl_w = np.dot(ht2.T, dl_ht1) + lamda * w
                du += dl_u
                dw += dl_w
            
                dl_ht2 = np.dot(dl_ht1, w.T)
                mid = (dl_ht2) * (1 - ht2 * ht2)
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
        
        n += 1
        
        
def predict():
    count = 0
    for num in test_data:
        count += 1
    all_loss = 0
    n = 0
    while n < count:
        count1 = len(test_data[n])
        ht1 = np.zeros((1, 10))
        for m in range(count1-1):      
            i = np.asarray(test_data[n][m])
            i = i.reshape(1,23)
            at = np.dot(i, u) + np.dot(ht1, w)
            ht = f(at)
            ht1 = ht
        y = np.dot(ht, r)
        e = 0
        for i in range(5):
            e += np.exp(y[0][i])
        e = np.exp(y) / e
        e = np.argsort(-e)
        j = test_data[n][count1-1][0]
        j = rank(j)
        out = -1
        for i in range(5):
            if j[0][i] == 1:
                out = i
        if e[0][0] == out:
            all_loss += 1
        n += 1
    print all_loss
    print count
    loss = float(all_loss) / count
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

#learning_rate = 0.001    lamda = 0.0005  15.1-16.3  0.653000594177
#more than 2 times in one month 0.701965  test number 916




   


    