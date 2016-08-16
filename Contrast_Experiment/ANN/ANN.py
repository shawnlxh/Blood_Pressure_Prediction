import numpy as np

f1 = open('/Users/lixiaohan/Desktop/train_data')
f2 = open('/Users/lixiaohan/Desktop/test_data')

count_x = 0
x_train = []
y_train = []
for i in f1:
    i = i.split('\t')
    i = i[0:-1]
    i = [float(i2) for i2 in i]
    y_train.append(i[0])
    x_train.append(i[1:])
    count_x += 1

count_y = 0
x_test= []
y_test = []
for i in f2:
    i = i.split('\t')
    i = i[0:-1]
    i = [float(i2) for i2 in i]
    y_test.append(i[0])
    x_test.append(i[1:]) 
    count_y += 1

u = np.random.randn(16, 10)*0.05
v = np.random.randn(10, 1)*0.05

learning_rate = 0.001
lamda = 0

def f(x):    #tanh
	output = ( np.exp(x) - np.exp(-x) ) /( np.exp(x) + np.exp(-x) )
	return output

def train():
	global u
	global v
	for m in range(count_x):
		i = np.asarray(x_train[m])
		i = i.reshape(1,16)
		j = y_train[m]
		a = np.dot(i, u)
		h = f(a)
		y = np.dot(h, v)

		e = y - j
		dl_v = e * h.T
		dl_h = e * v.T

		mid = dl_h * (1 - h * h)
		dl_u = np.dot(i.T, mid)

		u -= learning_rate * dl_u
		v -= learning_rate * dl_v

for m in range(1000):
	train()

all_loss = 0
all_loss2 = 0
count = 0
for m in range(count_y):
	i = np.asarray(x_test[m])
	i = i.reshape(1,16)
	j = y_test[m]
	a = np.dot(i, u)
	h = f(a)
	y = np.dot(h, v)
	y = y * 127 + 29
 	j = j * 127 + 29
 	one_loss = abs(y - j)
 	one_loss2 = (y - j) * (y - j)
 	all_loss += one_loss
 	all_loss2 += one_loss2
 	count += 1

loss = all_loss / count
loss2 = np.sqrt(all_loss2 / count)
print loss
print loss2





