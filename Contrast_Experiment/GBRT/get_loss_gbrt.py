import os
#os.chdir('desktop/libsvm-3.21/python')
f1 = open('test_data')
f2 = open('label')

label1 = []
for i in f1:
    i = i.split('\t')
    i = i[0]
    i = float(i) * 127 + 29
    label1.append(i)
    


label2 = []
for i in f2:
    i = float(i) * 127 + 29
    label2.append(i)
    
loss = 0
loss2 = 0
count = 0
for i in label1:
    loss += abs(label1[count] - label2[count])
    count += 1
    
loss = float(loss) / count
print loss
