import os

os.chdir('desktop/libsvm-3.21/python')
from svmutil import *
y, x = svm_read_problem('train_data')
m = svm_train(y,x,'-s 4 -t 0 -h 0')

y, x = svm_read_problem('test_data')
p_label, p_acc, p_val = svm_predict(y, x, m)
f = open('label', 'w')
for i in p_label:
    f.write(str(i))
    f.write('\n')
    
f.close()
