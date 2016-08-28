import numpy as np
import json
import time
import operator

with open('Downloads/trim_new_total_with_missing_value.json','r') as f:
    data = json.load(f)
    
count = len(data)

train_data = []
train_label = []
test_data = []
for i in range(count):
    count1 = len(data[i])
    ob = []
    if data[i][0][17] != None and data[i][0][18] != None:
        for k in range(3):
            temp = data[i][count1-k-1][0:3]
            ob.extend(temp)
        temp2 = data[i][k][17:19]
        temp2 = tuple(temp2)
        train_data.append(ob)
        train_label.append(temp2)
    else:
        for k in range(3):
            temp = data[i][count1-k-1][0:3]
            ob.extend(temp)
        test_data.append(ob)

train_data = np.array(train_data)
test_data = np.array(test_data)

def knnClassifier(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]
    
def predict():
    count = len(test_data)
    test_label = []
    for i in range(count):
        temp = knnClassifier(test_data[i], train_data, train_label, 3)
        temp = list(temp)
        test_label.append(temp)
    return test_label
    
test_label = predict()
count = len(data)
m = 0
for i in range(count):
    count1 = len(data[i])
    for q in range(count1):
        if data[i][q][17] == None or data[i][q][18] == None:
            data[i][q][17] = test_label[m][0]
            data[i][q][18] = test_label[m][1]
    m += 1

f2 = open('desktop/filling_missing_value.json','w')
json.dump(data, f2)
f2.close()
