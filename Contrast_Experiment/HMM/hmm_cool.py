import numpy as np
from hmmlearn import hmm
import json
import time

with open('/Users/lixiaohan/Downloads/trim_new(1).json','r') as f:
    all_data = json.load(f)

count = len(all_data)
print count

data = []
for i in range(count):
    line = []
    for m in all_data[i]:
        m[0] = m[0] * 127 + 29
        m_list = [m[0]]
        line.append(m_list)
    data.append(line)

all_loss = 0
for user in data:
    num = len(user)
    train = user[0:-1]
    test = user[-1]
    train = np.asarray((train))
    model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=2000)
    model.fit(train)
    y = model.sample(num)
    y = y[0][num-1]
    one_loss = abs(test-y)
    all_loss += one_loss

result = all_loss / count
print result
