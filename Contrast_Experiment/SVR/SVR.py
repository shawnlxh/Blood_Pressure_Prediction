import json

f_train = open('desktop/train_data','w')
f_test = open('desktop/test_data','w')

with open('Downloads/trim_new(1).json','r') as f:
    all_data = json.load(f)

for index in range(len(all_data)):

	record1 = ''

	user_h = all_data[index]
	target1 = user_h[-1][0]

	record1 += str(target1)+' '

	#user profile
	for i in range(4):
		record1 += str(i)+":"+str(user_h[-1][-4:][i])+' '

	#hisotry pressure measures
	for i in range(4):
		record1 += str(i+4)+":"+str(user_h[-2][:4][i])+' '

	for i in range(4):
		record1 += str(i+8)+":"+str(user_h[-3][:4][i])+' '
		
	for i in range(4):
		record1 += str(i+12)+":"+str(user_h[-4][:4][i])+' '

	record1 += '\n'
	f_test.write(record1)
	
	record2 = ''

	target2 = user_h[-2][0]

	record2 += str(target2)+' '

	#user profile
	for i in range(4):
		record2 += str(i)+":"+str(user_h[-2][-4:][i])+' '

	#hisotry pressure measures
	for i in range(4):
		record2 += str(i+4)+":"+str(user_h[-3][:4][i])+' '

	for i in range(4):
		record2 += str(i+8)+":"+str(user_h[-4][:4][i])+' '
		
	for i in range(4):
		record2 += str(i+12)+":"+str(user_h[-5][:4][i])+' '

	record2 += '\n'
	f_train.write(record2)

f_train.close()
f_test.close()

