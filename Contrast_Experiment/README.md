#Contrast Experiment
We apply SVR, GBRT, FM, HMM, traditional RNN and traditonal LSTM to constrast with our rectified LSTM model.  
We also use FM with libFM, but we don't need to write the script, because the input of libFM is the same as libSVM. If you want to use libFM, move the files train_data and test_data into the path 'libfm-master/bin', then run './libFM -task r -train train_data -test test_data -out label -dim ’1,1,8’ -iter 1000' in your cmd. 
