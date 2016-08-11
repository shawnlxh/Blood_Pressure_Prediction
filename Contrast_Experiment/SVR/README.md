##SVR
First I try SVR to handle our data, because SVR is the most popular and efficient method to deal with regression problems. I use libSVM to
simplify my work. The basic setting is '-s 4 -t 0 -h 0'. The following is the description of the code files:  
  
**SVR.py**: This is how to transform the basic data file into what we need to adjust the requirement of libSVM.  
**svr_get_label.py**: This is to train the model and store the result of the prediction into a file.  
**svr_get_loss.py**: The aim of this file is to evaluate the performance of SVR. The evaluating indicator is MAE.
