# Blood Pressure Prediction
## Background
We cooperate with ihealth company and apply the blood pressure data collected from their intelligent device for home use.

Our data includes the user information data, such as **height, weight(), sex, age**, and the observation data, including **systolic blood 
pressure, diastolic blood pressure, heart rate, the time of the measure, whether taking in drug, mood**.

The target of the project is to predict the average blood pressure value of each user in the next month, using the former measure data of each 
user.

We have two sets of data. One is the users without missing value, and the other is users with missing value of BMI and age.

## Recurrent Neural Network
We apply Recurrent Neural Network(RNN) to this project. RNN can make full use of the sequential information, so we consider that it is the
best choice.

**tanh.py**: The basic RNN model, whose activation function is tanh. The MAE is 3.15
**softmax**: We divide the value of the average blood pressure into several intervals to transfer the regression problem into classifier problem. As a result, we use softmax classifier to the end of RNN.  
**momentum.py**: We apply an advanced gradient descending algorithm called momentum to train our model.  
**dropout.py**: We use a trick called dropout to reduce overfitting.  
**new_model_latent**: We propose a new method to fill in the vacant position in user information, such as bmi and age. A latent vector will be learned to represent the vacant position, and the latent vector is somehow the approximation of all the user. By this way, we can easily use data of all users.
**KNN.py**: We use KNN model to fill in the missing value of BMI and age. After the filling, the MAE is 3.501
