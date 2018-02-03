# Vinay Chetnani - V, I, N
# Pranay Dharamsare - P, R, A
# Himanshu Bibyan - H, M, S
We are using alphabets from our names, they are 'V', 'I', 'N', 'P', 'R', 'A', 'H', 'M', 'S'
We are using "EMNIST-byclass" dataset, from that we are only using capital alphabets as mentioned above




# Mapping of dataset labels is as follows:
    # V - 31
    # I - 18
    # N - 23
    # P - 25
    # R - 27
    # A - 10
    # H - 17
    # M - 22
    # S - 28
We will extract only these aplhabets into training_images1 as done in code. These is done for both training and testing data.

This dataset is unbalanced dataset, I figure out that in alphabets I used for training, "H" has least number of data in both training and testing data. So to balance the data, I took the aplhabets data equal to the data available for H.
It is shown below.

Training data
    A       H        I          M        N        P        R        S           V
6407  3152  4946   9002   8237  8347   5073   20764   4637

Testing data
    A      H       I         M        N        P       R      S        V
1062  521  2048  1485   1351  1397   809  3508   796


Implemented a Basic DNN with Backpropogation algorithm, It contains 2 hidden layers,
Input Layer : 784
1st hidden layer : 100
2nd hidden layer : 80
Output layer : 9(corresponding to each alphabet)


Open the python interpretor(cmd) in the folder and type the command given below to run the training:

COMMAND : python code.py   --file="emnist-matlab\emnist-byclass.mat"

Currently this model is giving about 50% accuracy. Here is the output of the model:

D:\sem8\ell888\assn1>python import.py --file="emnist-matlab\emnist-byclass.mat"
(28368, 784)
(28368,)
(4689, 784)
(4689,)
Enterted in iteration = 1
Enterted in iteration = 2
Enterted in iteration = 3
Enterted in iteration = 4
Enterted in iteration = 5
Enterted in iteration = 6
Enterted in iteration = 7
Enterted in iteration = 8
Correctly classifed : 2338
Incorrectly Classified: 2351

We need to tune the hyperparameters and also do the reamaining work as mentioned below to improve the accuracy.



Things which is to be done in future :
1. L1 regularization
2. L2 regularization
3. Adding Droping out layer(Dont know what this is)