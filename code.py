# Vinay Chetnani - V, I, N
# Pranay Dharamsare - P, R, A
# Himanshu Bibyan - H, M, S
# We are using alphabets from our names, they are 'V', 'I', 'N', 'P', 'R', 'A', 'H', 'M', 'S'
# We are using "EMNIST-byclass" dataset, from that we are only using capital alphabets as mentioned above
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
import argparse
import numpy as np

def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    # Local functions
    def rotate(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        flipped = np.fliplr(img)
        return np.rot90(flipped)

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        return render

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('bin/mapping.p', 'wb' ))

    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

    # Reshape training data to be valid
    if verbose == True: _len = len(training_images)
    for i in range(len(training_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        training_images[i] = rotate(training_images[i])
    if verbose == True: print('')

    # Reshape testing data to be valid
    if verbose == True: _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        testing_images[i] = rotate(testing_images[i])
    if verbose == True: print('')

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)

    return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='A training program for classifying the EMNIST dataset')
    parser.add_argument('-f', '--file', type=str, help='Path .mat file data', required=True)
    parser.add_argument('--width', type=int, default=28, help='Width of the images')
    parser.add_argument('--height', type=int, default=28, help='Height of the images')
    parser.add_argument('--max', type=int, default=None, help='Max amount of data to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train on')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enables verbose printing')
    args = parser.parse_args()

    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    (training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes = load_data(args.file, width=args.width, height=args.height, max_=args.max, verbose=args.verbose)
    
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
    # We will extract only these aplhabets into training_images1 as given below. These is done for both training and testing data



    co1 = (3152*9)
    co2 = (521*9)


    training_images1 = np.zeros((co1, 784))
    training_labels1 = np.int_([])
    testing_images1 = np.zeros((co2, 784))
    testing_labels1 = np.int_([])

    cou = 0
    te1 = np.int_(np.zeros((9)))
    te2 = np.int_(np.zeros((9)))


    for idx, co in enumerate(training_labels):
        if (co == 10):
            te1[0] += 1
            if (te1[0] <= 3152):
                training_images1[cou] = training_images[idx].reshape(1, 784)[0]
                cou += 1
                training_labels1 = np.append(training_labels1, (training_labels[idx]))
        elif(co == 17):
            te1[1] += 1
            if (te1[1] <= 3152):
                training_images1[cou] = training_images[idx].reshape(1, 784)[0]
                cou += 1
                training_labels1 = np.append(training_labels1, (training_labels[idx]))
        elif(co == 18):
            te1[2] += 1
            if (te1[2] <= 3152):
                training_images1[cou] = training_images[idx].reshape(1, 784)[0]
                cou += 1
                training_labels1 = np.append(training_labels1, (training_labels[idx]))
        elif(co == 22):
            te1[3] += 1
            if (te1[3] <= 3152):
                training_images1[cou] = training_images[idx].reshape(1, 784)[0]
                cou += 1
                training_labels1 = np.append(training_labels1, (training_labels[idx]))
        elif(co == 23):
            te1[4] += 1
            if (te1[4] <= 3152):
                training_images1[cou] = training_images[idx].reshape(1, 784)[0]
                cou += 1
                training_labels1 = np.append(training_labels1, (training_labels[idx]))
        elif(co == 25):
            te1[5] += 1
            if (te1[5] <= 3152):
                training_images1[cou] = training_images[idx].reshape(1, 784)[0]
                cou += 1
                training_labels1 = np.append(training_labels1, (training_labels[idx]))
        elif(co == 27):
            te1[6] += 1
            if (te1[6] <= 3152):
                training_images1[cou] = training_images[idx].reshape(1, 784)[0]
                cou += 1
                training_labels1 = np.append(training_labels1, (training_labels[idx]))
        elif(co == 28):
            te1[7] += 1
            if (te1[7] <= 3152):
                training_images1[cou] = training_images[idx].reshape(1, 784)[0]
                cou += 1
                training_labels1 = np.append(training_labels1, (training_labels[idx]))
        elif(co == 31):
            te1[8] += 1
            if (te1[8] <= 3152):
                training_images1[cou] = training_images[idx].reshape(1, 784)[0]
                cou += 1
                training_labels1 = np.append(training_labels1, (training_labels[idx]))

    cou1 = 0
    for idx, co in enumerate(testing_labels):
        if (co == 10):
            te2[0] += 1
            if (te2[0] <= 521):
                testing_images1[cou1] = testing_images[idx].reshape(1, 784)[0]
                cou1 += 1
                testing_labels1 = np.append(testing_labels1 , testing_labels[idx])
        elif(co == 17):
            te2[1] += 1
            if (te2[1] <= 521):
                testing_images1[cou1] = testing_images[idx].reshape(1, 784)[0]
                cou1 += 1
                testing_labels1 = np.append(testing_labels1 , testing_labels[idx])
        elif(co == 18):
            te2[2] += 1
            if (te2[2] <= 521):
                testing_images1[cou1] = testing_images[idx].reshape(1, 784)[0]
                cou1 += 1
                testing_labels1 = np.append(testing_labels1 , testing_labels[idx])
        elif(co == 22):
            te2[3] += 1
            if (te2[3] <= 521):
                testing_images1[cou1] = testing_images[idx].reshape(1, 784)[0]
                cou1 += 1
                testing_labels1 = np.append(testing_labels1 , testing_labels[idx])
        elif(co == 23):
            te2[4] += 1
            if (te2[4] <= 521):
                testing_images1[cou1] = testing_images[idx].reshape(1, 784)[0]
                cou1 += 1
                testing_labels1 = np.append(testing_labels1 , testing_labels[idx])
        elif(co == 25):
            te2[5] += 1
            if (te2[5] <= 521):
                testing_images1[cou1] = testing_images[idx].reshape(1, 784)[0]
                cou1 += 1
                testing_labels1 = np.append(testing_labels1 , testing_labels[idx])
        elif(co == 27):
            te2[6] += 1
            if (te2[6] <= 521):
                testing_images1[cou1] = testing_images[idx].reshape(1, 784)[0]
                cou1 += 1
                testing_labels1 = np.append(testing_labels1 , testing_labels[idx])
        elif(co == 28):
            te2[7] += 1
            if (te2[7] <= 521):
                testing_images1[cou1] = testing_images[idx].reshape(1, 784)[0]
                cou1 += 1
                testing_labels1 = np.append(testing_labels1 , testing_labels[idx])
        elif(co == 31):
            te2[8] += 1
            if (te2[8] <= 521):
                testing_images1[cou1] = testing_images[idx].reshape(1, 784)[0]
                cou1 += 1
                testing_labels1 = np.append(testing_labels1 , testing_labels[idx])

    print(training_images1.shape)
    print(training_labels1.shape)
    print(testing_images1.shape)
    print(testing_labels1.shape)

    # plt.title('Label is {label}'.format(label=training_labels1[25]))
    # plt.imshow(training_images1[25].reshape(28,28), cmap='gray')
    # plt.show()
    # plt.title('Label is {label}'.format(label=testing_labels1[25]))
    # plt.imshow(testing_images1[25].reshape(28,28), cmap='gray')
    # plt.show()
    np.random.seed(20)
    w2 = np.random.randn(100,784)
    w3 = np.random.randn(80,100)
    w4 = np.random.randn(9,80)
    b2 = np.random.randn(100)
    b3 = np.random.randn(80)
    b4 = np.random.randn(9)
    err2 = np.zeros(100)
    err3 = np.zeros(80)
    err4 = np.zeros(9)
    delw2 = np.zeros((100,784))
    delw3 = np.zeros((80,100))
    delw4 = np.zeros((9,80))
    delb2 = np.zeros(100)
    delb3 = np.zeros(80)
    delb4 = np.zeros(9)
    z2 = np.zeros(100)
    z3 = np.zeros(80)
    z4 = np.zeros(9)
    a2 = np.zeros(100)
    a3 = np.zeros(80)
    a4 = np.zeros(9)
    # aracost = np.zeros(60000)



    def activfun(Z):
        temp = Z*(-1)
        temp1 = np.exp(temp)
        temp2 = temp1 + 1
        temp3 = temp2**(-1)
        return temp3

    def convert(Z):
        temp = np.zeros(9)
        if(Z == 10):
            temp[0] = 1
        elif(Z == 17):
            temp[1] = 1
        elif(Z == 18):
            temp[2] = 1
        elif(Z == 22):
            temp[3] = 1
        elif(Z == 23):
            temp[4] = 1
        elif(Z == 25):
            temp[5] = 1
        elif(Z == 27):
            temp[6] = 1
        elif(Z == 28):
            temp[7] = 1
        elif(Z == 31):
            temp[8] = 1
        return temp


    def initialization(train_image):
        global w2
        global w3
        global w4
        global b2
        global b3
        global b4
        global z2
        global z3
        global z4
        global a2
        global a3
        global a4
        z2 = np.dot(w2,(train_image)) + b2
        a2 = activfun(z2)
        z3 = np.dot(w3,a2) + b3
        a3 = activfun(z3)
        z4 = np.dot(w4,a3) + b4
        a4 = activfun(z4)
        return


    def delbanao(err_n, a_n):
        temp = np.zeros((err_n.shape[0], a_n.shape[0]))
        for bita in range(err_n.shape[0]):
            temp[bita] = np.array(a_n)
        for idx1, bita1 in enumerate(err_n):
            temp[idx1] = temp[idx1]*bita1
        return temp





    def findel(train_image,train_label):
        global a2
        global a3
        global a4
        global w4
        global w3
        global w2
        global err2
        global err3
        global err4
        global delw2
        global delw3
        global delw4
        global delb2
        global delb3
        global delb4
        global b2
        global b3
        global b4
        err4 = (a4 - convert(train_label))*a4*(1-a4)
        err3 = (np.dot(w4.T, err4))*a3*(1-a3)
        err2 = (np.dot(w3.T, err3))*a2*(1-a2)
        delb2 = np.array(err2)
        delb3 = np.array(err3)
        delb4 = np.array(err4)
        delw4 = delbanao(err4, a3)
        delw3 = delbanao(err3, a2)
        delw2 = delbanao(err2, (train_image))
        w2 = w2 - (0.5*delw2)
        w3 = w3 - (0.5*delw3)
        w4 = w4 - (0.5*delw4)
        b2 = b2 - (0.5*delb2)
        b3 = b3 - (0.5*delb3)
        b4 = b4 - (0.5*delb4)
        return 




    ghj = 8
    while (ghj>0):
        print("Enterted in iteration = " + str(9-ghj))
        for idx, bita in enumerate (training_images1):
            initialization(bita)
            findel(bita, training_labels1[idx])
        ghj -= 1

    trai4 = np.int_(np.zeros(28368))
    trai5 = np.int_(np.zeros(4689))
    for idx, bita in enumerate(testing_images1):
        initialization(bita)
        temp = np.argmax(a4)
        trai5[idx] = temp
    for idx, bita in enumerate(training_images1):
        initialization(bita)
        temp = np.argmax(a4)
        trai4[idx] = temp
    for idx, bita in enumerate(trai5):
        if (bita == 0):
            trai5[idx] = 10
        elif(bita == 1):
            trai5[idx] = 17
        elif(bita == 2):
            trai5[idx] = 18
        elif(bita == 3):
            trai5[idx] = 22
        elif(bita == 4):
            trai5[idx] = 23
        elif(bita == 5):
            trai5[idx] = 25
        elif(bita == 6):
            trai5[idx] = 27
        elif(bita == 7):
            trai5[idx] = 28
        elif(bita == 8):
            trai5[idx] = 31
    coun = 0
    coun1 = 0
    for idx, bita in enumerate(testing_labels1):
        if (bita == trai5[idx]):
            coun += 1
        else:
            coun1 += 1
    print(coun)
    print(coun1)