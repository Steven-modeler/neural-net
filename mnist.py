# Neural netork program to identify written characters.
# Successfully ran multiple learning rates, multiple numbers of hidden nodes,
# and five runs each, on 12/23/17.
# can i run 1 epoch, lr = 0.1, 100 HN?  yes, on 1/5/2018.

import numpy
import scipy.special
import matplotlib.pyplot
get_ipython().magic('matplotlib inline')
import csv

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # learning rate
        self.lr = learningrate
        # activation function is the sigmoid function.  Where does this get called again?
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
        
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array.
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # calculate signals into hidden layer.
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from the hidden layer.
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer.
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from the final output layer.
        final_outputs = self.activation_function(final_inputs)
        # error is the (target - actual).
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by wts, 
        #  recombined at hidden nodes.
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers.
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers.
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    def query(self, inputs_list):
        # convert inputs list to 2d array.
        inputs = numpy.array(inputs_list, ndmin=2).T
        # calculate signals into hidden layer.
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer.
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer.
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from the final output layer.
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# Main program.
print ("beginning program")

input_nodes = 784
hidden_nodes_list =  [100] # [400] #  500]    
epoch_list =  [1] # [80, 60, 100, 80, 100, 60, 40]
output_nodes = 10
onodes = 10
learning_rate_list = [0.1] # [0.05]       

# initialize output epochs3.csv file with columns: Performance, LR, Epochs, and LR.
headerRow = ["Perf", "LR", "Epochs", "HN"]
outputFile = "OneDrive/neural net/epochLR05.csv"
with open(outputFile, "wt") as mycsv:   # structure closes file when passes.
    wr = csv.writer(mycsv, dialect='excel')
    wr.writerow(headerRow)
    pass

# Get training data.
# training_data_file = open("OneDrive/neural net/mnist_train_100.csv", 'r')
training_data_file = open("OneDrive/neural net/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# now load the test data.
# test_data_file = open("OneDrive/neural net/mnist_test_10.csv", 'r')
test_data_file = open("OneDrive/neural net/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# create loops here to explore parameters.
for learning_rate in learning_rate_list:
    for epoch in epoch_list:  
        for hidden_nodes in hidden_nodes_list  :
            scorecard = []    # scorecard should be in innermost for loop.
            # Create instance of neural net.
            n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

            # Train the neural network.
            for record in training_data_list * int(epoch):
                # split the record by the comma, because data was in csv file.
                # all_values = record.split(',')
                train_values = record.split(',')
                # scale and shift the inputs to be 0.01 < input <= 1.0.
                train_inputs = (numpy.asfarray(train_values[1:]) / 255.0 * 0.99) + 0.01
                targets = numpy.zeros(onodes) + 0.01
                targets[int(train_values[0])] = 0.99
                n.train(train_inputs, targets) 
            pass
            # Now test and score the test data.
            for record in test_data_list:
                test_values = record.split(',')
                # correct answer is first value.
                correct_label = int(test_values[0])
                # scale and shift inputs so as not to blow up the weights.
                test_inputs = (numpy.asfarray(test_values[1:]) / 255.0 * 0.99) + 0.01
                # query the net.
                # outputs = n.query(inputs)
                outputs = n.query(test_inputs)
                # the index of the highest value corresponds to net's guess label.
                label = numpy.argmax(outputs)
                # append correct or incorrect to list.
                if (label == correct_label):
                    # if correct, add 1 to scorecard.
                    scorecard.append(1)
                else:
                    # if not correct, add 0 to scorecard.
                    scorecard.append(0)
                    pass
            # calculate the performance of the scorecard.
            scorecard_array = numpy.asarray(scorecard)
            print ("performance = ", scorecard_array.sum() / scorecard_array.size)
            performance = scorecard_array.sum() / scorecard_array.size
            # time between results may be significant.  
            # open results file, write Performance, LR, Epochs, and LR.
            resultsRow = [str(performance), str(learning_rate), str(epoch), str(hidden_nodes)]
            with open(outputFile, "at") as mycsv:   # structure closes file when passes.
                wr = csv.writer(mycsv, dialect='excel')
                wr.writerow(resultsRow)
                pass
            pass
        pass
    pass
pass

print ("mycsv file closed? ", mycsv.closed, " quitting")
