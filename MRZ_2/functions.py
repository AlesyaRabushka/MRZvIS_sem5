from matrix_operations import *
import numpy

def create_matrix_from_sequence(m,p,sequence):
    sequence_matrix = generate_matrix(m,p)
    sequence_value_index = 0 # index of value in sequence
    count = 0
    for row_index in range(m):
        for col_index in range(p):
            sequence_matrix[row_index][col_index] = sequence[sequence_value_index]
            sequence_value_index += 1
        count += 1
        sequence_value_index = count
    return sequence_matrix

# calculate results
def nn_result(k,p,e,a,N,n,input_sequence_number):
    input_sequence = []
    # generate input sequence
    if input_sequence_number == 1:
        input_sequence = fibonacciFunction(k + n)
    elif input_sequence_number == 2:
        input_sequence = factorialFunction(k + n)
    elif input_sequence_number == 3:
        input_sequence = powerFunction(k + n)
    elif input_sequence_number == 4:
        input_sequence = periodicFunction(k + n)

    # nn training process
    output, context_layer, bias_first_layer_weight_matrix, bias_second_layer_weight_matrix, first_layer_weight_matrix, second_layer_weight_matrix, context_layer_weight_matrix = nn_train(k,p,e,a,N,n,input_sequence)
    m = k - p
    # result vector
    output_values = []



    # input vector
    input = []
    for index in range(p):
        input.append(input_sequence[m+index])


    # calculate result values ( amount of values = n )
    for epoch_index in range(n):
        # input sequence
        input[p-1] = output


        # create matrix [[]] from vector
        input_layer_vector = []
        input_layer_vector.append(input)

        # hidden layer calculation
        hidden_layer = matrix_sum(matrix_multiplication(input_layer_vector, first_layer_weight_matrix),
                                  matrix_multiplication(context_layer, context_layer_weight_matrix))  # 1xm
        # hidden_layer = matrix_sum(hidden_layer, bias_first_layer_weight_matrix)
        # hidden layer activation
        for h_index in range(m):
            hidden_layer[0][h_index] = activation_function(hidden_layer[0][h_index])

        # output layer calculation
        output_layer = matrix_multiplication(hidden_layer, second_layer_weight_matrix)  # 1x1

        # activation
        # output_layer[0][0] = output_layer[0][0] + bias_second_layer_weight_matrix[0][0]
        output_layer[0][0] = activation_function(output_layer[0][0])


        output = output_layer[0][0]
        output_values.append(output)

        for value_index in range(len(input)-1):
            input[value_index] = input[value_index + 1]

    # save matrices into files
    numpy.save('first_layer_weight_matrix.npy', first_layer_weight_matrix)
    numpy.save('second_layer_weight_matrix.npy', second_layer_weight_matrix)
    numpy.save('context_weight_matrix.npy', context_layer_weight_matrix)



    print('----------------------------------------------------')
    print(input_sequence)
    for out_index in range(len(output_values)):
        print('Result : ', output_values[out_index], ' Expected : ', input_sequence[k + out_index], ' Line error: ', input_sequence[k + out_index] - output_values[out_index])




# neuron network trainig
def nn_train(k,p,e,a,N,n,input_sequence):
    epoch = 1 # to count number of epoches
    E = 0 # rmse in each epoch
    m = k - p # amount of rows in matrix

    # convert vector into needed matrix
    input_matrix = create_matrix_from_sequence(m,p,input_sequence)

    # generate weight matrices
    first_layer_weight_matrix = generate_matrix(p, m) # pxm
    second_layer_weight_matrix = generate_matrix(m, 1) # mx1
    context_layer_weight_matrix = generate_matrix(1, m) # 1xm

    # contextx matrix
    context_layer = generate_matrix(1, 1)

    # expected values sequence
    expected_values = []


    for value_index in range(m):
        expected_values.append(input_sequence[p+value_index])


    # output value
    output = 0

    # bias
    bias_first_layer_weight_matrix = generate_matrix(1,m)
    bias_second_layer_weight_matrix = generate_matrix(1,1)


    while True:
        output_matrix = []
        if epoch % 10 == 0:
            print('epoch ', epoch, end=' ')
        E = 0

        for epoch_index in range(m):
            input_layer = input_matrix[epoch_index] # X(i)
            # transform a list [] into matrix [[]]
            input_layer_vector = []
            input_layer_vector.append(input_layer)

            # hidden layer calculation
            hidden_layer = matrix_sum(matrix_multiplication(input_layer_vector, first_layer_weight_matrix),matrix_multiplication(context_layer, context_layer_weight_matrix))# 1xm
            # hidden_layer = matrix_sum(hidden_layer, bias_first_layer_weight_matrix)

            # hidden layer activation
            for h_index in range(m):
                hidden_layer[0][h_index] = activation_function(hidden_layer[0][h_index])

            # output layer calculation and activation
            output_layer = matrix_multiplication(hidden_layer, second_layer_weight_matrix) # 1xm
            # output_layer[0][0] = output_layer[0][0] + bias_second_layer_weight_matrix[0][0]
            output_layer[0][0] = activation_function(output_layer[0][0])

            # dX(i)
            dx = output_layer[0][0] - expected_values[epoch_index]
            # print(output_layer[0][0], expected_values[epoch_index])


            # transform a list [] into matrix [[]]
            dx_vector = []
            dx_matrix = []
            dx_vector.append(dx)
            dx_matrix.append(dx_vector)

            # weight matrices correction
            first_layer_weight_matrix = first_layer_neurons_training(a,input_layer_vector, dx_matrix, first_layer_weight_matrix, second_layer_weight_matrix)

            context_layer_weight_matrix = first_layer_neurons_training(a, context_layer, dx_matrix, context_layer_weight_matrix, second_layer_weight_matrix)
            #second_layer_weight_matrix = second_layer_neurons_training(a, second_layer_weight_matrix, hidden_layer,
             #                                                          dx_matrix)

            bias_first_layer_weight_matrix = number_matrix_multiplication(a, matrix_multiplication(matrix_multiplication([[1]], dx_matrix), matrix_transposition(second_layer_weight_matrix)))
            second_layer_weight_matrix = second_layer_neurons_training(a, second_layer_weight_matrix, hidden_layer,
                                                                       dx_matrix)
            # convert number into matrix [[*]]
            bias_vector = []
            bias_matrix = []
            bias_vector.append(dx*a)
            bias_matrix.append(bias_vector)

            bias_second_layer_weight_matrix = matrix_difference(bias_second_layer_weight_matrix, bias_matrix)

            context_matrix = output_matrix
            output_matrix.append(output_layer)
            # rmse
            E += total_rmse(dx_matrix)

            # the most appropriate result
            output = output_layer[0][0]
            exp = expected_values[epoch_index]
        if epoch % 10 == 0:
            print('E = ', E)

        epoch += 1
        if E <= e:
            print(output, exp)
            return output, context_layer, bias_first_layer_weight_matrix, bias_second_layer_weight_matrix, first_layer_weight_matrix, second_layer_weight_matrix, context_layer_weight_matrix



# activation functions lict
def activation_function(value):
    return value

def fibonacciFunction(k):
    sequence = []
    sequence.append(0)
    sequence.append(1)
    for index in range(1,k):
        sequence.append(sequence[index]+sequence[index-1])
    return sequence

def factorialFunction(value):
    sequence = []
    for index in range(value):
        if index == 0:
            sequence.append(1)
        else:
            sequence.append(index*sequence[index-1])
    return sequence

def periodicFunction(value):
    sequence = []
    for index in range(value):
        if index % 2 == 0:
            sequence.append(1)
        elif index % 2 == 1:
            sequence.append(-1)
    return sequence

def powerFunction(value):
    sequence = []
    for index in range(value):
        sequence.append(index ** 2)
    return sequence