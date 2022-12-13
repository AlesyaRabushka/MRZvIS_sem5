from matrix_operations import *
import numpy
import math


# write matrix into file.txt
def write_into_file(file_name, matrix):
    with open(file_name, 'w') as file:
        for row_index in range(len(matrix)):
            for value_index in range(len(matrix[0])):
                if value_index == len(matrix[0])-1:
                    file.write((str(matrix[row_index][value_index])+'\n'))
                else:
                    file.write(str(matrix[row_index][value_index]) + ',')



# read all models from file
def read_model_from_file(file_name):
    models = []
    model = []
    with open(file_name, 'r') as file:
        for line in file:
            if line == '\n':
                models.append(model)
                model = []
            else:
                for x in line.split(','):
                    if int(x) == 1:
                        model.append(1)
                    elif int(x) == 0:
                        model.append(-1)

        models.append(model)

    file.close()
    return models

# read weight matrix from file
def read_matrix_from_file(file_name):
    matrix = []
    with open(file_name, 'r') as file:
        matrix = [[int(x) for x in line.split(',')] for line in file]
    file.close()

    return matrix

# show model in human readable way
def show_model(model):
    count_new_line = 0
    for value_index in range(len(model[0])):
        count_new_line += 1
        if count_new_line == 4:
            if model[0][value_index] == 1:
                print(1)
            elif model[0][value_index] == -1:
                print(0)
            count_new_line = 0

        else:
            if model[0][value_index] == 1:
                print(1, end='')
            elif model[0][value_index] == -1:
                print(0, end='')


# convert vector [] into matrix [[]]
def from_vector_to_matrix(vector):
    new_vector = []
    new_vector.append(vector)
    return new_vector


# calculate weight matrix from given models
def get_weight_matrix_from_models(models):
    weight_matrix = matrix_multiplication(matrix_transposition(from_vector_to_matrix(models[0])), from_vector_to_matrix(models[0]))
    # for j in weight_matrix:
    #     print(j)
    for model_index in range(1, len(models)):
        matrix = matrix_sum(weight_matrix, matrix_multiplication(matrix_transposition(from_vector_to_matrix(models[model_index])),from_vector_to_matrix(models[model_index])))
        weight_matrix = matrix
        # for i in weight_matrix:
        #     print(i)
        # print('---------------------------')
    weight_matrix = zero_main_diagonal(weight_matrix)
    return weight_matrix


def transform_into_matrix(vector):
    matrix = []
    for index in range(len(vector)):
        row_in_matrix = []
        row_in_matrix.append(vector[index])
        matrix.append(row_in_matrix)
    return matrix



# fill the main diagonal with 0s
def zero_main_diagonal(matrix):
    for row_index in range(len(matrix)):
        for col_index in range(len(matrix[0])):
            if row_index == col_index:
                matrix[row_index][col_index] = 0
    return matrix


# activation function
def activation_function(matrix):
    new_matrix = generate_matrix(len(matrix), len(matrix[0]))
    for row_index in range(len(matrix)):
        for col_index in range(len(matrix[0])):
            new_matrix[row_index][col_index] = round(math.tanh(matrix[row_index][col_index]))
    return new_matrix