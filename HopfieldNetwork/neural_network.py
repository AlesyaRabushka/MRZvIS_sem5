from functions import *
import functools



def recognize(X, weight_matrix):
    previous_state = matrix_transposition(X)
    curent_state = []
    relax = False
    iteration = 0

    while relax is not True:
        if iteration >= 1000:
            print('Невозможно распознать образ!')
            break

        current_state = activation_function(matrix_multiplication(weight_matrix, previous_state))


        show_model(matrix_transposition(previous_state))
        print()
        if functools.reduce(lambda x, y: x and y, map(lambda a, b: a == b, previous_state, current_state), True):

            print('Образ распознан!')
            relax = True



        previous_state = current_state
        iteration += 1
