from matrix_operations import matrix_transposition
from matrix_operations import *
import random

# convert color to digits = c(ij)
def pixel_color_to_digits(c_i):
    return float('{:.2f}'.format(((2 * c_i) / 255) - 1))


# calculate the ADAPTIVE STEP = a
def adaptive_step(y_i):
    return float('{:.1f}'.format(1/from_vector_to_number(matrix_multiplication(y_i,matrix_transposition(y_i)))))

def first_layer_neurons_training(adaptive_step, x_i, d_x_i, w):
    first_arg = number_matrix_multiplication(adaptive_step, matrix_transposition(x_i))
    second_arg = matrix_multiplication(first_arg,d_x_i)
    third_arg = matrix_multiplication(second_arg, matrix_transposition(matrix_transposition(w)))
    return matrix_difference(w, third_arg)
# calculate total RMSE (root mean square error) = E
def total_rmse(n, d_x_vector):
    rmse = 0
    for i in range(n):
        rmse  += d_x_vector[i]*d_x_vector[i]
    return rmse


# convert digits to color = u(k)
def pixel_digits_to_color(x_):
    return 255*(x_ + 1)/2


# calculate the compression factor = Z
def compression_factor(n, l, p):
    return (n*l)/((n+l)*p+2)
