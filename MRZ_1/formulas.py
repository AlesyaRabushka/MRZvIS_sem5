from matrix_operations import *

# convert color to digits = c(ij)
def pixel_color_to_digits(c_i):
    return (((2 * c_i) / 255) - 1)

# calculate the ADAPTIVE STEP = a
def adaptive_step(y_i):
    arg = matrix_multiplication(y_i,matrix_transposition(y_i))
    return (1/arg[0][0])


# the training of neurons on the first layer
# W(t+1) = W(t) - a * [X(i)]T * dX(i) * [W`(t)]T
def first_layer_neurons_training(adaptive_step, xi, dx_i, w, w_):
    first_arg = number_matrix_multiplication(adaptive_step, matrix_transposition(xi))
    second_arg = matrix_multiplication(first_arg,dx_i)
    third_arg = matrix_multiplication(second_arg, matrix_transposition(w_))
    return matrix_difference(w, third_arg)


# the training of neurons on the second layer
# W`(t+1) = W`(t) - a` * [Y(i)]T * dX(i)
def second_layer_neurons_training(adaptive_step, w, y_i, dx_i):
    return matrix_difference(w, matrix_multiplication(number_matrix_multiplication(adaptive_step, matrix_transposition(y_i)), dx_i))



# calculate total RMSE (root mean square error) = E(q)
def total_rmse(d_x_vector):
    rmse = 0
    for i in d_x_vector[0]:
        # take dx_vector[0][i] because the vector is actually
        # a one row matrix
        rmse  += (i**2)
    return rmse


# convert digits to color = u(k)
def pixel_digits_to_color(x_):
    return (x_ + 1)/2


# calculate the compression factor = Z
def compression_factor(n, l, p):
    return (n*l)/((n+l)*p+2)
