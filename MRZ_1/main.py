from matrix_operations import *
from formulas import *
from picture_functions import *
import random
pic_binary = pic_to_binary('Pictures/hunter.jpg')


n = 16
m = 9

p = int(input('Введите количество нейронов второго слоя '))
# convert rgb values into new ones [-1,1]
new_pic_binary = convert_rgb(pic_binary)
x_vector_of_vectors = pic_matrix_split(new_pic_binary,n,m)

# generate first W matrix with random values
# in range [-1,1]
w = generate_weight_matrix(len(x_vector_of_vectors[0][0]), p)
print('len W ',len(w))

# recreate_pic_matrix(split,16,9)

L = len(pic_binary)*len(pic_binary[0])//n//m
print('q amount ',L)


y_vector = []

c = 0
# for each q-matrix
for i in range(1):
    for j in range(1):
        # X(i)
        x_i = []
        x_i.append(x_vector_of_vectors[i][j])
        print('len of X(i) = ',len(x_i))
        print(x_i)
        print('------------------------------------')
        print(w)
        # Y(i) = X(i) * W
        y_i = matrix_multiplication(x_i,w)
        y_vector.append(y_i)
        print(y_i)
        # W`
        w_ = matrix_transposition(w)
        # X`(i) = Y(i) * W`
        x_i_ = matrix_multiplication(y_i, w_)
        # dX(i) = X`(i) - X(i)
        d_x_i_ = matrix_difference(x_i_, x_i)
