from matrix_operations import *
from formulas import *
from picture_functions import *
import random
pic_binary = pic_to_binary('Pictures/hunter.jpg')


n = 16
m = 9
# convert rgb values into new ones [-1,1]
new_pic_binary = convert_rgb(pic_binary)
split = pic_matrix_split(new_pic_binary,n,m)


# recreate_pic_matrix(split, n,m)
recreate_pic_matrix(split,16,9)

# for i in range(40,43):
#     print(new_pic_binary[i])
#
# # эталонный вектор
# X_q = []
# for row in new_pic_binary:
#     X_q.append(from_matrix_to_vector(row))
#
# print('--------------------')
# for i in range(40,42):
#     print(X_q[i])

m = [[2,2,2],
     [2,4,3],
     [4,5,2]]


# to_inverse_matrix(m)

x1 = [[0.1,0.2],
      [0.3,0.4]]

x2 = [[0.4,0.5],
      [0.7,0.6],
      [0.1,0.2],
      [0.7,0.8]]



w1 = [[0.1,0.2],
     [0.3,0.4],
     [0.5,0.6],
     [0.7,0.8]]






f = [[6,-5,8,4 ],
     [9,7,5,2],
     [7,5,3,7],
     [-4,8,-8,-3]]



# n = int(input('n: '))
# m = int(input('m: '))
# p = int(input('p: '))
# L = 4
# N = n * n
#
# c_vector = []
# x_q_vector = []
# for i in range(L):
#     c_i = (generate_weight_matrix(n,n))
#     c_vector.append(c_i)
#
#     x_i = from_matrix_to_vector(c_i)
#     x_q_vector.append(x_i)
#
# w = generate_weight_matrix(N,p)
# print(x_q_vector[0])
#
# y_vector=[]
# x_vector = []
# d_x_vector = []
#
# for i in range(L):
#     y_i = matrix_multiplication(x_q_vector[i], w)
#     y_vector.append(y_i)
#     x__i = matrix_multiplication(y_i,matrix_transposition(w))
#     x_vector.append(x__i)
#     d_x_i = matrix_difference(x__i, x_q_vector[i])
#
#     new_w = first_layer_neurons_training(adaptive_step(x_q_vector[i]),x_q_vector[i], d_x_i, w)
#     w = new_w
#     print('--------------')
#     for h in w:
#         print(h)