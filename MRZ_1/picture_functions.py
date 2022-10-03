import matplotlib.pylab as plt
from formulas import *
from matrix_operations import *

def pic_to_binary(pic):
    pic_binary = plt.imread(pic)
    # print(pic_binary)
    # print(pic_binary.shape)
    print(pic_binary.size, pic_binary.shape)
    # plt.imshow(pic_binary)
    return pic_binary


def pic_matrix_split(pic_matrix, n, m):
    count = 0
    amount_in_row = len(pic_matrix[0])//m

    l = len(pic_matrix)*len(pic_matrix[0])//n//m//amount_in_row
    # all x_q_vectors in one
    splitted_q_vectors = []

    n1,m1 = 0,0
    n2, m2 = n, m
    # go through all the q matrices
    while count != l:
        splitted_q_vectors.append(pics_matrix_split(pic_matrix, n1,n2,m1,m2))
        count += 1
        n1 += n
        m1 += m
        n2 += n
        m2 += m



# get q vectors from splitting MATRIX into q matrices
def pics_matrix_split(pic_matrix, n1, n2, m1, m2):
    x_q_vector = []
    for i in range(n1, n2):
        for j in range(m1, m2):
            for k in range(len(pic_matrix[i][j])):
                x_q_vector.append(pic_matrix[i][j][k])

    return x_q_vector
