import matplotlib.pylab as plt
from formulas import *
from matrix_operations import *

def pic_to_binary(pic):
    pic_binary = plt.imread(pic)
    # print(pic_binary)
    # print(pic_binary.shape)
    print(pic_binary.size, pic_binary.shape)
    return pic_binary

# convert pixel's rgb values into new ones in [-1,1]
def convert_rgb(pic_binary):
    new_pic_binary = []
    for row_id in range(len(pic_binary)):
        new_row = []
        for i in range(len(pic_binary[0])):
            new_pixel = []
            for j in range(len(pic_binary[0][0])):
                new_pixel.append(pixel_color_to_digits(pic_binary[row_id][i][j]))
            new_row.append(new_pixel)
        new_pic_binary.append(new_row)
    return new_pic_binary


# split the pic matrix into L pieces
def pic_matrix_split(pic_matrix, n, m):
    # amount of q-matrices in col
    q_amount_in_col = len(pic_matrix) // n

    # all x_q_vectors in one
    splitted_q_vectors = []

    # start values
    n1,m1 = 0,0
    n2, m2 = n, m

    count = 0 # amount of rows we've gone through
    # go through all the q-matrices
    # to make vectors
    while count != q_amount_in_col:
        splitted_q_vectors.append(create_splitted_vector(pic_matrix, m, n1,n2,m1,m2))
        count += 1
        n1 += n
        n2 += n
        m1 = 0
        m2 = m
    print('len of big vector ', len(splitted_q_vectors))
    print('amount of vectors in row ', len(splitted_q_vectors[0]))
    print('len of one q-vector ',len(splitted_q_vectors[0][0]))
    return splitted_q_vectors



# get q vectors from splitting MATRIX into q matrices
def create_splitted_vector(pic_matrix, m, n1, n2, m1, m2):
    x_row_vector = [] # vector of q-matrices vectors

    # go through the q-matrices in row
    while m2 <= len(pic_matrix[0]):
        x_q_vector = [] # vector of one q-matrix
        for i in range(n1, n2):
            for j in range(m1, m2):
                for k in range(len(pic_matrix[i][j])):
                    x_q_vector.append(pic_matrix[i][j][k])
        x_row_vector.append(x_q_vector)
        # go to the next q-matrix in row
        m1 += m
        m2 += m

    return x_row_vector


# fo back from pic vector to matrix
def recreate_pic_matrix(pic_vector, n, m):
    count = 0
    pic_matrix = []

    while count != len(pic_vector):
        pic_matrix.append(convert_vector_into_matrix(pic_vector[count], n, m))
        count += 1
    print(len(pic_matrix[0][0][0][0]))
    return pic_matrix


# convert each q-vector into q-matrix
def convert_vector_into_matrix(q_vector_row, n, m):
    # row of q-matrices
    q_matrices_rows = []

    start_id = 0
    end_id = 3

    q_matrix_amount = 0 # amount of q-matrices in a row
    while q_matrix_amount < len(q_vector_row):
        q_matrix = []
        for i in range(n):
            q_matrix_row = []
            for j in range(m):
                q_matrix_pixel = []
                for value_id in range(start_id, end_id):
                    q_matrix_pixel.append(q_vector_row[q_matrix_amount][value_id])
                start_id += 3
                end_id += 3
                q_matrix_row.append(q_matrix_pixel)
            q_matrix.append(q_matrix_row)
        q_matrices_rows.append(q_matrix)
        q_matrix_amount += 1
        start_id = 0
        end_id = 3

    return q_matrices_rows
