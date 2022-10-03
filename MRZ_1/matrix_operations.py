import random



# function to calculate the MULTIPLICATION OF MATRIXES
def matrix_multiplication(matrix_1, matrix_2):
    # check for consistence of the matrixes
    if len(matrix_1[0]) == len(matrix_2):
        # fill result matrix with 0's
        matrix_result = []
        for m in range(len(matrix_1)):
            row = []
            for n in range(len(matrix_2[0])):
                row.append(0)
            matrix_result.append(row)


        # calculate the values of the RESULT MATRIX
        for i in range(len(matrix_1)):
            for j in range(len(matrix_1[0])):
                for k in range(len(matrix_2[0])):
                    matrix_result[i][k] += float('{:.2f}'.format(matrix_1[i][j] * matrix_2[j][k]))

        return matrix_result



# function to create a TRANSPOSED MATRIX
def matrix_transposition(matrix):
    result_matrix = []
    # go through rows
    for i in range(len(matrix[0])):
        col = []
        # go through columns
        for j in range(len(matrix)):
            col.append(matrix[j][i])
        result_matrix.append(col)
    return result_matrix


# multiplication of number on matrix
def number_matrix_multiplication(number, matrix):
    result_matrix = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            row.append(float('{:.2f}'.format(number * matrix[i][j])))
        result_matrix.append(row)
    return result_matrix

# function for MATRIX DIFFERENCE
def matrix_difference(matrix1, matrix2):
    if len(matrix1) == len(matrix2) and len(matrix1[0]) == len(matrix2[0]):
        result_matrix = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix1[0])):
                row.append(float('{:.2f}'.format(matrix1[i][j] - matrix2[i][j])))
            result_matrix.append(row)
        return result_matrix


# create a VECTOR from MATRIX
def from_matrix_to_vector(matrix):
    result_vector = []
    v = []
    for row in matrix:
        for value in row:
            result_vector.append(value)
    # result_vector.append(v)
    print(len(result_vector))
    return result_vector

# create a NUMBER from VECTOR 1x1
def from_vector_to_number(vector):
    return vector[0][0]

# generate weight matrix
def generate_weight_matrix(n, p):
    w_matrix = []
    for i in range(n):
        w_matrix_row = []
        for j in range(p):
            w_matrix_row.append(float('{:.2f}'.format(random.uniform(-1,1))))
        w_matrix.append(w_matrix_row)
    return w_matrix

















#
# def find_minor(matrix, vector):
#     for i in range(len(matrix)):
#         row_v = []
#         for j in range(len(matrix[0])):
#             new_matrix = []
#
#             for k in range(len(matrix)):
#                 row = []
#                 for l in range(len(matrix[0])):
#                     if k != i and l != j:
#                         row.append(matrix[k][l])
#                 if len(row) != 0:
#                     new_matrix.append(row)
#
#             if len(new_matrix) == 2:
#                 print('matrix in progress:')
#                 for h in new_matrix:
#                     print(h)
#                 print('det = ', determinant(new_matrix))
#
#                 row_v.append(algebraic_complement(i+1, j+1, determinant(new_matrix)))
#
#             else:
#                 find_minor(new_matrix, vector)
#         vector.append(row_v)
#
#
#
#
# def to_inverse_matrix(matrix):
#     # find the matrix of minors
#     algebraic_complement_matrix = []
#     mat = find_minor(matrix, algebraic_complement_matrix)
#     print(algebraic_complement_matrix)
#     transported_algebraic_complement_matrix = matrix_transposition(algebraic_complement_matrix)
#     print('mat: ', mat)
#     return number_matrix_multiplication(1/(determinant(matrix)),algebraic_complement_matrix)
