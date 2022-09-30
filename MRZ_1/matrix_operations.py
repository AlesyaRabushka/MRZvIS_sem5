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
                    matrix_result[i][j] += matrix_1[i][k] * matrix_2[k][j]

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
            row.append(number * matrix[i][j])
        result_matrix.append(row)
    return result_matrix

# function for MATRIX DIFFERENCE
def matrix_difference(matrix1, matrix2):
    if len(matrix1) == len(matrix2) and len(matrix1[0]) == len(matrix2[0]):
        result_matrix = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix1[0])):
                row.append(matrix1[i][j] - matrix2[i][j])
            result_matrix.append(row)
        return result_matrix


# create a VECTOR from MATRIX
def from_matrix_to_vector(matrix):
    result_vector = []
    for row in matrix:
        for value in row:
            result_vector.append(value)
    return result_vector