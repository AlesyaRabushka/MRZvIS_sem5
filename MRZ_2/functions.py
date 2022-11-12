from matrix_operations import *

def create_matrix_from_sequence(m,p,sequence):
    sequence_matrix = generate_matrix(m,p)
    sequence_value_index = 0 # index of value in sequence
    count = 0
    row_index = 0
    col_index = 0
    while count != m:
        for i in range(p):
            sequence_matrix[row_index][col_index] = sequence[sequence_value_index]
            col_index += 1
            sequence_value_index += 1

        col_index = 0
        row_index += 1
        count += 1
        sequence_value_index = count
    return sequence_matrix




def open():
    first_layer_weight_matrix = generate_matrix(7,4)
    second_layer_weight_matrix = generate_matrix(4,1)