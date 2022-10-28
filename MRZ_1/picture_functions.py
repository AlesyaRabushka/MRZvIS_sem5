import matplotlib.pyplot as plt
import matplotlib
import numpy
matplotlib.use('TkAgg')
from formulas import *
from matrix_operations import *

# convert img into binary format
def pic_to_binary(pic):
    pic_binary = plt.imread(pic)
    return pic_binary

# save matrix
def save_matrix(matrix, filename):
    numpy.save(filename, matrix)
# get matrix from file
def recreate_matrix_from_file(filename):
    return numpy.load(filename)


# save image
def save_image(img_binary, img_name):
    fig = plt.figure()
    ax = fig.subplots()
    ax.imshow(img_binary)
    ax.axis('off')
    img_path = 'Pictures/result_'+str(img_name)
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)


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
    amount_in_col = len(pic_matrix) // n

    # all x_q_vectors in one
    splitted_vectors = []

    # start values
    n1,m1 = 0,0
    n2, m2 = n, m

    count = 0 # amount of rows we've gone through
    # go through all the q-matrices
    # to make vectors
    while count != amount_in_col:
        splitted_vectors.append(create_splitted_vector(pic_matrix, m, n1,n2,m1,m2))
        count += 1
        n1 += n
        n2 += n
        m1 = 0
        m2 = m
    # print('len of big vector ', len(splitted_q_vectors))
    # print('amount of vectors in row ', len(splitted_q_vectors[0]))
    # print('len of one q-vector ',len(splitted_q_vectors[0][0]))
    return splitted_vectors




# get q vectors from splitting MATRIX into q matrices
def create_splitted_vector(pic_matrix, m, n1, n2, m1, m2):
    x_row_vector = [] # vector of q-matrices vectors

    # go through the q-matrices in row
    while m2 <= len(pic_matrix[0]):
        x_vector = [] # vector of one q-matrix
        for i in range(n1, n2):
            for j in range(m1, m2):
                for k in range(len(pic_matrix[i][j])):
                    x_vector.append(pic_matrix[i][j][k])
        x_row_vector.append(x_vector)
        # go to the next q-matrix in row
        m1 += m
        m2 += m

    return x_row_vector


# fo back from pic vector to matrix
def recreate_pic_matrix(pic_vector, n, m, extra_cols, extra_rows):
    count = 0
    pic_matrix = []
    m1 = 0
    while count < len(pic_vector):
        convert_vector_into_matrix(pic_vector[count], n, m, m1, pic_matrix)
        count += 1
    new_pic_matrix = []
    for row in range(len(pic_matrix)-extra_rows):
        new_pic_matrix_row = []
        for col in range(len(pic_matrix[0]) - extra_cols):
            new_pic_matrix_row.append(pic_matrix[row][col])
        new_pic_matrix.append(new_pic_matrix_row)
    return new_pic_matrix


# fo back from pic vector to matrix
def recreate_mid_pic_matrix(pic_vector, n, m, extra_cols, extra_rows):
    count = 0
    pic_matrix = []
    m1 = 0
    while count < len(pic_vector):
        convert_vector_into_matrix(pic_vector[count], n, m, m1, pic_matrix)
        count += 1
    new_pic_matrix = []
    for row in range(len(pic_matrix)-extra_rows):
        new_pic_matrix_row = []
        for col in range(len(pic_matrix[0]) - extra_cols):
            new_pic_matrix_row.append(pic_matrix[row][col])
        new_pic_matrix.append(new_pic_matrix_row)
    return new_pic_matrix


# convert each q-vector into q-matrix
def convert_vector_into_matrix(pic_vector, n, m, m1, matrix):
    for i in range(n):
        row = []
        for vector_id in range(len(pic_vector)):
            # print('vector ', vector_id)
            # the id of the value in vector
            # that is meant as the start of rgb pixel's values
            start_pixel_id = m1
            for pixel_id in range(m): # select pixels for each row of the result matrix
                # one pixel's rgb values
                pixel_rgb = []
                # select 3 rgb values for each pixel
                for value_id in range(start_pixel_id, start_pixel_id + 3):
                    pixel_rgb.append(pixel_digits_to_color(pic_vector[vector_id][0][value_id]))
                # add pixel into row of matrix
                row.append(pixel_rgb)
                # step to the next rgb-vector of the next pixel
                start_pixel_id += 3
        # add row to the matrix
        matrix.append(row)


# split the pic matrix into L pieces
def img_matrix_split(pic_matrix, n, m):
    division_remainder_row = (len(pic_matrix) % n) # остаток от деления
    division_reminder_col = (len(pic_matrix[0]) % m)
    amount_of_cols_to_add = 0
    amount_of_rows_to_add = 0
    if division_reminder_col != 0:
        amount_of_cols_to_add = m - division_reminder_col
        for row in range(len(pic_matrix)):
            for new_cols in range(amount_of_cols_to_add):
                pixel = []
                for rgb in range(3):
                    pixel.append(random.uniform(-1,1))
                pic_matrix[row].append(pixel)

    if division_remainder_row != 0:
        amount_of_rows_to_add = n - division_remainder_row
        for row in range(amount_of_rows_to_add):
            new_row = []
            for value_id in range(len(pic_matrix[0])):
                rgb = []
                for pixel_rgb in range(3):
                    rgb.append(random.uniform(-1,1))
                new_row.append(rgb)
            pic_matrix.append(new_row)

    # all x_q_vectors in one
    splitted_vectors = []

    # start values
    n1,m1 = 0,0
    n2, m2 = n, m

    amount_in_col = len(pic_matrix) // n
    # print(q_amount_in_col)

    count = 0 # amount of rows we've gone through
    # go through all the q-matrices
    # to make vectors
    while count != amount_in_col:
        splitted_vectors.append(create_img_vector(pic_matrix, m, n1,n2,m1,m2))
        count += 1
        n1 += n
        n2 += n
        m1 = 0
        m2 = m
    return splitted_vectors, amount_of_cols_to_add, amount_of_rows_to_add



# get q vectors from splitting MATRIX into q matrices
def create_img_vector(pic_matrix, m, n1, n2, m1, m2):
    x_row_vector = [] # vector of q-matrices vectors
    # go through the q-matrices in row
    while m2 <= len(pic_matrix[0]):
        x_vector = [] # vector of one q-matrix
        for i in range(n1, n2):
            for j in range(m1, m2):
                for k in range(len(pic_matrix[i][j])):
                    x_vector.append(pic_matrix[i][j][k])
        x_row_vector.append(x_vector)

        # go to the next q-matrix in row
        m1 += m
        m2 += m
    return x_row_vector


# decompress image
def decompress():
    # Y
    y_matrix_from_file = recreate_matrix_from_file('compressed_matrix.npy')
    # W`
    w_matrix_second_layer = recreate_matrix_from_file('second_layer_matrix.npy')

    pic_matrix = []
    for row in range(len(y_matrix_from_file)):
        matrix_row = []
        for id in range(len(y_matrix_from_file[0])):
            # Y(i) = X(i) * W`
            yi = y_matrix_from_file[row][id]
            x_i = matrix_multiplication(yi, w_matrix_second_layer)

            matrix_row.append(x_i)
        pic_matrix.append(matrix_row)

    # get sizes
    size_file = open('pic_size.txt', 'r')
    n = int(size_file.readline())
    m = int(size_file.readline())
    size_file.close()
    # create img matrix
    matrix = recreate_pic_matrix(pic_matrix, n, m, 0, 0)
    # recreate img from matrix
    save_image(matrix, 'pic_deompressed')
    print('The decompression is DONE')


# nn train
def nn_train():
    print('Список картинок:')
    print('- hamster.jpg\n- kilua.jpg\n- ovechka.jpg\n- cat.jpg')
    pic_name = str(input('Введите название картинки: '))
    # get img matrix
    pic_path = 'Pictures/' + pic_name
    pic_binary = pic_to_binary(pic_path)

    # size of each q-matrix
    n = int(input('n '))
    m = int(input('m '))

    # convert rgb values into new ones [-1,1]
    new_pic_binary = convert_rgb(pic_binary)

    # create vector of pixels rgb values
    # Neural Network input vector
    # of q-vectors
    x_vector_of_vectors, extra_cols, extra_rows = img_matrix_split(new_pic_binary, n, m)

    # N - size of each q-vector
    N = len(x_vector_of_vectors[0][0])
    # L - amount of all q-matrices
    L = len(pic_binary) * len(pic_binary[0]) // n // m
    print('N ', N, 'L ', L)

    # total rmse value
    E = 0

    # p - amount of neurons on the hidden level
    p = int(input('Введите количество нейронов второго слоя '))
    # e - maximum allowable error
    e = int(input('Максимально допустимая ошибка: '))

    # weight matrices
    w_matrix_first_layer = generate_weight_matrix(N, p)  # first layer matrix
    w_matrix_second_layer = matrix_transposition(w_matrix_first_layer)  # second layer matrix

    # TRAIN CYCLE
    x1_matrix, amount_of_iterations, E = train_cycle(e, n, m, extra_rows, extra_cols, x_vector_of_vectors, w_matrix_first_layer, w_matrix_second_layer)


    # create img matrix
    matrix = recreate_pic_matrix(x1_matrix, n, m, extra_cols, extra_rows)
    # recreate img from matrix
    save_image(matrix, pic_name)
    # compression factor
    z = compression_factor(N, L, p)

    # output
    print('---------------------------')
    print('Достигнуая ошибка: ', E)
    print('Максимально допустимая ошибка: ', e)
    print('Количество итераций: ', amount_of_iterations)
    print('Степень сжатия: ', z)
    print('Матрица весов на первом слое: ')
    for row in range(len(w_matrix_first_layer)):
        print(w_matrix_first_layer[row])

    print('Матрица весов на втором слое: ')
    for row in range(len(w_matrix_second_layer)):
        print(w_matrix_second_layer[row])


def vector_train(x_vector_of_vectors, w_matrix_first_layer, w_matrix_second_layer, E, row):
    x_matrix_row = []
    y_matrix_row = []
    x1_matrix_row = []
    for vector_id in range(len(x_vector_of_vectors[0])):
        # X(i)
        xi = x_vector_of_vectors[row][vector_id]
        # because i need a 1xN matrix
        # not just a N-sized vector
        xi_vector = []
        xi_vector.append(xi)
        x_matrix_row.append(xi_vector)

        # Y(i) = X(i) * W`
        yi = matrix_multiplication(xi_vector, w_matrix_first_layer)
        # print(len(yi[0]))
        y_matrix_row.append(yi)

        # X`(i) = Y(i) * W
        x_i = matrix_multiplication(yi, w_matrix_second_layer)
        x1_matrix_row.append(x_i)

        # dX(i) = X`(i) - X(i)
        dxi = matrix_difference(x_i, xi_vector)

        # E(q) = sum(dX(q)i * dX(q)i), 1 <= i <= N
        rmse_i = total_rmse(dxi)
        E += rmse_i

        # neurons training
        w_matrix_second_layer = second_layer_neurons_training(0.005, w_matrix_second_layer, yi, dxi)
        w_matrix_first_layer = first_layer_neurons_training(0.005, xi_vector, dxi, w_matrix_first_layer,
                                                            w_matrix_second_layer)

        w_matrix_second_layer = normalize_w_matrices(w_matrix_second_layer)
        w_matrix_first_layer = normalize_w_matrices(w_matrix_first_layer)

        # add matrices into the files
        save_matrix(w_matrix_first_layer, 'first_layer_matrix.npy')
        save_matrix(w_matrix_second_layer, 'second_layer_matrix.npy')

    return w_matrix_first_layer, w_matrix_second_layer, x1_matrix_row, y_matrix_row, E


def train_cycle(e, n, m, extra_rows, extra_cols, x_vector_of_vectors, w_matrix_first_layer, w_matrix_second_layer):
    # amount of iterations
    amount_of_iterations = 0
    while True:
        print('----- новая итерация -----')
        E = 0  # total rmse for all q-matrices
        x1_matrix = []  # X`
        y_matrix = []  # Y
        amount_of_iterations += 1

        # go through each vector
        for row in range(len(x_vector_of_vectors)):
            w_matrix_first_layer, w_matrix_second_layer, x1_matrix_row, y_matrix_row, E = vector_train(x_vector_of_vectors, w_matrix_first_layer, w_matrix_second_layer, E, row)
            x1_matrix.append(x1_matrix_row)
            y_matrix.append(y_matrix_row)

        print('E = ', E)
        print('итерация №', amount_of_iterations)

        if E <= e:
            file_size = open('pic_size.txt', 'w')
            file_size.write(str(n) + '\n')
            file_size.write(str(m))
            file_size.close()

            save_matrix(y_matrix, 'compressed_matrix.npy')
            file_y_matrix_size = open('y_matrix_size.txt', 'w')
            file_y_matrix_size.write(str(len(y_matrix)) + '\n')
            file_y_matrix_size.write(str(len(y_matrix[0])))
            file_y_matrix_size.close()
            break

        else:
            mx = recreate_pic_matrix(x1_matrix, n, m, extra_cols, extra_rows)
            name = 'pic_' + str(amount_of_iterations)
            save_image(mx, name)

    return x1_matrix, amount_of_iterations, E

# compress the pic
def compress():
    print('Список картинок:')
    print('- hamster.jpg\n- kilua.jpg\n- ovechka.jpg\n- cat.jpg')
    pic_name = str(input('Введите название картинки: '))
    print('Введите размеры:')
    n = int(input('n: '))
    m = int(input('m: '))
    pic_path = 'Pictures/' + pic_name
    pic_binary = pic_to_binary(pic_path)


    # convert rgb values into new ones [-1,1]
    new_pic_binary = convert_rgb(pic_binary)

    # create vector of pixels rgb values
    # Neural Network input vector
    # of q-vectors
    x_vector_of_vectors, extra_cols, extra_rows = img_matrix_split(new_pic_binary, n, m)

    # W
    w_matrix_first_layer = recreate_matrix_from_file('first_layer_matrix.npy')

    pic_matrix = []
    for row in range(len(x_vector_of_vectors)):
        matrix_row = []
        for id in range(len(x_vector_of_vectors[0])):
            # Y(i) = X(i) * W`
            xi = x_vector_of_vectors[row][id]
            xi_vector = []
            xi_vector.append(xi)
            x_i = matrix_multiplication(xi_vector, w_matrix_first_layer)

            matrix_row.append(x_i)
        pic_matrix.append(matrix_row)

    save_matrix(pic_matrix, 'compressed_matrix.npy')
    print('the compression is DONE')