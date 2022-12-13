from neural_network import *

if __name__ == '__main__':

    while(True):
        print('Выберите действие:')
        print('1 - Обучение')
        print('2 - Распознавание')
        print('0 - Выход')
        menu = (input())


        if int(menu) == 1:
            file_name = input('Введите название файла с образами ')
            models = read_model_from_file(file_name)
            matrix = get_weight_matrix_from_models(models)
            for i in matrix:
                print(i)
            print('Сохранить матрицу весов? ')
            print('1 - Да\n2 - Нет')
            choice = int(input())
            if choice == 1:
                matrix_file_name = input('Введите название файла: ')
                write_into_file(matrix_file_name, matrix)
            else:
                pass


        elif int(menu) == 2:
            model_file_name = input('Введите название файла с образом ')
            matrix_file_name = input('Введите название файла с матрицей весов ')
            matrix = read_matrix_from_file(matrix_file_name)
            model = read_model_from_file(model_file_name)

            recognize(model, matrix)

        elif int(menu) == 0:
            break
        else:
            pass
