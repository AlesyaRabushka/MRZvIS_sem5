from picture_functions import *



if __name__ == '__main__':
    print('Выберите вариант взаимодействия:')
    print('1 - Сжать картинку')
    print('2 - Разжать картинку')
    print('3 - Обучение сети')
    index = int(input())

    # decompress image
    if index == 2:
        decompress()
    # training
    elif index == 3:
        nn_train()
    # compress image
    elif index == 1:
        compress()