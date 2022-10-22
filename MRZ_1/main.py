from picture_functions import *



if __name__ == '__main__':
    print('Выберите вариант взаимодействия:')
    print('1 - Разжать картинку')
    print('3 - Обучение сети')
    index = int(input())

    # decompress image
    if index == 1:
        decompress()
        print('The decompression is DONE')
    # training
    elif index == 2:
        print('Список картинок:')
        print('- hamster.jpg\n- kilua.jpg\n- ovechka.jpg\n- cat.jpg')
        pic_name = str(input('Введите название картинки: '))
        ns_train(pic_name)
