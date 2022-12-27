from typing import List
import math


def tf_transform(count_matrix: List[list]) -> List[list]:
    """
    Возвращает term frequency матрицу
    """
    tf_matrix = []
    for vector in count_matrix:
        cur_v = []
        vect_sum = sum(vector)
        for el in vector:
            cur_v.append(round(el / vect_sum, 3))
        tf_matrix.append(cur_v)
    return tf_matrix


def idf_transform(count_matrix: List[list]) -> list:
    """
    Возвращает idf матрицу
    """
    idf_matrix = []
    number_of_docs = len(count_matrix)
    number_of_words = len(count_matrix[0])
    # цикл по словам
    for col in range(number_of_words):
        docs_with_word = 0
        # цикл по векторам
        for vector in count_matrix:
            if vector[col] > 0:
                docs_with_word += 1
        cur_idf = round(
            math.log((number_of_docs + 1) / (docs_with_word + 1)) + 1, 2
            )
        idf_matrix.append(cur_idf)

    # в идеале тут транспонировать нужно, так будет быстрее
    return idf_matrix


if __name__ == '__main__':
    # Задача 1: tf
    count_matrix = [
        [1, 1, 2, 0, 0],
        [0, 0, 1, 2, 1]
    ]
    tf_matrix = tf_transform(count_matrix)
    assert tf_matrix == [
        [0.25, 0.25, 0.5, 0, 0],
        [0, 0, 0.25, 0.5, 0.25]
    ]

    count_matrix = [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
     ]
    tf_matrix = tf_transform(count_matrix)

    assert tf_matrix == [
        [0.143, 0.143, 0.286, 0.143, 0.143, 0.143, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.143, 0, 0, 0, 0.143, 0.143, 0.143, 0.143, 0.143, 0.143]
        ]
    # Задача 2: idf
    count_matrix = [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
     ]
    idf_matrix = idf_transform(count_matrix)
    print(idf_matrix)

