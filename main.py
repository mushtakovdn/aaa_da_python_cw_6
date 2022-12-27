from os import stat
from typing import List, Dict, Iterable
import string

from utils import tf_transform, idf_transform


class TfidfTransformer():

    def __init__(self):
        self._tf_idf_matrix = None

    def fit_transform(self, count_matrix: List[list]) -> List[list]:
        self._tf_idf_matrix = []
        tf = tf_transform(count_matrix)
        idf = idf_transform(count_matrix)

        for vector in tf:
            cur_tfidf = []
            for i, word_tf in enumerate(vector):
                cur_tfidf.append(round(word_tf * idf[i], 3))
            self._tf_idf_matrix.append(cur_tfidf)
        return self._tf_idf_matrix


class CountVectorizer:
    """
    Класс для векторизации текстов
    """

    def __init__(self):
        self._vocabulary = {}
        self.count_matrix = []

    def get_feature_names(self, ):
        """Возвращет список всех слов, которые были в корпусе текста"""
        return list(self._vocabulary.keys())

    def _fill_vectors(self):
        """
        После обучения на всех документах из обучающего корпуса
        заполняет все промежуточные вектора до полной длины
        """
        for vector in self.count_matrix:
            vector_shortage = len(self._vocabulary) - len(vector)
            vector.extend([0 for i in range(vector_shortage)])

    def _process_tokens(self, tokens: List[str]) -> Dict[str, int]:
        """
        Выполняет с текущим списком токенов два действия:
        1. обновляет перечень слов присутствующих в корпусе текста
        {слово: номер столбца в матрице}
        2. Возвращает счетчик - словарь {слово: количество вхождений}

        Два действия объеденены в один метод, чтобы не итерироваться дважды
        """
        counter = {}
        for word in tokens:
            # 1
            if word not in self._vocabulary:
                self._vocabulary[word] = len(self._vocabulary)
            # 2
            counter[word] = counter.get(word, 0) + 1
        return counter

    def _get_interim_vector(self, tokens: List[str]) -> List[int]:
        """
        Возвращает промежуточный вектор.
        Длина вектора равна количеству уникальных слов найденных в обучающем
        корпусе текста на текущим момент
        """
        counter = self._process_tokens(tokens)
        vector = [0 for w in self._vocabulary]  # инициализация вектора
        for word, counts in counter.items():
            position = self._vocabulary[word]
            vector[position] = counts
        return vector

    def _tokenize(self, doc: str) -> List[str]:
        """
        Принимает на вход один документ (строку) и токенизирует
        """
        doc = doc.lower()
        # remove punctuation
        for symbol in string.punctuation:
            doc = doc.replace(symbol, '')
        return doc.split()

    def fit_transform(self, raw_documents: Iterable[str]) -> List[int]:
        """
        Обучение на корпусе текста.
        Затем преобразование корпуса текста в вектора
        """
        for cur_doc in raw_documents:
            tokens = self._tokenize(cur_doc)
            vector = self._get_interim_vector(tokens)
            self.count_matrix.append(vector)
        self._fill_vectors()
        return self.count_matrix


class TfidfVectorizer(CountVectorizer):

    def __init__(self):
        super().__init__()
        self._tfidf_transformer = TfidfTransformer()

    def fit_transform(self, corpus: Iterable[str]) -> List[int]:
        count_matrix = super().fit_transform(corpus)
        return self._tfidf_transformer.fit_transform(count_matrix)


if __name__ == '__main__':
    # corpus = [
    #     'Crock Pot Pasta Never boil pasta again',
    #     'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    # ]
    # vectorizer = CountVectorizer()
    # count_matrix = vectorizer.fit_transform(corpus)
    # assert vectorizer.get_feature_names() == [
    #     'crock', 'pot', 'pasta', 'never', 'boil', 'again', 'pomodoro',
    #     'fresh', 'ingredients', 'parmesan', 'to', 'taste'
    #     ]
    # assert count_matrix == [
    #     [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    #     ]
    # print("CountVectoriser tests paseed correctly")
    # # задача 3
    # count_matrix = [
    #     [1,1,2,1,1,1,0,0,0,0,0,0],
    #     [0,0,1,0,0,0,1,1,1,1,1,1]
    # ]
    # transformer = TfidfTransformer()
    # tfidf_matrix = transformer.fit_transform(count_matrix)
    # print(tfidf_matrix)
    # #  Out: [[0.2, 0.2, 0.286, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0],
    # #        [0, 0, 0.143, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]]

    # Задача 4
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(tfidf_matrix)
    # Out:['crock', 'pot', 'pasta', 'never', 'boil', 'again', 'pomodoro', 'fresh', 'ingredients', 'parmesan', 'to', 'taste']
    # Out: [[0.2, 0.2, 0.286, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0],
    #       [0, 0, 0.143, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]]
