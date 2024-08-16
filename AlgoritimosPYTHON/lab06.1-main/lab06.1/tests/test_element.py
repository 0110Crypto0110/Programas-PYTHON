import sys

sys.path.append("..")

# importing
from ordenador_high import Sorter


def test_vazio():
    lst = Sorter()
    resposta, index = lst.particiona([5, 2, 3, 4, 1], 0, 4)
    assert resposta == [1, 2, 3, 4, 5]
    assert index == 4
    resposta, index = lst.particiona([2, 5, 1, 4, 3], 0, 4)
    assert resposta == [1, 2, 5, 4, 3]
    assert index == 1
    resposta = lst.merge([1, 2, 3], [4, 5])
    assert resposta == [1, 2, 3, 4, 5]
    resposta = lst.merge([5, 9, 10], [6, 7, 8])
    assert resposta == [5, 6, 7, 8, 9, 10]
