import sys

sys.path.append("..")

# importing
from ordenador import Sorter


def test_vazio():
    lst = Sorter()
    resposta = lst.ordene_selection([5, 2, 3, 4, 1])
    assert resposta == [1, 2, 3, 4, 5]
    resposta = lst.ordene_insertion([5, 2, 3, 4, 1])
    assert resposta == [1, 2, 3, 4, 5]
