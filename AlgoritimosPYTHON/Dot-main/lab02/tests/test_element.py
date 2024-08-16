import sys

sys.path.append("..")

# importing
from lista_array import ArrayList


def test_vazio():
    lst = ArrayList()
    assert lst.esta_vazio() == True
    lst.inserir(1)
    assert lst.esta_vazio() == False
    lst.inserir(2)
    assert lst.esta_vazio() == False

def test_tamanho_1():
    lst = ArrayList()
    assert lst.tamanho() == 0
    lst.inserir(1)
    assert lst.tamanho() == 1
    lst.inserir(2)
    assert lst.tamanho() == 2

def test_tamanho_2():
    lst = ArrayList()
    assert lst.tamanho() == 0
    lst.inserir(1)
    assert lst.tamanho() == 1
    lst.inserir(2)
    assert lst.tamanho() == 2
    lst.remove(1)
    assert lst.tamanho() == 1
    lst.limpa()
    assert lst.tamanho() == 0

def test_procurar():
    lst = ArrayList()
    assert lst.procura(1) == False
    lst.inserir(1)
    assert lst.procura(1) == True
    assert lst.procura(2) == False
    lst.inserir(2)
    assert lst.procura(2) == True
    assert lst.procura(3) == False
    lst.remove(1)
    assert lst.procura(1) == False
    lst.limpa()
    assert lst.procura(2) == False
    assert lst.procura(1) == False
    assert lst.procura(3) == False
    assert lst.procura(4) == False

def test_indice_de():
    lst = ArrayList()
    assert lst.indice_de(1) == -1
    lst.inserir(1)
    assert lst.indice_de(1) == 0
    assert lst.indice_de(2) == -1
    lst.inserir(2)
    assert lst.indice_de(2) == 1
    assert lst.indice_de(3) == -1
    lst.remove(1)
    assert lst.indice_de(1) == -1
    assert lst.indice_de(2) == 0
    lst.limpa()
    assert lst.indice_de(2) == -1

def test_recupera_indice():
    lst = ArrayList()
    assert lst.recupera_indice(0) == None
    lst.inserir(1)
    assert lst.recupera_indice(0) == 1
    assert lst.recupera_indice(1) == None
    lst.inserir(2)
    assert lst.recupera_indice(0) == 1
    assert lst.recupera_indice(1) == 2
    assert lst.recupera_indice(2) == None
    lst.remove(1)
    assert lst.recupera_indice(0) == 2
    assert lst.recupera_indice(1) == None
    lst.limpa()
    assert lst.recupera_indice(0) == None
    assert lst.recupera_indice(1) == None
    assert lst.recupera_indice(2) == None