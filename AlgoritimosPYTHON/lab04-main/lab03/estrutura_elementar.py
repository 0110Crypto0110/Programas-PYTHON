from abc import ABC, abstractmethod


class estrutura_elementar(ABC):
    @abstractmethod
    def esta_vazio(self) -> bool:
        pass

    @abstractmethod
    def inserir_inicio(self, item):
        pass

    @abstractmethod
    def inserir_fim(self, item):
        pass

    @abstractmethod
    def remove(self, item):
        pass

    @abstractmethod
    def tamanho(self) -> int:
        pass

    @abstractmethod
    def limpa(self):
        pass

    @abstractmethod
    def procura(self, item) -> bool:
        pass

    @abstractmethod
    def indice_de(self, item) -> int:
        pass

    @abstractmethod
    def recupera_indice(self, index):
        pass
