from lista_ligada import LinkedList



class pilha(LinkedList):
    def _init_(self):
        self.pilha = LinkedList()
        
    def push(self):
        self.pilha.inserir_fim(valor)

    
    def pop(self ) -> int:
        self.pilha.tamanho = indice
        indice = indice -1
        self.pilha.remove_indice(indice)
        
    def search (self):
        self.pilha.procura(valor)

        


class fila(LinkedList):
    def _init_(self):
        self.fila = LinkedList()

    def push(self):
        self.fila.inserir_fim(valor)
        
    def pop(self) -> int:
        self.fila.remove_indice(0)
        
    def search (self):
        self.fila.procura()
        

        pass


class fila_pri(LinkedList):
    def _init_(self):

        pass
    def push(n):

        pass
    def pop() -> int:

        pass
    def search (n):

        pass