from estrutura_elementar import estrutura_elementar


class ArrayList(estrutura_elementar):
    def __init__(self):
        self.lista1 = []
        #------------------------

    def esta_vazio(self) -> bool:
        if len(self.lista1) == 0:
            return True
        else:
            return False
        #------------------------

    def inserir(self, item):
       self.lista1.append(item)


        #------------------------
    def remove(self, item):
       self.lista1.remove(item)


        #------------------------

    def tamanho(self) -> int:
       return len(self.lista1)

        #------------------------

    def limpa(self):
        self.lista1.clear()

       #-------------------------

    def procura(self, item) -> bool:
        for i in range(len(self.lista1)):
            if self.lista1[i] == item:
                return True
        return False

        #------------------------

    def indice_de(self, item):
        for i in range(len(self.lista1)):
            if self.lista1[i] == item:
                return i
        return -1
        #-----------------------

    def recupera_indice(self, index):
        if len(self.lista1) == 0:
           return None
           
        if len(self.lista1) < index:
            return None
            
        if len(self.lista1) > index:
            return self.lista1[index]
                
         
        
        
        
        
        
            
        

        