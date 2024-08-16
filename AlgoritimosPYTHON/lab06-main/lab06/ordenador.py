class Sorter:
    def __init__(self):
        pass

#--------------------------------------- 
def ordene_insertion_Letras(valores):
    for i in range(1, len(valores)):
        posicao1 = valores[i]
        j = i - 1

        while j >= 0 and posicao1 < valores[j]:
            valores[j + 1] = valores[j]
            j -= 1
        
        valores[j + 1] = posicao1
    
    return valores
    
    

    
#---------------------------------------



#-------------------------------------


def ordene_selection_LINKEDLIST(self):
        if self.inicio is None:
            return
        
        else:
            cont = 1
            aux = self.inicio
            while aux.get_proximo() is not None:
                cont = cont+1
                aux = aux.get_proximo()
                      
        
            primeiroNO = self.inicio
            
            for i in cont:
                MenorNo = primeiroNO
                ProximoNo = primeiroNO.proximo

                j = i +1
                for j in cont:
                    if ProximoNo.valor < MenorNo.valor:
                        MenorNo = ProximoNo
                    ProximoNo = ProximoNo.proximo
            
                if MenorNo != primeiroNO:
                    primeiroNO.valor = MenorNo.valor,
                    MenorNo.valor = primeiroNO.valor
            
                primeiroNO = primeiroNO.proximo

            
            
#-------------------------------------      
      
      
      
      
#---------------------------------------------------     
def ordene_insertion_Recursivo(self, valores, tamanho):
    if tamanho <= 1:
        return

    ordene_insertion_Recursivo(self, valores, tamanho - 1)
    posição1 = tamanho - 1
    valor_inserir = valores[posição1]

    j = posição1 - 1
    while j >= 0 and valores[j] > valor_inserir:
        valores[j + 1] = valores[j]
        j = j - 1

    valores[j + 1] = valor_inserir
    

def ordene_insertion(self, valores) :
        for i in range( 0 ,len(valores) ):
            
            posiçao1 = valores[i]
            
            for j in range( len(valores) ):

                posiçao2 = valores[j]

                if posiçao1 < posiçao2:

                    memoria = valores[i]
                    valores[i] = valores[j]
                    valores[j] = memoria
        return valores
#--------------------------------------------




  
      
#----------------------------------------------        
def ordene_selection_Recursivo(self, valores):
    if len(valores) <= 1:
        return valores
    
    menor = valores[0]
    indice_menor = 0

    for i, elemento in enumerate(self, valores):
        if elemento < menor:
            menor = elemento
            indice_menor = i
    
    valores[0], valores[indice_menor] = valores[indice_menor], valores[0]
    
    return [valores[0]] + ordene_selection_Recursivo(valores[1:])
    

def ordene_selection(self, valores) :
        valores2 = []
    
        while valores:
            menor = valores[0]
            indice_menor = 0
        
            for i, elemento in enumerate(valores):
                if elemento < menor:
                    menor = elemento
                    indice_menor = i
    
            valores2.append(menor)
            valores.pop(indice_menor)
    
        return valores2
#-------------------------------------------------------