class Sorter:
    def __init__(self):
        pass

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
            
            
            
        
        
