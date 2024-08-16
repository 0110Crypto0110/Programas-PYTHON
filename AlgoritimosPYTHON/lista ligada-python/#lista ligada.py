

''' programa de um lista ligada em python'''



#__________________________________________________________________
''' define a estrura do no'''
#__________________________________________________________________
class No:  
    def __init__(self, valor):
        self.valor = valor
        self.proximo = None

    def set_proximo(self, no):
        self.proximo = no

    def get_proximo(self):
        return self.proximo


#__________________________________________________________________
''' define a estrura da lista ligada'''
#__________________________________________________________________
class LinkedList(): 
    def __init__(self):
        self.inicio = None


#__________________________________________________________________
    ''' metodo inserir_inicio  '''
#__________________________________________________________________
    '''Este método insere um valor no início da lista.'''
    '''É útil quando você precisa adicionar um elemento no início da lista sem se preocupar com a ordem dos elementos.'''
    def inserir_inicio(self, valor): 
        if self.inicio is None:
            self.inicio = No(valor)
        else:
            novoNo = No(valor)
            novoNo.set_proximo(self.inicio)
            self.inicio = novoNo
        return 
    
#__________________________________________________________________
    ''' metodo inserir_inicio  '''
#__________________________________________________________________
    '''Este método insere um valor em uma posição específica da lista.'''
    '''É útil quando você precisa inserir um elemento em uma posição específica da lista.'''
    def inserir_meio(self, Novo_valor, index): 
            novo_no = No(Novo_valor)
            procurado = self.inicio
            cont = 0
            while procurado.get_proximo() is not None:
                if cont == index-1:
                    novo_no.set_proximo(procurado.get_proximo())
                    procurado.set_proximo(novo_no)
                    break
                    
                cont = cont + 1
                procurado = procurado.proximo
            return 
        
#__________________________________________________________________
    ''' metodo inserir_fim  '''
#__________________________________________________________________
    '''Este método insere um valor no final da lista.'''
    '''É útil quando você precisa adicionar um elemento ao final da lista sem se preocupar com a ordem dos elementos.'''   
    def inserir_fim(self, valor): 
        if self.inicio is None:
            self.inicio = No(valor)
        else:
            novoNo = No(valor)
            aux = self.inicio
            while aux.get_proximo() is not None:
                aux = aux.get_proximo()
            aux.set_proximo(novoNo)
        return 

#__________________________________________________________________
    ''' metodo esta_vazio  '''
#__________________________________________________________________
    '''Este método verifica se a lista está vazia.'''
    '''É útil quando você precisa verificar se a lista não contém nenhum elemento.'''
    def esta_vazio(self) -> bool: 
        if self.inicio is None:
            return True
        else:
            return False
        
        #return self.inicio is None

#__________________________________________________________________
    ''' metodo remove  '''
#__________________________________________________________________
    '''Este método remove um valor específico da lista (o primeiro encontrado).'''
    '''É útil quando você precisa remover um elemento específico da lista.'''
    def remove(self, item): 
        procurado1 = self.inicio
        if procurado1 is not None:
            if procurado1.valor == item:
                self.inicio = procurado1.proximo
                procurado1 = None
                return
            
            while procurado1 is not None:
                if procurado1.valor == item:
                    break
                memoria = procurado1
                procurado1 = procurado1.proximo
                
            if procurado1 == None:
                return
            memoria.proximo = procurado1.proximo
            procurado1 = None
        return 
                  
#__________________________________________________________________
    ''' metodo tamanho  '''
#__________________________________________________________________  
    '''Este método retorna o tamanho da lista.'''
    '''É útil quando você precisa saber quantos elementos a lista contém.'''
    def tamanho(self) -> int: 
        if self.inicio is None:
            return 0
        else:
            cont = 1
            aux = self.inicio
            while aux.get_proximo() is not None:
                cont = cont+1
                aux = aux.get_proximo()
                
            return cont

#__________________________________________________________________
    ''' metodo limpa  '''
#__________________________________________________________________ 
    '''Este método apaga todos os nós da lista.'''
    '''É útil quando você precisa limpar a lista e liberar memória.'''
    def limpa(self): 
        while self.inicio is not None:
            A = self.inicio
            self.inicio = self.inicio.proximo
            A = None
        return
                
#__________________________________________________________________
    ''' metodo procura  '''
#__________________________________________________________________ 
    '''Este método verifica se um determinado valor está na lista.'''
    '''É útil quando você precisa verificar se um elemento está presente na lista.'''
    def procura(self, item) -> bool: 
       
        procurado = self.inicio
        index=0
        while procurado:
            if procurado.valor == item:

                return True

            else:
                procurado = procurado.proximo
                index = index + 1

        return False
       

#__________________________________________________________________
    ''' metodo indice_de  '''
#__________________________________________________________________ 
    '''Este método retorna o índice do primeiro elemento com um determinado valor na lista.'''
    '''É útil quando você precisa encontrar a posição de um elemento na lista.'''
    def indice_de(self, item): 
        if self.tamanho() == 0:
            return -1
        
        else:
            procurado = self.inicio
            indice=0
            while procurado:
                if procurado.valor == item:
                
                    return indice

                procurado = procurado.proximo
                indice = indice + 1
        return -1

         

#__________________________________________________________________
    ''' metodo recupera_indice  '''
#__________________________________________________________________ 
    '''Este método retorna o valor de um elemento em uma determinada posição na lista.'''
    '''É útil quando você precisa acessar um elemento em uma posição específica na lista.'''
    def recupera_indice(self, index):
        if self.tamanho() > index:
            procurado = self.inicio
            cont = 0
            while procurado.get_proximo() is not None:
                if cont == index:
                    return procurado.valor
                
                cont = cont + 1
                procurado = procurado.proximo
            return procurado.valor
        
        return None
    

#__________________________________________________________________
    ''' metodo transforma_em_lista  '''
#__________________________________________________________________ 
    '''Este método transforma a lista ligada em uma lista comum (Python list).'''
    '''É útil quando você precisa trabalhar com a lista em um formato mais comum do Python.'''
    def transforma_em_lista(self): 
        novaLista = []
        if self.inicio is None:
            return novaLista
        else:
            aux = self.inicio
            while aux is not None:
                novaLista.append(aux.valor)
                aux = aux.get_proximo()
        
        return novaLista
        


#__________________________________________________________________
    '''Exemplos de funcionamento  '''
#__________________________________________________________________ 

#-----------------------------------------------------
#lista_ligada = LinkedList()
#inserir_inicio
#inserir_fim
#esta_vazio
#remove
#tamanho
#limpa
#procura
#indice_de (item)pequisa o index do item
#recupera_indice(index) retorna o valor do index
#tranforma_lista


#chama a funcao que adiciona valores
#lista_ligada.inserir_inicio()
#lista_ligada.inserir_meio()
#lista_ligada.inserir_fim()

#chama a funcao que apaga valores
#lista_ligada.remove()
#lista_ligada.limpa() #apaga a lista

# funcao que mensura os valores 
#lista_ligada.tamanho()
#lista_ligada.esta_vazio() 
#lista_ligada.indice_de()
#lista_ligada.recupera_indice()
#lista_ligada.transforma_em_lista()


#__________________________________________________________________
    '''Exemplos praticos de funcionamento  '''
#__________________________________________________________________ 

lista_ligada = LinkedList()

lista=[4,1,3,6,5,6,8,-10,9,11,13,22,10,20,35,44,55,78,80,6]
for i in range(len(lista)):
    if i == 1:
        lista_ligada.inserir_inicio(lista[i-1])
    else: 
        lista_ligada.inserir_fim(lista[i-1])
print(lista_ligada.transforma_em_lista())
lista_ligada.inserir_meio(-77, 5)
print(lista_ligada.transforma_em_lista())
lista_ligada.inserir_meio(-88, 2)
print(lista_ligada.transforma_em_lista())
lista_ligada.remove(10)
print(lista_ligada.transforma_em_lista())

#zerando a lista_____________________________________
lista_ligada.limpa()
print(lista_ligada.transforma_em_lista())




#__________________________________________________________________
'''lendo um arquivo de texto externo1  '''
#__________________________________________________________________ 
#o programa le as instrucoes do arquivo em texto da seguinto forma
#primeira linha lista inicial de valores
#segunda linha quantidade de operacoes
# a proximas linhas sao as operacoes em si
lista_ligada = LinkedList()
print("________________________________________________")
print("lendo um arquivo de texto externo")
print("________________________________________________")


arquivo = open("arq1.txt", "r")# arquivo torna-se uma lista, atributos( nome do arquivo, funcao ler)
linhas = arquivo.readlines() # linhas se torna uma lista de strings
print("________________________________________________")
print("quantidade de linhas do arquivo:",len(linhas))# quantidade de linhas
print("________________________________________________")


#define o intervalo da lista e numero de metodos
primeiro_indice = 0 # indice da lista inicial no arquivo
segundo_indice = 1 # indice da quantidade de metodos
partes = linhas[1].strip('\ufeff').split() #salva linha de inndice 1 
Numero = int(partes[0]) #salva o valor de posicao 0 da linha


#iteracao para inserir a lista que esta no arquivo.txt
for i in range(primeiro_indice, segundo_indice ):
    partes = linhas[i].strip('\ufeff').split()
    
    for i in range(len(partes)) :
        lista_ligada.inserir_fim(int(partes[ i ]))

print("________________________________________________")
print(f"numero de metodos a serem chamados = {linhas[1]} ")
print("________________________________________________")


#iteracao para representar as o arquivo de forma indexada
for indice2, linha in enumerate(linhas):
    print(f"Índice: {indice2}, Linha: {linha}")
    

#define o intervalo dos metodos
inicio_indice = 2 #define o valor inicial do for
ultimo_indice = 2 + Numero #define o valor final 
print("________________________________________________")
print()


for j in range(inicio_indice, ultimo_indice ):
    partes = linhas[j].strip('\ufeff').split()#remove a particula \ufeff 
    for i in range(len(partes)) :
        if partes[i].isalpha():#checa se é letra
            if partes[i] == "A": 
                valor = int(partes[i+1]) #proxima posicao depois de A
                posicao = int(partes[i+2])#a proxima da proxima de A
                lista_ligada.inserir_meio(valor, posicao)
            if partes[i] == "R":
                valor = int(partes[i+1])
                lista_ligada.remove(valor)
            if partes[i] == "P":
                print(lista_ligada.transforma_em_lista())
                print()

lista_ligada.limpa()
#__________________________________________________________________
'''lendo um arquivo de texto externo2  '''
#__________________________________________________________________       
lista_ligada = LinkedList()

print("________________________________________________")
print("lendo um arquivo de texto externo")
print("________________________________________________")


arquivo1 = open("arq.txt", "r")# arquivo torna-se uma lista, atributos( nome do arquivo, funcao ler)
linhas = arquivo1.readlines() # linhas se torna uma lista de strings
print("________________________________________________")
print("quantidade de linhas do arquivo:",len(linhas))# quantidade de linhas
print("________________________________________________")


#define o intervalo da lista e numero de metodos
primeiro_indice = 0 # indice da lista inicial no arquivo
segundo_indice = 1 # indice da quantidade de metodos
partes = linhas[1].strip('\ufeff').split() #salva linha de inndice 1 
Numero = int(partes[0]) #salva o valor de posicao 0 da linha


#iteracao para inserir a lista que esta no arquivo.txt
for i in range(primeiro_indice, segundo_indice ):
    partes = linhas[i].strip('\ufeff').split()
    
    for i in range(len(partes)) :
        lista_ligada.inserir_fim(int(partes[ i ]))

print("________________________________________________")
print(f"numero de metodos a serem chamados = {linhas[1]} ")
print("________________________________________________")


#iteracao para representar as o arquivo de forma indexada
for indice2, linha in enumerate(linhas):
    print(f"Índice: {indice2}, Linha: {linha}")
    

#define o intervalo dos metodos
inicio_indice = 2 #define o valor inicial do for
ultimo_indice = 2 + Numero #define o valor final 
print("________________________________________________")
print()


for j in range(inicio_indice, ultimo_indice ):
    partes = linhas[j].strip('\ufeff').split()#remove a particula \ufeff 
    for i in range(len(partes)) :
        if partes[i].isalpha():#checa se é letra
            if partes[i] == "A": 
                valor = int(partes[i+1]) #proxima posicao depois de A
                posicao = int(partes[i+2])#a proxima da proxima de A
                lista_ligada.inserir_meio(valor, posicao)
            if partes[i] == "R":
                valor = int(partes[i+1])
                lista_ligada.remove(valor)
            if partes[i] == "P":
                print(lista_ligada.transforma_em_lista())
                print()


#lista_ligada.limpa()
     
#__________________________________________________________________
    
'''usado para diferenciar letras e numero'''
#__________________________________________________________________
'''if j.isalpha():
            print("É uma letra.")
        elif j.startswith('-') and j[1:].isdigit():
            print("É um número negativo.")
        elif j.isdigit():
            print("É um número positivo.")
        else:
            print("Não é uma letra nem um número.")
        #lista_ligada.inserir_fim(caractere)'''

#__________________________________________________________________
    
'''usado para remover a particula \ufeff'''
#__________________________________________________________________
'''strip('\ufeff')'''