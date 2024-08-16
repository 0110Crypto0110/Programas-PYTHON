
''' Lista com interface no terminal'''



#define a lista inicial (lista especifica)
lista=[1,2,3,4,5,6,7,8,9,10]
#__________________________________________________________________
'''metodo inserir_final'''
#__________________________________________________________________
'''Este método insere um valor no final de uma lista específica.'''
'''É útil quando você precisa adicionar um elemento ao final da lista e visualizar a lista atualizada no terminal.'''
def inserir_final (lista, valor):
    lista.append(valor)
    print("---------------------------")
    print(lista)


#__________________________________________________________________
'''metodo inserir_meio'''
#__________________________________________________________________
'''Este método insere um valor em uma posição específica de uma lista específica.'''
'''É útil quando você precisa inserir um elemento em uma posição específica na lista e visualizar a lista atualizada no terminal. Se a posição especificada for maior que o tamanho atual da lista, uma mensagem de erro será exibida.'''
def inserir_meio(lista, valor, posicao):
    if posicao > len(lista):    #se a posicao exceder o tamanho, ou seja uma posicao a mais que a ultima levanta erro
        print("---------------------------")
        print("erro: a posicao escolhida excede o os indices da lista")
    if posicao <=  len(lista):  #checa se a posicao é esta dentro do tamanho da lista
        for i in range( len(lista) ):
            if i == posicao:
                lista.insert(posicao, valor)
        print(lista)
    if posicao == len(lista):    #checa se a posicao procurada é a ultima, se for chama a funcao inserir fim
        inserir_final(valor)
        print("---------------------------")
        print(lista)


#__________________________________________________________________
'''metodo apaga_valor'''
#__________________________________________________________________
'''Este método apaga um valor específico da lista.'''
'''É útil quando você precisa remover um elemento específico da lista e visualizar a lista atualizada no terminal. Se o valor especificado não existir na lista, uma mensagem de erro será exibida.'''
def apaga_valor(lista, valor):
    for i in range( len(lista) ):
        if lista[i-1] == valor: # o [i-1] ajusta os valores do for ao da lista
            E = lista[i-1]
            lista.remove(E)
            print("---------------------------")
            print(lista)
            break # se o valor for encontrado e removido, imprime a lista e finaliza a execucao
    print("---------------------------")
    print("erro: valor nao encontrado") # caso contrario levanta erro


#__________________________________________________________________
'''metodo procura_valor'''
#__________________________________________________________________
'''Este método procura um valor na lista e retorna o índice desse valor.'''
'''É útil quando você precisa encontrar a posição de um elemento na lista e visualizar o índice no terminal. Se o valor especificado não existir na lista, uma mensagem de erro será exibida.'''
def procura_valor (lista, valor):
    for i in range( len(lista) ):
        if lista[i-1] == valor: # o [i-1] ajusta os valores do for ao da lista
            print("indice:",i-1)
            print("lista:", lista)
            break
    print("---------------------------")
    print("valor nao existe na lista",valor)
    print("lista", lista)

#__________________________________________________________________
'''metodo procura_indice'''
#__________________________________________________________________
'''Este método retorna o valor em uma posição específica da lista.'''
'''É útil quando você precisa recuperar o valor em uma posição específica da lista e visualizar o valor no terminal. Se a posição especificada for maior que o tamanho atual da lista, uma mensagem de erro será exibida.'''
def procura_indice (lista, posicao):
    for i in range( len(lista)+1 ):
        if i-1 == posicao: # o [i-1] ajusta os valores do for ao da lista
            print(lista[i-1])#0 como posicao inicial


#__________________________________________________________________
'''interface do terminal'''
#__________________________________________________________________

print("---------------------")
print("Manipulador de listas")
print("---------------------")
print("---------------------")
print("lista =",lista)
print("---------------------")
print("---------------------")
print("1- para iniciar")
print("---------------------")
valor = input("inicie: ")
while int(valor) <= 6 and int(valor) >= 1: #escolhe a operacao e mantem a interacao enquanto nao receber nemnhum valor diferente das operacoes 
    print("---------------------")
    print("escolha a operacao:")
    print("1- inserir no fim")
    print("2-inserir no meio")
    print("3-apagar")
    print("4-procura valor")
    print("5-procura indice")
    print("qualquer outro valor - cancelar")
    print("---------------------")
    
    valor = input("operacao: ")
    if int(valor) == 1: # chama a funcao inserir final
        valor1 = input("Digite um valor para adicionar à lista: ")
        listaNova = inserir_final (lista, int(valor1) )

    if int(valor) == 2: # chama a funcao inserir meio
        valor2 = input("Digite um valor para adicionar à lista: ")
        posicao2 = input("Digite a posicao do valor: ")
        listaNova = inserir_meio(lista, int(valor2), int(posicao2))

    if int(valor) == 3: #chama a funcao apaga valor
        valor3 = input("Digite um valor para remover da lista: ")
        listaNova = apaga_valor(lista, int(valor3))

    if int(valor) == 4:# retorna o valor
        valor4 = input("Digite um valor para procurar lista: ")
        listaNova = procura_valor(lista, int(valor4))

    if int(valor) == 5: #retorna o indice
        posicao5 = input("Digite o inidce: ")
        listaNova = procura_indice(lista,int(posicao5))
print("-------------------")
print("operacao cancelada")
print("-------------------")
print("lista",lista)      

   