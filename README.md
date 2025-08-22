# Prova 2 - Python

Segunda prova de introdução aos softwares estatísticos - UFPB \
Aluno: André Werlang Menegazzo

## Objetivos
O objetivo deste trabalho consiste na criação de uma biblioteca simples em 
python que apresente funcionalidades da biblioteca Numpy, especialmente a
manipulação de arrays, álgebra linear e geração de números pseudo-aleatórios. 

Além disso, este projeto tem como objetivo a prática de gestão de projetos 
de python, como a criação e versionamento de bibliotecas.

## Funcionalidades implementadas
1. Função para gerar dados aleatórios conjuntamente uniformes entre 0 e 1, em dimensão (n,p)

 ```python   
rand_multivariate_uniform()
```
2. Função que calcula a inversa generalizada de Moore-Penrose para de uma matriz.

```python
moore_penrose_gen_inv()
```

3. Função que calcula os coeficientes de uma regressão linear com a inversa generalizada de Moore-Penrose ao invés da inversa convencional. 

```python
betas_linreg_moore_penrose()
```

## Exemplos de uso
   
```python 
import numpy as np
from est_numpy_awm import funcoes
```

1. Gerar amostra uniforme multivariada
    
```python
X = funcoes.rand_multivariate_uniform(n=100, p=3, seed=42)
print(X)
```

2. Calcular a pseudoinversa de Moore-Penrose

```python
A = np.array([[1, 2], [3, 4]])
A_gen_inv = funcoes.moore_penrose_gen_inv(A)
print(A_gen_inv)
```


3. Estimar coeficientes de regressão linear

```python
y = np.array([1, 2])
beta = funcoes.linear_regression_coefficients(A, y)
print(beta)
```

## Como instalar
Para instalar esta biblioteca:

    pip install git+https://github.com/Andre-WM/Prova2_python_andre.git
