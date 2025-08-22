import numpy as np

def rand_multivariate_uniform(n : int = 1, p: int = 1, seed = None) -> np.ndarray:
    """"
    Gera pontos uniformemente distribuídos no hipercubo [0,1]^p.
    De forma equivalente, as distribuições marginais são Uniformes no intervalo [0,1].

    Args:
        - n: Tamanho da amostra a ser gerada;
        - p: Número de covariaveis na amostra a ser gerada;
        - seed: Semente para reprodutibilidade.

    Returns:
        Um array de dimensão (n x p) com os dados gerados.

    Raises:
        ValueError: Se n ou p não forem positivos.
    """

    if seed is not None:
        np.random.seed(seed)

    if n <= 0 or p <= 0:
        raise ValueError("n e p devem ser números inteiros positivos")

    return np.random.rand(n, p)


def moore_penrose_gen_inv(matrix_a: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
        Calcula a inversa generalizada de Moore-Penrose pela decomposição SVD.

        Args:
            matrix_a: Matriz de entrada (m x n)
            tol: Tolerância para valores singulares zero

        Returns:
            Matriz pseudoinversa de A (n x m)
        """

    U, s, Vt = np.linalg.svd(matrix_a, full_matrices=False)

    s_inv = np.zeros_like(s)
    s_nonzero = s > tol
    s_inv[s_nonzero] = 1 / s[s_nonzero]

    A_pinv = Vt.T @ (s_inv[:, None] * U.T)

    return A_pinv

def betas_linreg_moore_penrose(x: np.ndarray, y: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Calcula os coeficientes da regressão linear usando pseudoinversa de Moore-Penrose.

    Args:
        x: Matriz de covariáveis (n x p)
        y: Vetor de resposta (n,)
        tol: Tolerância para singularidades

    Returns:
        Vetor de coeficientes beta (p+1,), onde o primeiro coeficiente é o intercepto.

    Raises:
        ValueError: Se as dimensões de x e y não forem compatíveis.
    """

    if x.shape[0] != y.shape[0]:
        raise ValueError("X e y devem ter o mesmo número de observações")

    x_design = np.column_stack([np.ones(x.shape[0]), x])

    x_mp_inv = moore_penrose_gen_inv(x_design, tol=tol)
    beta = x_mp_inv @ y

    return beta