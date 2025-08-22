import numpy as np
import pytest

from est_numpy_awm import rand_multivariate_uniform, moore_penrose_gen_inv, betas_linreg_moore_penrose


class TestRandMultivariateUniform:
    """Testes para a função rand_multivariate_uniform"""

    def test_correct_shape(self):
        """Testa se o shape da saída está correto"""
        n, p = 100, 5
        resultado = rand_multivariate_uniform(n, p, seed=42)
        assert resultado.shape == (n, p)

    def test_valores_no_intervalo(self):
        """Testa se todos os valores estão entre 0 e 1"""
        dados = rand_multivariate_uniform(50, 3, seed=42)
        assert np.all(dados >= 0) and np.all(dados <= 1)

    def test_reprodutibilidade(self):
        """Testa se a seed garante reprodutibilidade"""
        dados1 = rand_multivariate_uniform(100, 4, seed=123)
        dados2 = rand_multivariate_uniform(100, 4, seed=123)
        np.testing.assert_array_equal(dados1, dados2)

    def test_valores_padrao(self):
        """Testa os valores padrão dos parâmetros"""
        dados = rand_multivariate_uniform()
        assert dados.shape == (1, 1)
        assert 0 <= dados[0, 0] <= 1

    def test_erro_valores_negativos(self):
        """Testa se levanta erro para valores negativos"""
        with pytest.raises(ValueError, match="n e p devem ser números inteiros positivos"):
            rand_multivariate_uniform(-1, 5)
        with pytest.raises(ValueError, match="n e p devem ser números inteiros positivos"):
            rand_multivariate_uniform(10, -2)
        with pytest.raises(ValueError, match="n e p devem ser números inteiros positivos"):
            rand_multivariate_uniform(0, 5)
        with pytest.raises(ValueError, match="n e p devem ser números inteiros positivos"):
            rand_multivariate_uniform(5, 0)


class TestMoorePenroseGenInv:
    """Testes para a função moore_penrose_gen_inv"""

    def test_matriz_identidade(self):
        """Testa com matriz identidade"""
        I = np.eye(3)
        I_pinv = moore_penrose_gen_inv(I)
        np.testing.assert_allclose(I_pinv, I, atol=1e-12)

    def test_matriz_retangular(self):
        """Testa com matriz retangular"""
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        A_pinv = moore_penrose_gen_inv(A)

        # Verifica propriedades da pseudoinversa
        A_reconstruida = A @ A_pinv @ A
        np.testing.assert_allclose(A, A_reconstruida, atol=1e-10)

    def test_matriz_singular(self):
        """Testa com matriz singular"""
        A = np.array([[1, 2], [1, 2], [1, 2]], dtype=float)
        A_pinv = moore_penrose_gen_inv(A)

        # Deve conseguir calcular sem erro
        assert A_pinv.shape == (2, 3)

        # Verifica propriedade fundamental
        A_reconstruida = A @ A_pinv @ A
        np.testing.assert_allclose(A, A_reconstruida, atol=1e-10)

    def test_tolerancia(self):
        """Testa o parâmetro de tolerância"""
        A = np.array([[1e-13, 0], [0, 1]], dtype=float)
        A_pinv = moore_penrose_gen_inv(A, tol=1e-12)

        # O valor singular muito pequeno deve ser tratado como zero
        expected = np.array([[0, 0], [0, 1]])
        np.testing.assert_allclose(A_pinv, expected, atol=1e-12)


class TestBetasLinregMoorePenrose:
    """Testes para a função betas_linreg_moore_penrose"""

    def test_regressao_simples(self):
        """Testa regressão linear simples"""
        # y = 2x + 1
        X = np.array([[1], [2], [3]])
        y = np.array([3, 5, 7])

        beta = betas_linreg_moore_penrose(X, y)
        expected = np.array([1.0, 2.0])  # intercepto, slope
        np.testing.assert_allclose(beta, expected, atol=1e-12)

    def test_multiplas_covariaveis(self):
        """Testa regressão com múltiplas covariáveis"""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        beta_true = np.array([2.0, 1.5, -0.8, 3.2])
        y = X @ beta_true[1:] + beta_true[0] + np.random.randn(n) * 0.01

        beta_estimado = betas_linreg_moore_penrose(X, y)
        np.testing.assert_allclose(beta_estimado, beta_true, atol=0.1)

    def test_matriz_singular(self):
        """Testa regressão com matriz singular (colinearidade)"""
        # Colinearidade perfeita
        X = np.array([[1, 2], [2, 4], [3, 6]])  # segunda coluna = 2 * primeira
        y = np.array([1, 2, 3])

        # Não deve levantar exceção
        beta = betas_linreg_moore_penrose(X, y)
        assert beta.shape == (3,)  # intercepto + 2 coeficientes

    def test_erro_dimensoes_incompativeis(self):
        """Testa erro para dimensões incompatíveis"""
        X = np.random.randn(10, 3)
        y = np.random.randn(8)  # tamanho diferente

        with pytest.raises(ValueError, match="X e y devem ter o mesmo número de observações"):
            betas_linreg_moore_penrose(X, y)

    def test_intercepto_correto(self):
        """Testa se o intercepto está sendo calculado corretamente"""
        # Dados com intercepto não-zero
        X = np.array([[1], [2], [3]])
        y = np.array([4, 5, 6])  # y = x + 3

        beta = betas_linreg_moore_penrose(X, y)
        expected = np.array([3.0, 1.0])  # intercepto = 3, slope = 1
        np.testing.assert_allclose(beta, expected, atol=1e-12)


def test_integracao_completa():
    """Teste de integração de todas as funções"""
    # Gerar dados
    X = rand_multivariate_uniform(100, 3, seed=42)

    # Criar y com relação linear conhecida
    beta_true = np.array([2.0, 3.0, -1.0, 1.5])  # intercepto + 3 coefs
    y = X @ beta_true[1:] + beta_true[0] + np.random.randn(100) * 0.01

    # Calcular coeficientes
    beta_estimado = betas_linreg_moore_penrose(X, y)

    # Verificar se está próximo dos valores verdadeiros
    np.testing.assert_allclose(beta_estimado, beta_true, atol=0.1)