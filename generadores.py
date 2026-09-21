"""
generadores.py
--------------
Implementa los dos métodos de generación de variables aleatorias
comparados en el proyecto, ambos generando muestras de
X ~ Binomial Negativa(r, p) (número de fallos antes del r-ésimo gol):

    GeneradorBernoulliSecuencial
        Simula tiro por tiro con ensayos Bernoulli hasta acumular
        r goles. Es el método "de definición", directo pero costoso.

    GeneradorTransformadaInversa
        Escribe X como la suma de r variables Geométricas(p) i.i.d.,
        cada una generada por transformada inversa, y vectoriza la
        generación con NumPy.

Ambas clases heredan de GeneradorVariableAleatoria y exponen el mismo
método `generar(size)`, de modo que el resto del programa las puede
usar de forma intercambiable (polimorfismo).
"""

from abc import ABC, abstractmethod
import numpy as np

from modelo import ModeloBinomialNegativa


class GeneradorVariableAleatoria(ABC):
    """Interfaz común para los métodos de generación de X ~ NB(r, p)."""

    nombre_metodo: str = "Genérico"

    def __init__(self, modelo: ModeloBinomialNegativa, rng: np.random.Generator):
        self.modelo = modelo
        self.rng = rng

    @abstractmethod
    def generar(self, size: int) -> np.ndarray:
        """Devuelve un arreglo de `size` observaciones de X (fallos)."""
        raise NotImplementedError

    def generar_tiros_totales(self, size: int) -> np.ndarray:
        """Tiros totales = fallos generados + r goles."""
        return self.generar(size) + self.modelo.r


class GeneradorBernoulliSecuencial(GeneradorVariableAleatoria):
    """
    Método 1: simulación secuencial de ensayos Bernoulli.

    Para cada observación se generan tiros uno a uno con
    U ~ Uniforme(0,1): si U < p se cuenta un gol, si no un fallo.
    Se repite hasta acumular r goles y se registra el número de
    fallos ocurridos.
    """

    nombre_metodo = "Bernoulli secuencial"

    def generar(self, size: int) -> np.ndarray:
        r, p = self.modelo.r, self.modelo.p
        fallos = np.empty(size, dtype=np.int64)
        for i in range(size):
            exitos = 0
            f = 0
            while exitos < r:
                u = self.rng.random()
                if u < p:       # gol
                    exitos += 1
                else:            # fallo
                    f += 1
            fallos[i] = f
        return fallos


class GeneradorTransformadaInversa(GeneradorVariableAleatoria):
    """
    Método 2: transformada inversa sobre variables Geométricas.

    Una NB(r, p) se puede escribir como la suma de r variables
    Geométricas(p) independientes (número de fallos antes de un
    único éxito). Cada Geométrica se genera invirtiendo su función
    de distribución acumulada:

        F(x) = 1 - (1-p)^(x+1)  =>  x = floor( ln(U) / ln(1-p) )
    """

    nombre_metodo = "Transformada inversa"

    def _geometrica_transformada_inversa(self, size: int) -> np.ndarray:
        p = self.modelo.p
        u = self.rng.random(size)
        return np.floor(np.log(u) / np.log(1 - p)).astype(np.int64)

    def generar(self, size: int) -> np.ndarray:
        total = np.zeros(size, dtype=np.int64)
        for _ in range(self.modelo.r):
            total += self._geometrica_transformada_inversa(size)
        return total
