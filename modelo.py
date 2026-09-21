"""
modelo.py
---------
Clase que representa el modelo teórico del sistema: un delantero que
tira al arco con probabilidad de gol `p`, y necesita `r` goles.
X = número de fallos antes del r-ésimo gol ~ Binomial Negativa(r, p).
"""

from dataclasses import dataclass
from scipy import stats
import numpy as np


@dataclass(frozen=True)
class ModeloBinomialNegativa:
    """
    Modelo teórico de la variable X = número de fallos antes de
    conseguir r goles, con probabilidad de gol por tiro p.
    """
    r: int
    p: float
    nombre: str = ""

    def __post_init__(self):
        if not (0 < self.p < 1):
            raise ValueError("p debe estar en (0, 1)")
        if self.r <= 0:
            raise ValueError("r debe ser un entero positivo")

    # ---- momentos teóricos ----
    @property
    def media_fallos(self) -> float:
        """E[X] = r(1-p)/p"""
        return self.r * (1 - self.p) / self.p

    @property
    def varianza_fallos(self) -> float:
        """Var[X] = r(1-p)/p^2"""
        return self.r * (1 - self.p) / self.p ** 2

    @property
    def media_tiros(self) -> float:
        """E[T] = E[X] + r  (tiros totales = fallos + goles)"""
        return self.media_fallos + self.r

    def pmf(self, k):
        """pmf teórica evaluada en k fallos, vía scipy.stats.nbinom."""
        return stats.nbinom.pmf(k, self.r, self.p)

    def __str__(self):
        etiqueta = f"{self.nombre} " if self.nombre else ""
        return f"{etiqueta}NB(r={self.r}, p={self.p})"
