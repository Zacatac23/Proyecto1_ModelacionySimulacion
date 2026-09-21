"""
validacion.py
-------------
Validación estadística de las muestras generadas: prueba de bondad
de ajuste Chi-cuadrado frente al pmf teórico de la Binomial Negativa.
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats

from modelo import ModeloBinomialNegativa


@dataclass
class ResultadoChiCuadrado:
    estadistico: float
    p_valor: float
    grados_libertad: int

    def rechaza_h0(self, alpha: float = 0.05) -> bool:
        """True si se rechaza H0 (la muestra NO proviene de la NB teórica)."""
        return self.p_valor < alpha

    def __str__(self):
        return f"χ²={self.estadistico:.3f}, gl={self.grados_libertad}, p-valor={self.p_valor:.4f}"


class PruebaBondadAjuste:
    """
    Aplica la prueba Chi-cuadrado de bondad de ajuste a una muestra de
    fallos frente al pmf teórico de un ModeloBinomialNegativa,
    agrupando categorías para mantener frecuencia esperada >= 5.
    """

    def __init__(self, modelo: ModeloBinomialNegativa, max_bins: int = 25):
        self.modelo = modelo
        self.max_bins = max_bins

    def evaluar(self, muestra: np.ndarray) -> ResultadoChiCuadrado:
        r, p = self.modelo.r, self.modelo.p
        max_val = int(muestra.max())
        edges = np.arange(0, min(max_val, self.max_bins) + 1)

        observado = np.array([(muestra == k).sum() for k in edges])
        observado = np.append(observado, (muestra > edges[-1]).sum())

        pmf_vals = self.modelo.pmf(edges)
        prob_cola = 1 - pmf_vals.sum()
        esperado = np.append(pmf_vals, prob_cola) * len(muestra)

        obs_agrupado, esp_agrupado = self._agrupar_categorias(observado, esperado)

        k_categorias = len(obs_agrupado)
        gl = max(k_categorias - 1 - 1, 1)  # -1 por p conocido/estimado externamente

        chi2_stat, _ = stats.chisquare(obs_agrupado, esp_agrupado)
        p_valor = 1 - stats.chi2.cdf(chi2_stat, gl)

        return ResultadoChiCuadrado(estadistico=float(chi2_stat), p_valor=float(p_valor), grados_libertad=gl)

    @staticmethod
    def _agrupar_categorias(observado: np.ndarray, esperado: np.ndarray, minimo_esperado: float = 5.0):
        """Agrupa categorías consecutivas hasta que la frecuencia esperada acumulada >= minimo_esperado."""
        obs_g, exp_g = [], []
        o_acc, e_acc = 0.0, 0.0
        for o, e in zip(observado, esperado):
            o_acc += o
            e_acc += e
            if e_acc >= minimo_esperado:
                obs_g.append(o_acc)
                exp_g.append(e_acc)
                o_acc, e_acc = 0.0, 0.0
        if e_acc > 0:
            if exp_g:
                obs_g[-1] += o_acc
                exp_g[-1] += e_acc
            else:
                obs_g.append(o_acc)
                exp_g.append(e_acc)

        obs_g = np.array(obs_g, dtype=float)
        exp_g = np.array(exp_g, dtype=float)
        exp_g *= obs_g.sum() / exp_g.sum()  # reescalar para evitar error numérico en scipy
        return obs_g, exp_g
