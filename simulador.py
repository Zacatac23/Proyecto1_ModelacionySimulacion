"""
simulador.py
------------
Orquesta la comparación de los métodos de generación para un jugador:
por cada método (Bernoulli secuencial, Transformada inversa) genera
una muestra, mide su tiempo de ejecución y valida el ajuste contra el
modelo teórico mediante la prueba Chi-cuadrado de validacion.py.
"""

import time
from dataclasses import dataclass
from typing import Dict

import numpy as np

from modelo import ModeloBinomialNegativa
from generadores import GeneradorVariableAleatoria
from validacion import PruebaBondadAjuste, ResultadoChiCuadrado


@dataclass
class ResultadoSimulacion:
    """Resultado de simular un método de generación para un jugador."""

    nombre_metodo: str
    muestra_fallos: np.ndarray
    muestra_tiros: np.ndarray
    tiempo_sec: float
    chi2: ResultadoChiCuadrado

    @property
    def media_fallos(self) -> float:
        return float(self.muestra_fallos.mean())

    @property
    def varianza_fallos(self) -> float:
        return float(self.muestra_fallos.var(ddof=1))

    @property
    def media_tiros(self) -> float:
        return float(self.muestra_tiros.mean())

    @property
    def varianza_tiros(self) -> float:
        return float(self.muestra_tiros.var(ddof=1))

    def prob_mas_tiros_que(self, umbral: float) -> float:
        """Proporción de la muestra que necesitó más de `umbral` tiros."""
        return float((self.muestra_tiros > umbral).mean())


class ComparadorMetodos:
    """
    Ejecuta y compara, para un mismo modelo teórico, todos los
    generadores de variables aleatorias que se le entreguen.
    """

    def __init__(self, modelo: ModeloBinomialNegativa, generadores: Dict[str, GeneradorVariableAleatoria]):
        self.modelo = modelo
        self.generadores = generadores
        self.resultados: Dict[str, ResultadoSimulacion] = {}

    def ejecutar(self, n_sim: int) -> Dict[str, ResultadoSimulacion]:
        prueba = PruebaBondadAjuste(self.modelo)
        for nombre_metodo, generador in self.generadores.items():
            inicio = time.perf_counter()
            fallos = generador.generar(n_sim)
            tiempo_sec = time.perf_counter() - inicio

            chi2 = prueba.evaluar(fallos)
            tiros = fallos + self.modelo.r

            self.resultados[nombre_metodo] = ResultadoSimulacion(
                nombre_metodo=nombre_metodo,
                muestra_fallos=fallos,
                muestra_tiros=tiros,
                tiempo_sec=tiempo_sec,
                chi2=chi2,
            )
        return self.resultados
