"""
reportador.py
-------------
Clase GeneradorReporte: a partir de los ComparadorMetodos de cada
jugador, imprime un resumen en consola y guarda una tabla resumen en
CSV con las métricas usadas en el informe (medias, varianzas, tiempo
de ejecución, resultados de la prueba Chi-cuadrado, etc.).
"""

import csv
import os
from typing import Dict, List

from modelo import ModeloBinomialNegativa
from simulador import ComparadorMetodos


class GeneradorReporte:
    def __init__(self, modelos: Dict[str, ModeloBinomialNegativa], comparadores: Dict[str, ComparadorMetodos]):
        self.modelos = modelos
        self.comparadores = comparadores

    def imprimir_resumen(self, n_sim: int):
        primero = next(iter(self.modelos.values()))
        print("=" * 78)
        print(f"Simulación: tiros necesarios para llegar a r = {primero.r} goles")
        print(f"N = {n_sim:,} simulaciones por jugador y por método")
        print("=" * 78)

        for nombre, modelo in self.modelos.items():
            comparador = self.comparadores[nombre]
            print(f"\n{nombre}  (p = {modelo.p})")
            print(f"  Media teórica de tiros           : {modelo.media_tiros:.3f}")
            print(f"  Varianza teórica (fallos)        : {modelo.varianza_fallos:.3f}")
            for nombre_metodo, resultado in comparador.resultados.items():
                print(f"  -- {nombre_metodo} --")
                print(f"     Media de tiros simulada       : {resultado.media_tiros:.3f}")
                print(f"     Varianza (fallos) simulada    : {resultado.varianza_fallos:.3f}")
                print(f"     Tiempo de ejecución (s)       : {resultado.tiempo_sec:.5f}")
                print(f"     {resultado.chi2}")

    def filas_resumen(self) -> List[dict]:
        filas = []
        for nombre, modelo in self.modelos.items():
            comparador = self.comparadores[nombre]
            for nombre_metodo, resultado in comparador.resultados.items():
                filas.append({
                    "jugador": nombre,
                    "p": modelo.p,
                    "metodo": nombre_metodo,
                    "media_tiros_simulada": round(resultado.media_tiros, 4),
                    "media_tiros_teorica": round(modelo.media_tiros, 4),
                    "varianza_fallos_simulada": round(resultado.varianza_fallos, 4),
                    "varianza_fallos_teorica": round(modelo.varianza_fallos, 4),
                    "tiempo_ejecucion_s": round(resultado.tiempo_sec, 6),
                    "chi2": round(resultado.chi2.estadistico, 4),
                    "gl": resultado.chi2.grados_libertad,
                    "p_valor": round(resultado.chi2.p_valor, 4),
                    "prob_mas_tiros_que_la_media": round(resultado.prob_mas_tiros_que(modelo.media_tiros), 4),
                })
        return filas

    def guardar_csv(self, ruta: str):
        filas = self.filas_resumen()
        os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=filas[0].keys())
            writer.writeheader()
            writer.writerows(filas)
        print(f"\nResumen guardado en {ruta}")
