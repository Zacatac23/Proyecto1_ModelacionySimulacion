"""
visualizacion.py
-----------------
Clase Visualizador: a partir de los ComparadorMetodos de cada jugador,
genera y guarda en OUTPUT_DIR las gráficas comparativas del informe:

    graficar_ajuste()      histograma de cada muestra simulada vs.
                           pmf teórica de la Binomial Negativa.
    graficar_tiempos()     barras comparando el tiempo de ejecución
                           de cada método, por jugador.
    graficar_tiros()       boxplot de los tiros totales simulados,
                           por jugador y método.
"""

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modelo import ModeloBinomialNegativa
from simulador import ComparadorMetodos


class Visualizador:
    def __init__(self, modelos: Dict[str, ModeloBinomialNegativa], comparadores: Dict[str, ComparadorMetodos],
                 output_dir: str):
        self.modelos = modelos
        self.comparadores = comparadores
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def graficar_ajuste(self, nombre_archivo: str = "ajuste_pmf.png"):
        """Histograma de fallos simulados vs. pmf teórica, un panel por jugador/método."""
        jugadores = list(self.modelos.keys())
        metodos = list(next(iter(self.comparadores.values())).resultados.keys())

        fig, ejes = plt.subplots(len(jugadores), len(metodos),
                                  figsize=(6 * len(metodos), 4 * len(jugadores)), squeeze=False)

        for i, nombre in enumerate(jugadores):
            modelo = self.modelos[nombre]
            comparador = self.comparadores[nombre]
            for j, metodo in enumerate(metodos):
                ax = ejes[i][j]
                resultado = comparador.resultados[metodo]
                fallos = resultado.muestra_fallos

                max_k = int(np.percentile(fallos, 99.5)) + 1
                bins = np.arange(0, max_k + 1)
                ax.hist(fallos, bins=bins, density=True, alpha=0.6,
                        color="tab:blue", edgecolor="white", label="Simulado")

                k_vals = np.arange(0, max_k)
                ax.plot(k_vals + 0.5, modelo.pmf(k_vals), "o-", color="tab:red",
                        markersize=3, label="Teórico (pmf)")

                ax.set_title(f"{nombre} — {metodo}", fontsize=10)
                ax.set_xlabel("Fallos antes del r-ésimo gol")
                ax.set_ylabel("Densidad")
                ax.legend(fontsize=8)

        fig.tight_layout()
        ruta = os.path.join(self.output_dir, nombre_archivo)
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
        print(f"Gráfica de ajuste guardada en {ruta}")

    def graficar_tiempos(self, nombre_archivo: str = "tiempos_ejecucion.png"):
        """Barras comparando el tiempo de ejecución de cada método, por jugador."""
        jugadores = list(self.modelos.keys())
        metodos = list(next(iter(self.comparadores.values())).resultados.keys())

        x = np.arange(len(jugadores))
        ancho = 0.8 / len(metodos)

        fig, ax = plt.subplots(figsize=(8, 5))
        for j, metodo in enumerate(metodos):
            tiempos = [self.comparadores[nombre].resultados[metodo].tiempo_sec for nombre in jugadores]
            ax.bar(x + j * ancho, tiempos, width=ancho, label=metodo)

        ax.set_xticks(x + ancho * (len(metodos) - 1) / 2)
        ax.set_xticklabels(jugadores)
        ax.set_ylabel("Tiempo de ejecución (s)")
        ax.set_title("Tiempo de ejecución por método")
        ax.legend()

        fig.tight_layout()
        ruta = os.path.join(self.output_dir, nombre_archivo)
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
        print(f"Gráfica de tiempos guardada en {ruta}")

    def graficar_tiros(self, nombre_archivo: str = "boxplot_tiros.png"):
        """Boxplot de los tiros totales simulados, agrupado por jugador y método."""
        jugadores = list(self.modelos.keys())
        metodos = list(next(iter(self.comparadores.values())).resultados.keys())

        datos, etiquetas = [], []
        for nombre in jugadores:
            for metodo in metodos:
                datos.append(self.comparadores[nombre].resultados[metodo].muestra_tiros)
                etiquetas.append(f"{nombre}\n{metodo}")

        fig, ax = plt.subplots(figsize=(2 * len(datos), 5))
        ax.boxplot(datos, labels=etiquetas, showmeans=True)
        ax.set_ylabel("Tiros totales")
        ax.set_title("Distribución de tiros totales por jugador y método")

        fig.tight_layout()
        ruta = os.path.join(self.output_dir, nombre_archivo)
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
        print(f"Gráfica de tiros guardada en {ruta}")

    def graficar_todo(self):
        self.graficar_ajuste()
        self.graficar_tiempos()
        self.graficar_tiros()
