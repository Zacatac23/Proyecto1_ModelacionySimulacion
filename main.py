"""
main.py
-------
Punto de entrada del proyecto. Arma, para cada jugador definido en
config.py, el modelo Binomial Negativa correspondiente, corre los dos
métodos de generación (Bernoulli secuencial y Transformada inversa),
valida cada muestra con la prueba Chi-cuadrado, imprime el resumen,
guarda el CSV de resultados y genera las 5 figuras comparativas.

Ejecutar con:  python main.py
"""

import numpy as np

import config
from modelo import ModeloBinomialNegativa
from generadores import GeneradorBernoulliSecuencial, GeneradorTransformadaInversa
from simulador import ComparadorMetodos
from reportador import GeneradorReporte
from visualizacion import Visualizador


def construir_modelos() -> dict:
    """Crea un ModeloBinomialNegativa por cada jugador en config.PLAYERS."""
    return {
        nombre: ModeloBinomialNegativa(r=config.R_GOALS, p=p, nombre=nombre)
        for nombre, p in config.PLAYERS.items()
    }


def correr_simulaciones(modelos: dict, rng: np.random.Generator) -> dict:
    """Para cada jugador, corre ambos métodos de generación y devuelve un
    ComparadorMetodos con los resultados listos para reportar/graficar."""
    comparadores = {}
    for nombre, modelo in modelos.items():
        comparador = ComparadorMetodos(modelo)
        comparador.agregar_metodo(GeneradorBernoulliSecuencial(modelo, rng), config.N_SIM)
        comparador.agregar_metodo(GeneradorTransformadaInversa(modelo, rng), config.N_SIM)
        comparadores[nombre] = comparador
    return comparadores


def main():
    rng = np.random.default_rng(seed=config.RANDOM_SEED)

    modelos = construir_modelos()
    comparadores = correr_simulaciones(modelos, rng)

    # ---- reporte de texto y CSV ----
    reportador = GeneradorReporte(modelos, comparadores)
    reportador.imprimir_resumen(config.N_SIM)
    reportador.guardar_csv(f"{config.OUTPUT_DIR}/resumen_resultados.csv")

    # ---- gráficas ----
    nombres_metodos = ["Bernoulli secuencial", "Transformada inversa"]
    visualizador = Visualizador(config.OUTPUT_DIR)

    visualizador.graficar_histograma_vs_teorica(
        modelos, comparadores, "Bernoulli secuencial", "fig1_histogramas_metodo1.png"
    )
    visualizador.graficar_histograma_vs_teorica(
        modelos, comparadores, "Transformada inversa", "fig2_histogramas_metodo2.png"
    )
    visualizador.graficar_comparacion_medias(modelos, comparadores, nombres_metodos)
    visualizador.graficar_boxplot(comparadores, "Bernoulli secuencial")
    visualizador.graficar_tiempos_ejecucion(comparadores, nombres_metodos)

    print("Gráficas guardadas en el directorio", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
