"""
config.py
---------
Parámetros globales del proyecto: jugadores, meta de goles y tamaño
de la simulación. Se centralizan aquí para poder modificarlos sin
tocar el resto del código (por ejemplo, durante la demostración en
la presentación oral).
"""

# Delanteros a comparar: probabilidad de gol por tiro (p)
PLAYERS = {
    "Delantero A": 0.25,
    "Delantero B": 0.18,
    "Delantero C": 0.12,
}

R_GOALS = 5          # goles objetivo (r)
N_SIM = 10_000       # simulaciones por jugador y por método
RANDOM_SEED = 2026   # semilla para reproducibilidad

OUTPUT_DIR = "output"
