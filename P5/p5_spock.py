import random

## Base de conocimientos (signal, signals que vence)
reglas_juego = [
    ["Piedra", ["Tijeras", "Lagarto"]],
    ["Papel", ["Piedra", "Spock"]],
    ["Tijeras", ["Papel", "Lagarto"]],
    ["Lagarto", ["Papel", "Spock"]],
    ["Spock", ["Piedra", "Tijeras"]]
]

signals = ["Piedra", "Papel", "Tijeras", "Lagarto","Spock"]

def jugada (jugadorA, jugadorB, reglas_juego, signals):
    
    signalA = signals[jugadorA]
    signalB = signals[jugadorB]


    if signalA == signalB :
        return "Empate"
    
    vencedoresA = reglas_juego[jugadorA][1]

    if (signalB in vencedoresA):
        return "Ganaste"
    else:
        return "Gana la PC"


while True:
    print("1-.Piedra, 2-.Papel, 3-.Tijeras, 4-.Lagarto o 5-.Spock\n")
    jugadorA = int(input("Selecciona una signal: ")) -1
    jugadorB = random.randint(0, 4)
    print(f"La PC eligio: {signals[jugadorB]}")

    ganador = jugada(jugadorA, jugadorB, reglas_juego,signals)
    print(f"EL resultado de la jugada es: {ganador}")
