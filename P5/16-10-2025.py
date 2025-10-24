pacientes = [
    ["Juan", ["tos", "fiebre", "dificultad para respirar"]],
    ["Ana", ["diarrea", "dolor abdominal", "vómitos"]],
    ["Carlos", ["sed excesiva", "orinar frecuentemente", "visión borrosa"]],
    ["Luis", ["tos", "fiebre", "escalofríos"]],
    ["María", ["calambres abdominales", "náuseas", "pérdida de apetito"]],
    ["Sofía", ["fatiga", "pérdida de peso", "heridas que no cicatrizan"]],
    ["Pedro", ["dolor de pecho", "dificultad para respirar", "escalofríos"]],
    ["Laura", ["diarrea", "calambres abdominales", "fatiga"]],
    ["Diego", ["sed excesiva", "visión borrosa", "heridas que no cicatrizan"]],
    ["Carmen", ["fiebre", "dolor de cabeza", "tos"]]
]

# BASE DE CONOCIMIENTO
enfermedades = [
    ["infección respiratoria", ["tos", "fiebre", "dificultad para respirar", "escalofríos"]],
    ["gastroenteritis", ["diarrea", "dolor abdominal", "náuseas", "vómitos"]],
    ["diabetes", ["sed excesiva", "orinar frecuentemente", "visión borrosa", "fatiga"]]
]


def sintomas (pacientes, enfermedades):
    diagnosticos = []

    for i in range(len(pacientes)):
        nombre_paciente = pacientes[i][0]
        sintomas_pacientes = pacientes[i][1]

        coincidencias_totales = 0

        for j in range(len(enfermedades)):
            nombre_enfermedad = enfermedades[j][0]
            nombre_sintomas = enfermedades[j][1]

            coincidencias = 0
            for k in range(len(sintomas_pacientes)):
                for l in range(len(nombre_sintomas)):
                    if(sintomas_pacientes[k] == nombre_sintomas[l]):
                        coincidencias += 1 ##coincidencias = coincidencias + 1
            if (coincidencias > coincidencias_totales):
                coincidencias_totales = coincidencias
                enfermedad_diagnosticada = nombre_enfermedad
        diagnosticos = diagnosticos + [[nombre_paciente, enfermedad_diagnosticada]]

    return diagnosticos

diagnostico = sintomas(pacientes, enfermedades)

for m in range(len(diagnostico)):
    print(f"Para el paciente {diagnostico[m][0]} su diagnostico es: {diagnostico[m][1]}")