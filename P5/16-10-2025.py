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
    ##Aquí se almacenará los resltados de los diagnosticos
    diagnosticos = []

    ## Recorremos cada paciente 
    for i in range(len(pacientes)):
        ## Paciente actual

        ##Guardamos el nombre del paciente actual
        nombre_paciente = pacientes[i][0]
        ## Guardamos la lista de sintomas del paciente actual
        sintomas_pacientes = pacientes[i][1]

        ## Variable de conteo de coincidencias del paciente actual
        coincidencias_totales = 0

        ## Recorremos cada enfermedad
        for j in range(len(enfermedades)):
            ##Enfermedad actual

            ## Se toma el nombre de la enfermedad actual
            nombre_enfermedad = enfermedades[j][0]
            ## Se toma la lista de sintomas de la enfermedad actual
            nombre_sintomas = enfermedades[j][1]

            ## Varible que cuenta cuántos síntomas del paciente coinciden
            ## para la enfermedad actual
            coincidencias = 0
            
            ## Vamos a comparar síntomas uno por uno

            ##Recorremos la lista de los sintomas del paciente
            for k in range(len(sintomas_pacientes)):
                
                ## Recorremos la lista de los sintomas de la enfermedad actual
                for l in range(len(nombre_sintomas)):

                    ## Si coincide el sintoma_paciente actual con el nombre_sintoma actual 
                    ## entonces se suma 1
                    if(sintomas_pacientes[k] == nombre_sintomas[l]):
                        coincidencias += 1 ##coincidencias = coincidencias + 1
            
            ## Estamos dentro de la enfermedad actual, entonces si esta enfermendad tiene más 
            ## coincidenias que la anterior , se guarda
            if (coincidencias > coincidencias_totales):

                ## coincidencias_totales ahora es igual a las concidencias de la enfermedad que 
                ## tiene más coincidencias
                coincidencias_totales = coincidencias

                ## Guardamos el nombre de la enfermedad que tuvo más coincidencias
                enfermedad_diagnosticada = nombre_enfermedad
        
        ## Se guarda dentro de diagnostico una lista con el paciente actual y la enfermedad diganosticada
        diagnosticos = diagnosticos + [[nombre_paciente, enfermedad_diagnosticada]]

    ## Se retorna la lista de pacientes con diagnosticos
    return diagnosticos

diagnostico = sintomas(pacientes, enfermedades)

for m in range(len(diagnostico)):
    print(f"Para el paciente {diagnostico[m][0]} su diagnostico es: {diagnostico[m][1]}")