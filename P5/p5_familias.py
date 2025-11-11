##BASE DE CONOCIMIENTO 
## Cada tupla ("padre/madre", "hijo/a")
padre = [
    ## Familia 1
    ("Arturo", "Beto"),
    ("Arturo", "Carla"),
    ("Carla", "David"),
    ("David", "Elena"),

    ## Familia 2
    ("Fernando", "Gabriela"),
    ("Gabriela", "Hugo"),
    ("Hugo", "Isaac"),
    ("Isaac", "Julia"),

    ## Familia 3
    ("Roberto", "Katia"),
    ("Katia", "Luis"),
    ("Luis", "Manuel"),

    ## Familia 4
    ("Luisa", "Mario"),
    ("Mario", "Nora"),
    ("Nora", "Oscar"),

    ## Familia 5
    ("Héctor", "Camila"),
    ("Camila", "Santiago"),
    ("Santiago", "Daniel"),

    ## Familia 6
    ("Isabel", "Ricardo"),
    ("Ricardo", "Emilio"),
    ("Emilio", "Paola"),

    ## Familia 7
    ("Dante", "Patricia"),
    ("Patricia", "Esteban"),
    ("Esteban", "Sara"),
    ("Sara", "Mateo"),

    ## Familia 8
    ("Adriana", "Javier"),
    ("Javier", "Marcos"),
    ("Marcos", "Raúl"),
    ("Raúl", "Felipe"),

    ## Familia 9
    ("David", "Carolina"),
    ("Carolina", "Diana"),
    ("Diana", "Clara"),

    ## Familia 10
    ("Francisco", "Beatriz"),
    ("Beatriz", "Eduardo"),
    ("Eduardo", "Carmen"),
    ("Carmen", "Olga"),

    ## Familia 11
    ("Miguel", "Renata"),
    ("Renata", "Julio"),
    ("Julio", "María"),

    ## Familia 12
    ("Pedro", "Hugo"),
    ("Hugo", "Mónica"),
    ("Mónica", "Alejandro"),
    ("Alejandro", "Iván"),

    ## Familia 13
    ("Valeria", "Lidia"),
    ("Lidia", "Andrés"),
    ("Andrés", "Silvia"),

    ## Familia 14
    ("Carlos", "Martín"),
    ("Martín", "Rebeca"),
    ("Rebeca", "Julián"),

    ## Familia 15
    ("Ramón", "Ema"),
    ("Ema", "Natalia"),
    ("Natalia", "Samuel"),

    ## Familia 16
    ("César", "Simón"),
    ("Simón", "Tomás"),
    ("Tomás", "Ivanna"),

    ## Familia 17
    ("Pablo", "Elisa"),
    ("Elisa", "Pilar"),
    ("Pilar", "Saúl"),
    ("Saúl", "Emma"),

    ## Familia 18
    ("Fabián", "Lucía"),
    ("Lucía", "Sofía"),
    ("Sofía", "Fernanda"),
    ("Fernanda", "Mateo"),

    ## Familia 19
    ("Miguel", "Laura"),
    ("Laura", "Antonio"),
    ("Antonio", "Iris"),

    ## Familia 20
    ("León", "Jorge"),
    ("Jorge", "Karen"),
    ("Karen", "Moisés"),
    ("Moisés", "Lara")
]
## Inferencias lógicas

##  Abuelos
abuelos = []
for p1, h1 in padre:
    for p2, h2 in padre:
        if h1 == p2:
            abuelos = abuelos + [[p1,h2]]

##  Hermanos
hermanos = []
for p1, h1 in padre:
    for p2, h2 in padre:
        if p1 == p2 and h1 != h2 and (h2, h1) not in hermanos:
            hermanos =  hermanos + [[h1,h2]]

##  Nietos
nietos = []
for x, z in abuelos:
    nietos = nietos + [[z,x]]

##  Tíos
tios = []
for x, y in hermanos:
    for a, b in padre:
        if y == a:
            tios = tios + [[x,b]]

print("** Relaciones Padre - Hijo **")
for p, h in padre:
    print(f"{p} es padre de {h}")

print("\n** Abuelos **")
for a, b in abuelos:
    print(f"{a} es abuelo/a de {b}")

print("\n** Hermanos **")
for a, b in hermanos:
    print(f"{a} es hermano/a de {b}")

print("\n** Nietos **")
for a, b in nietos:
    print(f"{a} es nieto de {b}")

print("\n** Tios **")
for a, b in tios:
    print(f"{a} es tio de {b}")
