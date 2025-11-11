
familia1 = [("José", "Beto"), ("José", "Karen"), ("Karen", "Miguel"), ("Miguel", "Laura")]
familia2 = [("Ana", "Luis"), ("Luis", "Carlos"), ("Carlos", "Ernesto"), ("Ernesto", "Pablo")]
familia3 = [("Roberto", "Ana"), ("Roberto", "Isaac"), ("Ana", "Sofía"), ("Sofía", "Raúl")]
familia4 = [("Luisa", "José"), ("José", "Lucía"), ("Lucía", "Tomás")]
familia5 = [("Héctor", "Camila"), ("Camila", "Santiago"), ("Santiago", "Daniel")]
familia6 = [("Isabel", "Ricardo"), ("Ricardo", "Emilio"), ("Emilio", "Nora")]
familia7 = [("Daniel", "Patricia"), ("Patricia", "Elena"), ("Elena", "Sara"), ("Sara", "Mateo")]
familia8 = [("Adriana", "Javier"), ("Javier", "Manuel"), ("Manuel", "Raúl"), ("Raúl", "Felipe")]
familia9 = [("David", "Carolina"), ("Carolina", "Diana"), ("Diana", "Clara")]
familia10 = [("Francisco", "Beatriz"), ("Beatriz", "Eduardo"), ("Eduardo", "Carmen"), ("Carmen", "Olga")]
familia11 = [("Marcos", "Renata"), ("Renata", "Julio"), ("Julio", "María")]
familia12 = [("Paola", "Hugo"), ("Hugo", "Mónica"), ("Mónica", "Alejandro"), ("Alejandro", "Iván")]
familia13 = [("Valeria", "Lidia"), ("Lidia", "Andrés"), ("Andrés", "Silvia")]
familia14 = [("Carmen", "Marcos"), ("Marcos", "Renata"), ("Renata", "Julio")]
familia15 = [("Ricardo", "Emilio"), ("Emilio", "Nora"), ("Nora", "Samuel")]
familia16 = [("Camila", "Santiago"), ("Santiago", "Tomás"), ("Tomás", "Ivanna")]
familia17 = [("Patricia", "Elena"), ("Elena", "Paola"), ("Paola", "Sara"), ("Sara", "Emma")]
familia18 = [("Francisco", "Lucía"), ("Lucía", "Sofía"), ("Sofía", "Fernanda"), ("Fernanda", "Mario")]
familia19 = [("Miguel", "Laura"), ("Laura", "Andrés"), ("Andrés", "Isabel")]
familia20 = [("Luisa", "José"), ("José", "Karen"), ("Karen", "Miguel"), ("Miguel", "Laura")]

todas_familias = [
    familia1, familia2, familia3, familia4, familia5,
    familia6, familia7, familia8, familia9, familia10,
    familia11, familia12, familia13, familia14, familia15,
    familia16, familia17, familia18, familia19, familia20
]

def obtener_abuelos(padre):
    abuelos = []
    for p1, h1 in padre:
        for p2, h2 in padre:
            if h1 == p2:
                abuelos = abuelos + [[p1, h2]]
    return abuelos

def obtener_hermanos(padre):
    hermanos = []
    for p1, h1 in padre:
        for p2, h2 in padre:
            if p1 == p2 and h1 != h2 and (h2, h1) not in hermanos:
                hermanos = hermanos + [[h1, h2]]
    return hermanos

def obtener_nietos(abuelos):
    nietos = []
    for x, z in abuelos:
        nietos = nietos + [[z, x]]
    return nietos

def obtener_tios(hermanos, padre):
    tios = []
    for x, y in hermanos:
        for a, b in padre:
            if y == a:
                tios = tios + [[x, b]]
    return tios

def mostrar_relaciones(nombre_familia, padre):
    print(f"############### {nombre_familia.upper()} ###############")


    abuelos = obtener_abuelos(padre)
    hermanos = obtener_hermanos(padre)
    nietos = obtener_nietos(abuelos)
    tios = obtener_tios(hermanos, padre)

    print("\n** Relaciones Padre - Hijo **")
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

    print("\n** Tíos **")
    for a, b in tios:
        print(f"{a} es tio de {b}")

contador = 1
for familia in todas_familias:
    mostrar_relaciones(f"Familia {contador}", familia)
    contador += 1
