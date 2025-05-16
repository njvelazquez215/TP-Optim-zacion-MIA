import os
import matplotlib.pyplot as plt
import numpy as np
import csv

# ---------------------------------------
# CONFIGURACIÓN
# ---------------------------------------

carpeta_resultados = "Resultados/Hiperparámetros"


# ---------------------------------------
# FUNCIONES PARA PARSEAR LOS ARCHIVOS
# ---------------------------------------

def leer_resultado(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Buscar el índice donde empieza la tabla
    for i, line in enumerate(lines):
        if "Resultados:" in line:
            tabla_idx = i + 2
            break
    else:
        return None  # No se encontró tabla

    # Leer encabezados y datos
    reader = csv.reader(lines[tabla_idx - 1:])
    headers = next(reader)
    datos = []

    for row in reader:
        if len(row) == 0:
            break
        datos.append([float(x) for x in row])

    return headers, np.array(datos)


# ---------------------------------------
# FUNCIONES DE VISUALIZACIÓN
# ---------------------------------------

def graficar_resultado(headers, datos, nombre_archivo):
    print(f" {nombre_archivo}")
    print(f"Columnas disponibles: {headers}")

    # Buscar índices relevantes
    try:
        x_idx = headers.index("step size")
        y_idx = headers.index("loss")
    except ValueError:
        print("No se encontró 'step size' o 'loss' en columnas.")
        return

    x = datos[:, x_idx]
    y = datos[:, y_idx]

    plt.figure()
    plt.scatter(x, y, c='blue', alpha=0.6)
    plt.xlabel("Step size")
    plt.ylabel("Loss final")
    plt.title(f"{nombre_archivo} - Step size vs Loss")
    plt.grid(True)

    # También se puede graficar batch size vs loss si existe
    if "batch size" in headers:
        bx = datos[:, headers.index("batch size")]
        plt.figure()
        plt.scatter(bx, y, c='green', alpha=0.6)
        plt.xlabel("Batch size")
        plt.ylabel("Loss final")
        plt.title(f"{nombre_archivo} - Batch size vs Loss")
        plt.grid(True)

    plt.show()


# ---------------------------------------
# MAIN
# ---------------------------------------

def main():
    archivos = [f for f in os.listdir(carpeta_resultados) if f.endswith(".txt")]
    if not archivos:
        print("No se encontraron archivos .txt en la carpeta de resultados.")
        return

    for archivo in archivos:
        path = os.path.join(carpeta_resultados, archivo)
        result = leer_resultado(path)
        if result is not None:
            headers, datos = result
            graficar_resultado(headers, datos, archivo)


if __name__ == "__main__":
    main()