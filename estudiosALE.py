from busquedaHiperparam import estudioALE
import matplotlib.pyplot as plt

# SGD Fix
logscaleFlags = [False, True]
#estudioALE('Resultados/Hiperparámetros/SGD-Fix-50-2025-05-18_18-09-14.txt', logscaleFlags=logscaleFlags)

# SGD Exponential
logscaleFlags = [False, True, True]
#estudioALE('Resultados\Hiperparámetros\SGD-Exponential-50-2025-05-18_19-48-47.txt', logscaleFlags=logscaleFlags)
logscaleFlags = [False, True, True]
#estudioALE(['Resultados\Hiperparámetros\SGD-Exponential-50-2025-05-18_19-48-47.txt', 'Resultados\Hiperparámetros\SGD-Exponential-50-2025-05-18_21-12-43.txt'], logscaleFlags=logscaleFlags)
#estudioALE(['Resultados\Hiperparámetros\SGD-Exponential-50-2025-05-18_19-48-47.txt', 'Resultados\Hiperparámetros\SGD-Exponential-50-2025-05-18_21-12-43.txt', 'Resultados\Hiperparámetros\SGD-Exponential-50-2025-05-19_00-53-43.txt'], logscaleFlags=logscaleFlags)

# SGD Power
logscaleFlags = [False, True, True, False]
#estudioALE('Resultados\Hiperparámetros\SGD-Power-50-2025-05-19_11-44-24.txt', logscaleFlags=logscaleFlags)
#estudioALE('Resultados\Hiperparámetros\SGD-Power-50-2025-05-19_15-20-46.txt', logscaleFlags=logscaleFlags)

# SGD Performance
logscaleFlags = [False, True, False, True]
estudioALE('Resultados\Hiperparámetros\SGD-Performance-50-2025-06-02_18-25-59.txt', logscaleFlags=logscaleFlags)

# ADAM Fix
logscaleFlags = [False, True, False, False]
#estudioALE('Resultados\Hiperparámetros\ADAM-Fix-50-2025-05-20_10-57-28.txt', logscaleFlags=logscaleFlags)

# ADAM Exponential
logscaleFlags = [False, True, False, False, True]
#estudioALE('Resultados\Hiperparámetros\ADAM-Exponential-50-2025-05-20_11-36-41.txt', logscaleFlags=logscaleFlags)

# ADAM Power
logscaleFlags = [False, True, True, False]
#estudioALE(['Resultados\Hiperparámetros\ADAM-Power-50-2025-05-19_18-23-53.txt', 'Resultados\Hiperparámetros\ADAM-Power-50-2025-05-19_21-59-15.txt'], logscaleFlags)
#estudioALE('Resultados\Hiperparámetros\ADAM-Power-50-2025-05-19_22-58-05.txt', logscaleFlags)
#estudioALE('Resultados\Hiperparámetros\ADAM-Power-500-2025-05-20_06-00-10.txt', logscaleFlags)

# ADAM Performance
logscaleFlags = [False, True, True, False]
#estudioALE('Resultados\Hiperparámetros\ADAM-Performance-50-2025-06-02_14-24-30.txt', logscaleFlags)
#estudioALE(['Resultados\Hiperparámetros\ADAM-Performance-50-2025-06-02_16-20-23.txt', 'Resultados\Hiperparámetros\ADAM-Performance-50-2025-06-02_17-07-31.txt'], logscaleFlags)

plt.show()