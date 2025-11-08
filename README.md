# Laboratorio 09

## Archivos

- `temp_change_mean_extended.cu`: inciso `a` del ejercicio 5.

- `temp_change_mean.cu`: ejercicio 3.

- `Environment_Temperature_change_filled.csv`: datos con 4318 filas.

- `Environment_Temperature_change_filled_extended.csv`: datos con 8636 filas.

- `comprative.py`: generación de graficas de tempo por cantidad de streams.

### Ejercicio 5

- inciso b

| Cantidad de Streams | Tiempo Host->Device | Tiempo Kernel | Tiempo Device->Host | Tiempo total |
|:--|:--|:--|:--|:--|
| 1 | `0.165888` | `0.100960` | `0.044256` | `0.311104` |
| 2 | `0.175424` | `0.108480` | `0.051200` | `0.335104` |
| 4 | `0.191424` | `0.104448` | `0.071680` | `0.367552` |
| 8 | `0.302176` | `0.137760` | `0.120256` | `0.560192` |

> *Tiempos en ms*


- inciso d

**d. ¿Puedes observar algún patrón de comportamiento? En cuál de los tres tiempos se ve una mayor influencia de cambio de acuerdo con la cantidad de streams utilizados. Explica por qué las gráficas se comportan de la forma que se muestra en tus dibujos.**

Se muestran que las transferencias de memoria son las que más se benefician del uso de múltiples streams, presentando una disminución del tiempo conforme aumentan los streams. Esto pasa porque los streams permiten sobreponerse entre sí de transferencias de memoria con ejecución de kernels, aprovechando el paralelismo entre el PCIe bus y la GPU. 

El tiempo del kernel también disminuye pero en menor proporción, ya que el cómputo en sí no cambia, solo se distribuye.