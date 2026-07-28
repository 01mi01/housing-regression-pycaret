# Predicción de Precios de Bienes Raíces en Australia - Regresión Avanzada

Modelo de regresión regularizada (Ridge, Lasso y ElasticNet) construido con PyCaret para predecir el precio de venta de viviendas e identificar las variables que mejor lo explican.

## Índice

- [Introducción](#introducción)
  - [Integrantes del Grupo](#integrantes-del-grupo)
  - [Métodos Utilizados](#métodos-utilizados)
  - [Tecnologías](#tecnologías)
- [Descarga y Configuración](#descarga-y-configuración)
  - [Requisitos Previos](#requisitos-previos)
  - [Cómo Ejecutar](#cómo-ejecutar)
  - [Estructura del Repositorio](#estructura-del-repositorio)
- [Declaración del Problema](#declaración-del-problema)
  - [Objetivo Comercial](#objetivo-comercial)
  - [Descripción del Conjunto de Datos](#descripción-del-conjunto-de-datos)
  - [Preparación de Datos](#preparación-de-datos)
  - [Hallazgos del Análisis Exploratorio](#hallazgos-del-análisis-exploratorio)
  - [Construcción y Evaluación del Modelo](#construcción-y-evaluación-del-modelo)
- [Conclusiones](#conclusiones)
  - [Regresión Ridge](#regresión-ridge)
  - [Regresión Lasso](#regresión-lasso)
  - [Regresión ElasticNet](#regresión-elasticnet)
  - [Las Variables Más Significativas](#las-variables-más-significativas)

## Introducción

Este repositorio contiene el análisis y el modelado desarrollados para el subproyecto del Módulo 14. El objetivo es construir un modelo de regresión con regularización que prediga el precio de venta (`SalePrice`) de una vivienda a partir de sus características, y determinar qué variables resultan significativas para explicar ese precio.

Todo el desarrollo se encuentra en un único notebook: `MFSDv1p2_Regresion_Avanzada_con_PyCaret.ipynb`.

### Integrantes del Grupo

* De los Rios Aliaga, Mijaelha
* Flores Velasquez, Maritza Karen
* Machaca Lamas, Sergio Alejandro
* Mamani Poma, Alexander Manuel
* Vilela, Bruno

### Métodos Utilizados

* Análisis exploratorio de datos (EDA)
* Limpieza de datos e imputación de valores faltantes
* Análisis univariable, bivariable y multivariable
* Selección de características mediante RFE y Factor de Inflación de Varianza (VIF)
* Regresión lineal y regresión regularizada: Ridge, Lasso y ElasticNet
* Validación cruzada y búsqueda de hiperparámetros (`KFold`, `GridSearchCV`)
* Evaluación mediante R², RMSE y análisis de residuos

### Tecnologías

* Python
* Pandas / NumPy
* Matplotlib / Seaborn
* scikit-learn
* statsmodels
* PyCaret
* Jupyter Notebook / Google Colab

## Descarga y Configuración

### Requisitos Previos

El proyecto puede ejecutarse de dos formas: en **Google Colab** (recomendado) o en un entorno **local** con Jupyter.

En ambos casos, PyCaret se instala desde el repositorio oficial en GitHub y no desde PyPI, porque la versión publicada en PyPI no es compatible con la versión de Python de Colab:

```python
!pip install git+https://github.com/pycaret/pycaret.git@v4.0.0a0 -q
```

Para el entorno local sirve cualquier instalación de Jupyter; por ejemplo mediante Anaconda (https://docs.anaconda.com/anaconda/install/index.html). Si PyCaret ya está instalado en el entorno, puede omitirse esa celda.

No se requiere GPU ni TPU: los modelos utilizados son de scikit-learn y se entrenan sobre un conjunto de datos pequeño, por lo que un entorno de **CPU** es suficiente y más rápido.

### Cómo Ejecutar

**Opción A — Google Colab (recomendada)**

1. Abrir el notebook en Colab, ya sea desde `Archivo > Abrir cuaderno > GitHub` o subiendo directamente el archivo `.ipynb`.
2. Seleccionar un entorno de ejecución de tipo **CPU**.
3. Ejecutar las celdas de arriba hacia abajo. La celda de carga de datos detecta automáticamente que se está ejecutando en Colab y solicitará que se suba el archivo `dataset.csv`, que se guarda por sí sola en la carpeta `_data/`.

**Opción B — Entorno local**

1. Abrir una terminal y clonar el repositorio:

```bash
git clone https://github.com/01mi01/housing-regression-pycaret.git
cd housing-regression-pycaret
```

2. Abrir el notebook:

```bash
jupyter lab MFSDv1p2_Regresion_Avanzada_con_PyCaret.ipynb
```

> **Importante:** el notebook debe ejecutarse con la raíz del repositorio como directorio de trabajo, ya que la ruta del conjunto de datos es relativa (`DATA_FILE_PATH = '_data/dataset.csv'`). Si el directorio de trabajo es otro, la celda de carga se detiene con un mensaje de error indicándolo.

### Estructura del Repositorio

```
.
├── MFSDv1p2_Regresion_Avanzada_con_PyCaret.ipynb   # Notebook con todo el desarrollo
├── _data/
│   └── dataset.csv                                 # Conjunto de datos de entrada
└── README.md
```

## Declaración del Problema

Una empresa de vivienda con sede en EE. UU. llamada Surprise Housing ha decidido ingresar al mercado australiano. La empresa utiliza el análisis de datos para comprar casas a un precio inferior a su valor real y venderlas a un precio más alto. Con ese propósito ha recopilado un conjunto de datos sobre la venta de casas, disponible en el archivo CSV de este repositorio.

La compañía está buscando posibles propiedades para comprar e ingresar al mercado, y necesita un modelo de regresión con regularización que prediga el valor real de esas propiedades para decidir si invertir o no en ellas.

La empresa quiere saber:

* Qué variables son significativas para predecir el precio de una casa, y
* Qué tan bien esas variables describen el precio de una casa.

Además, se debe determinar el valor óptimo de lambda para la regresión de Ridge y de Lasso.

### Objetivo Comercial

Modelar el precio de las casas con las variables independientes disponibles. La gerencia utilizará el modelo para comprender cómo varían exactamente los precios en función de dichas variables y, en consecuencia, ajustar la estrategia de la empresa y concentrarse en las áreas que generarán altos rendimientos. El modelo será, además, una buena manera de que la gerencia entienda la dinámica de precios de un nuevo mercado.

### Descripción del Conjunto de Datos

* **Registros:** 1460 filas × 81 columnas.
* **Variable objetivo:** `SalePrice`, el precio de venta de la vivienda.
* **Tipos de variables:** 38 numéricas y 43 categóricas.
* **Valores faltantes:** en varias columnas categóricas el valor `NA` del archivo original no representa un dato ausente, sino que la vivienda **no posee esa característica** (por ejemplo `FireplaceQu`, `GarageType` o `BsmtQual`). Estos casos se rellenan con la categoría `'None'` en lugar de imputarse o eliminarse, para no destruir información real.
* **Tras la limpieza:** 1459 filas × 76 columnas, sin valores nulos ni registros duplicados.

---

### Preparación de Datos

1. Limpieza de Datos y Análisis de Datos Faltantes.
2. Análisis y Tratamiento de Valores Atípicos.
3. Derivación de Columnas Categóricas.
4. Análisis Univariable.
5. Análisis Bivariable.
6. Análisis Multivariable.

### Hallazgos del Análisis Exploratorio

Resultados obtenidos sobre el conjunto de datos ya limpio (1459 filas × 76 columnas: 37 variables numéricas y 39 categóricas).

**Variable objetivo**

`SalePrice` presenta una asimetría positiva marcada (**skew = 1.88**, curtosis = 6.53): la media (180 930) supera a la mediana (163 000) y el valor máximo alcanza 755 000. Al aplicar la transformación logarítmica la asimetría cae a **0.12**, normalizando la distribución, lo que resulta conveniente dado que la regresión lineal asume residuos normales.

**Principales predictores numéricos** (correlación de Pearson con `SalePrice`)

| Variable | r |
|---|---|
| `OverallQual` | 0.79 |
| `GrLivArea` | 0.71 |
| `GarageCars` | 0.64 |
| `GarageArea` | 0.62 |
| `TotalBsmtSF` | 0.61 |
| `1stFlrSF` | 0.61 |

Las correlaciones negativas son todas débiles; la mayor en valor absoluto es `KitchenAbvGr` (-0.14).

**Principales predictores categóricos** (eta cuadrado, proporción de varianza de `SalePrice` explicada por cada variable)

| Variable | η² | Categorías |
|---|---|---|
| `Neighborhood` | 0.546 | 25 |
| `ExterQual` | 0.477 | 4 |
| `BsmtQual` | 0.465 | 5 |
| `KitchenQual` | 0.457 | 4 |

La ubicación y la calidad de los acabados son los factores categóricos determinantes del precio.

**Multicolinealidad**

Tres pares de variables numéricas superan una correlación de 0.7 y miden esencialmente lo mismo, lo que respalda el uso de modelos regularizados:

* `GarageCars` ~ `GarageArea` (0.883)
* `GrLivArea` ~ `TotRmsAbvGrd` (0.826)
* `TotalBsmtSF` ~ `1stFlrSF` (0.819)

**Variables con poca información**

Cinco variables categóricas están dominadas por una sola categoría y no pueden explicar diferencias de precio: `Utilities` (99.9% `AllPub`), `Street` (99.6% `Pave`), `Condition2` (99.0% `Norm`), `RoofMatl` (98.2% `CompShg`) y `Heating` (97.8% `GasA`).

**Asimetría y valores atípicos**

Varias variables numéricas están muy sesgadas por corresponder a características que la mayoría de las viviendas no posee: `MiscVal` (24.47), `PoolArea` (14.82) y `LotArea` (12.20). Además, se identificaron **dos ventas atípicas** de viviendas con más de 4000 pies cuadrados y calidad máxima (`OverallQual` = 10) vendidas por debajo de 185 000, que podrían afectar de forma desproporcionada a los coeficientes de la regresión.

### Construcción y Evaluación del Modelo

Tras el análisis exploratorio, el dataset queda preparado para modelar: se descartan variables categóricas casi constantes y dos ventas atípicas, el objetivo se transforma con `log1p(SalePrice)`, las categóricas se codifican con one-hot y los datos se separan en entrenamiento/prueba (80/20). Están disponibles `X_train`, `X_test`, `y_train` y `y_test`.

El modelado (sección 7 del notebook) sigue estos pasos, ya implementados:

1. **Escalado** de características con `StandardScaler` (ajustado solo con *train* para evitar *data leakage*).
2. **Selección de características** mediante **VIF** (multicolinealidad) y **RFE**.
3. **Comparación de modelos con PyCaret** (`setup` + `compare_models`) para justificar la familia elegida.
4. **Ridge, Lasso y ElasticNet** con búsqueda del **λ óptimo** por validación cruzada (`KFold` de 10 pliegues).
5. **Análisis de residuos** (normalidad, homocedasticidad, Durbin-Watson).
6. **Evaluación** en el *hold-out* con R² y RMSE, en escala log y en precio real (`expm1`).
7. **Predicción** de ejemplo y comparación real vs. predicho.
8. **Conclusión y análisis final.**

> **Enfoque de evaluación:** PyCaret se usa para comparar rápidamente varias familias de modelos. Las
> métricas reportadas abajo provienen de la **evaluación reproducible sobre el *hold-out* del 20 %**
> reservado antes del entrenamiento (semilla fija `random_state=42`), ajustando el λ con `GridSearchCV`.

## Conclusiones

**Modelo recomendado: Regresión Lasso.** Logra el mejor R² en prueba con muchas menos variables que
Ridge, lo que facilita explicar al negocio qué mueve el precio. ElasticNet (`l1_ratio ≈ 0.9`) converge
prácticamente a la misma solución que Lasso. Las métricas provienen de la evaluación sobre el *hold-out*
del 20 % (292 viviendas), no vistas durante el entrenamiento ni el tuning del λ.

### Regresión Ridge

* **Valor óptimo de lambda:** ≈ 172.7
* **R² en entrenamiento:** 0.94
* **R² en prueba:** 0.91
* **RMSE en prueba:** $21 420

### Regresión Lasso

* **Valor óptimo de lambda:** ≈ 0.0009
* **R² en entrenamiento:** 0.94
* **R² en prueba:** 0.92
* **RMSE en prueba:** $19 202

### Regresión ElasticNet

* **Valor óptimo de lambda:** ≈ 0.0009 (`l1_ratio = 0.9`)
* **R² en entrenamiento:** 0.94
* **R² en prueba:** 0.92
* **RMSE en prueba:** $19 199

### Precisión sobre el hold-out completo

Sobre las 292 viviendas del conjunto de prueba (modelo Lasso, precio real vs. predicho):

* **MAPE (error porcentual absoluto medio):** 8.5 %
* **MAE (error absoluto medio):** $13 962
* **Error porcentual mediano:** 5.9 %

### Las Variables Más Significativas

Lasso conserva 98 de las 225 variables (tras one-hot encoding). Las de mayor peso, en orden de
importancia, son: **tamaño habitable** (`GrLivArea`), **calidad general** (`OverallQual`),
**antigüedad y remodelación** (`YearBuilt`, `YearRemodAdd`), **sótano** (`TotalBsmtSF`, `BsmtFinSF1`),
**ubicación** (`Neighborhood`, `MSZoning`) y **tipo de venta** (`SaleType_New`, `SaleCondition_Normal`).

El contraste con RFE (que preselecciona un top-30 de forma independiente) coincide en 23 variables con
las que conserva Lasso, justo en ese mismo núcleo, lo que refuerza la confianza en la selección.

### Recomendación de negocio y límites del modelo

Dado que el MAPE del modelo es ~8.5 %, un criterio de compra razonable es priorizar viviendas cuyo
precio real esté **más de un 8-10 % por debajo del precio predicho**: un margen menor puede deberse al
ruido propio del modelo y no a una oportunidad real. Los residuos son aproximadamente normales y sin
patrón (Durbin-Watson ≈ 2), por lo que las predicciones son fiables **dentro del rango de datos
entrenado** (`GrLivArea ≤ 4000`, sin los dos outliers grandes-y-baratos removidos en la preparación).
Aplicar el modelo a propiedades muy fuera de ese rango es una extrapolación y su error esperado será
mayor al reportado aquí.
