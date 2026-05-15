---
title: Sistema Predictivo de Intención de Compra
sdk: docker
app_port: 7860
---
# Sistema Predictivo de Intención de Compra E-commerce

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7)](https://render.com/)
[![Accuracy](https://img.shields.io/badge/Accuracy-89.41%25-brightgreen)]()

Servicio de Machine Learning para predecir la intención de compra de sesiones de navegación en un sitio de e-commerce. Incluye un pipeline completo de entrenamiento offline con Random Forest, búsqueda manual de hiperparámetros, threshold tuning y una API REST con FastAPI para inferencia en tiempo real.

---

## Tabla de Contenidos

- [Contexto del Negocio](#contexto-del-negocio)
- [Arquitectura](#arquitectura)
- [Pipeline de Data Science](#pipeline-de-data-science)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [API Documentation](#api-documentation)
- [Setup Local](#setup-local)
- [Docker](#docker)
- [Despliegue](#despliegue)
- [Decisiones de Arquitectura](#decisiones-de-arquitectura)
- [Estructura del Proyecto](#estructura-del-proyecto)

---

## Contexto del Negocio

Una empresa de comercio electrónico requiere mejorar su inteligencia de negocios detectando de forma temprana si un usuario realizará una compra durante su sesión de navegación. Este sistema permite a la tienda mostrar dinámicamente productos de mayor valor para aumentar el ingreso general (*revenue*), consumiendo predicciones en tiempo real desde el frontend web.

El dataset utilizado contiene **12,330 sesiones de navegación** con **17 atributos predictivos** agrupados en tres categorías de negocio:

| Categoría | Features | Significado comercial |
|-----------|----------|----------------------|
| **Navegación** | `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration` | Cantidad de páginas visitadas por tipo y tiempo invertido. Un usuario que pasa más tiempo en páginas de producto (`ProductRelated_Duration`) tiene mayor probabilidad de compra. |
| **Métricas de Google Analytics** | `BounceRates`, `ExitRates`, `PageValues` | `BounceRates`: porcentaje de sesiones que abandonan sin interactuar. `ExitRates`: porcentaje de sesiones donde la página fue la última visitada. `PageValues`: valor comercial promedio de la página antes de una transacción. |
| **Contexto temporal y demográfico** | `SpecialDay`, `Month`, `Weekend`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType` | `SpecialDay` mide proximidad a fechas festivas (0 a 1). `VisitorType` distingue entre visitantes nuevos y recurrentes, siendo estos últimos más propensos a comprar. |

La variable objetivo es **`Revenue`** (booleana), que indica si la sesión finalizó en una compra. El dataset presenta un desbalance natural: aproximadamente **84.5 % de sesiones sin compra** y **15.5 % con compra**, un factor crítico a considerar en el diseño del modelo.

---

## Arquitectura

> ```
> ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
> │  dataset_shop   │────▶│  src/pipeline.py │────▶│  src/train.py       │
> │   (raw CSV)     │     │  Clean + Split   │     │  ManualSearch       │
> └─────────────────┘     └──────────────────┘     └─────────────────────┘
>                                                        │
>                                                        ▼
>                                           ┌─────────────────────┐
>                                           │ artifacts/ecommerce │
>                                           │   _pipeline.pkl     │
>                                           └─────────────────────┘
>                                                        │
>                                                        ▼
>                                           ┌─────────────────────┐
>                                           │    src/api.py       │
>                                           │   FastAPI + UVicorn │
>                                           │    Puerto 7860      │
>                                           └─────────────────────┘
> ```

El sistema sigue un patrón **offline training / online serving**:
1. El pipeline de datos (`src/pipeline.py`) limpia y divide el dataset.
2. El script de entrenamiento (`src/train.py`) ejecuta HPO, evalúa y serializa el modelo.
3. La API (`src/api.py`) carga el artefacto al iniciar y expone predicciones vía REST.

---

## Pipeline de Data Science

### Dataset

El dataset contiene **17 features de entrada** (10 numéricas y 7 categóricas/booleanas) más la variable objetivo `Revenue`:

| Tipo | Features |
|------|----------|
| Numéricas | `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay` |
| Categóricas / Booleanas | `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend` |
| Objetivo | `Revenue` (boolean) |

### 1. Limpieza de Datos

El dataset no contiene valores nulos ni duplicados. Se verificó la integridad de tipos de datos (numéricos vs categóricos) y se validó que `Revenue` solo contenga valores booleanos. No fue necesario eliminar registros ni corregir inconsistencias.

### 2. Imputación de Valores Faltantes

Aunque el dataset actual no tiene valores faltantes, el pipeline incorpora **`SimpleImputer`** de scikit-learn como medida preventiva para datos futuros:

- **Variables numéricas**: `SimpleImputer(strategy='median')` — reemplaza faltantes con la mediana, robusta ante outliers.
- **Variables categóricas**: `SimpleImputer(strategy='most_frequent')` — reemplaza faltantes con la categoría más frecuente.

Ambos imputadores están integrados dentro del `ColumnTransformer` del pipeline, garantizando que los valores de imputación se aprendan **únicamente del conjunto de entrenamiento** (sin *data leakage*). La implementación se encuentra en `src/pipeline.py`.

> **¿Por qué no KNN Imputer?** KNN requiere calcular distancias entre filas, lo cual no es semánticamente válido para variables categóricas como `Month` o `VisitorType`. `SimpleImputer` es determinístico, rápido y no introduce sesgos de distancia sobre datos nominales.

### 3. Aumento de Datos

**El modelo final NO utiliza aumento de datos.** Se evaluó **SMOTE** (*Synthetic Minority Oversampling Technique*) como técnica de sobremuestreo de la clase minoritaria (`Revenue=True`), documentado en [`EXPERIMENTOS.md`](EXPERIMENTOS.md). Los resultados mostraron una mejora marginal en precisión (+2.6 pp) pero una caída en recall (-2.4 pp), con un ROC-AUC prácticamente idéntico (+0.0009).

Dado que el desbalance no es extremo (~15.5 % de clase positiva) y que `class_weight='balanced'` ya ajusta las ponderaciones internas del clasificador, se optó por **no aplicar SMOTE** en el pipeline de producción. Esta decisión simplifica el modelo, reduce el riesgo de *data leakage* por ejemplos sintéticos y mantiene la integridad del conjunto de validación.

### 4. Preprocesamiento

- **División estratificada** 70/15/15 (train / validación / test) preservando la proporción de la clase positiva en cada partición.
- **Escalado numérico**: `RobustScaler` (mediana + IQR) para mitigar el impacto de outliers en duraciones y tasas.
- **Codificación categórica**: `OneHotEncoder(handle_unknown="ignore")` para variables categóricas, evitando que categorías no vistas en producción rompan la inferencia.
- Todo se encapsula en un único `sklearn.Pipeline` con `ColumnTransformer`, eliminando el riesgo de *data leakage* entre entrenamiento e inferencia.

### 5. Manejo del Desbalance de Clases

El dataset presenta un desbalance de **84.5 % no compra / 15.5 % compra**. Para mitigarlo se utiliza `class_weight='balanced'` en el `RandomForestClassifier`, que ajusta automáticamente los pesos de cada clase de forma inversamente proporcional a su frecuencia:

```
w_j = n_samples / (n_classes × n_samples_j)
```

Esto penaliza más los errores en la clase minoritaria sin necesidad de modificar los datos de entrenamiento.

---

## Entrenamiento del Modelo

> **Resumen de Resultados**  
> El modelo final (`rf-bal-20`) alcanza **89.41% de accuracy** en test con un threshold optimizado de **0.36**, detectando **199 de 286 compradores reales** (recall 69.58%) con solo 109 falsos positivos. El pipeline completo — desde la carga del CSV hasta la predicción — está contenido en un único `sklearn.Pipeline` serializado.

### Algoritmo

`RandomForestClassifier(class_weight='balanced', random_state=42)`

### Optimización de Hiperparámetros (HPO)

Se utiliza una **búsqueda manual explícita** sobre una grilla de 20 configuraciones nombradas de `RandomForestClassifier`, seguida de **threshold tuning** para optimizar el punto de corte de decisión. Este enfoque es determinístico, reproducible y pedagógicamente transparente.

Protocolo de búsqueda:
1. **División estricta** 70/15/15 (train / validación / test).
2. Para cada configuración de la grilla:
   - Construir un pipeline fresco.
   - Entrenar **únicamente** sobre el conjunto de entrenamiento (70 %).
   - Evaluar sobre el conjunto de validación (15 %).
3. **Threshold tuning**: buscar el umbral óptimo (no fijo 0.5) que maximice la métrica objetivo (F1 o accuracy).
4. Seleccionar la mejor configuración por criterios balanceados (F1 ≥ 0.65, accuracy ≥ 89%).
5. Reentrenar el pipeline ganador sobre **Train + Validation (85 %)**.
6. Evaluar **una única vez** sobre el conjunto de test (15 %).

Grilla de configuraciones (`SEARCH_CONFIGS`) — 20 combinaciones explorando:
- Diferentes números de estimadores (100-800)
- Profundidades variables (10, 20, 30, None)
- Criterios de split (`gini`, `entropy`)
- Opciones estructurales (`bootstrap`, `max_features`)

### Métricas Finales (Test Set — Modelo Balance-Focused)

| Métrica | Valor |
|---------|-------|
| Accuracy | **0.8941** |
| Precision | 0.6461 |
| Recall | **0.6958** |
| F1 Score | **0.6700** |
| ROC AUC | **0.9289** |
| Threshold óptimo | **0.36** |

### Matriz de Confusión (Test Set)

|  | Pred: No | Pred: Sí |
|--|----------|----------|
| **Actual: No** | 1455 | 109 |
| **Actual: Sí** | 87 | 199 |

- **Verdaderos Negativos (1455)**: Sesiones que no compraron y el modelo acertó.
- **Falsos Positivos (109)**: Sesiones que no compraron pero el modelo predijo compra.
- **Falsos Negativos (87)**: Sesiones que compraron pero el modelo predijo no compra.
- **Verdaderos Positivos (199)**: Sesiones que compraron y el modelo acertó.

> **¿Por qué threshold tuning?**  
> El umbral por defecto (0.5) no siempre es óptimo. Ajustarlo a 0.36 permitió ganar ~3 puntos de F1 y detectar 33 compradores más que con el umbral fijo, demostrando que el punto de corte es tan importante como las predicciones probabilísticas.

> **¿Por qué accuracy no es suficiente?**  
> Con un desbalance del 84.5 %, un modelo trivial que siempre predice "no compra" alcanzaría ~84.5 % de accuracy sin haber aprendido nada. Por eso optimizamos **F1** como métrica principal de balance, complementada con **ROC-AUC** para medir la capacidad de discriminación.

### Experimentos Adicionales

Se implementaron experimentos comparativos documentados en [`EXPERIMENTOS.md`](EXPERIMENTOS.md):

| Experimento | Accuracy | F1 | Test ROC-AUC | Notas |
|-------------|----------|-----|--------------|-------|
| **Balance-Focused** (20 configs + threshold tuning) | 0.8941 | **0.6700** | **0.9289** | **Modelo de producción** - mejor equilibrio |
| **Accuracy-Focused** (20 configs + threshold tuning) | **0.9011** | 0.6447 | 0.9279 | Si se prioriza maximizar accuracy |
| **Optuna** (TPE 30 trials) | 0.8859 | 0.6512 | 0.9263 | Mejora marginal; herramienta industrial |
| **SMOTE** (Oversampling) | 0.8978 | 0.6631 | 0.9268 | Mayor precision pero menor recall |
| **SMOTE+HPO** (Oversampling + Random Search) | 0.8973 | 0.6520 | 0.9250 | Mejor precision (0.68); menor recall (0.62) |
| **XGBoost** (Gradient Boosting) | 0.8622 | 0.6512 | **0.9301** | Mayor ROC-AUC y recall; menor precision |

Todos los experimentos incluyen sus propias **Matrices de Confusión** detalladas en `EXPERIMENTOS.md`.

---

## API Documentation

### Endpoints

#### `GET /health`

Verifica que el servicio está activo y que el modelo se cargó correctamente.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### `POST /api/predict_intent`

Recibe las 17 features de una sesión y devuelve la predicción de intención de compra.

**Request body (JSON):**
```json
{
  "Administrative": 2,
  "Administrative_Duration": 53.0,
  "Informational": 0,
  "Informational_Duration": 0.0,
  "ProductRelated": 23,
  "ProductRelated_Duration": 1668.28,
  "BounceRates": 0.0083,
  "ExitRates": 0.0163,
  "PageValues": 0.0,
  "SpecialDay": 0.0,
  "Month": "Feb",
  "OperatingSystems": 1,
  "Browser": 1,
  "Region": 9,
  "TrafficType": 3,
  "VisitorType": "Returning_Visitor",
  "Weekend": false
}
```

**Response body (JSON):**
```json
{
  "classification": "compra",
  "probability": 0.7845,
  "human_readable_message": "El usuario presenta un 78.45% de probabilidades de hacer la compra, lo que lo hace bastante probable"
}
```

> **Nota para testing**: Se incluye una [colección de Postman](postman_collection.json) (`postman_collection.json`) con todos los endpoints configurados para pruebas rápidas.

#### `POST /api/predict_intent_fast`

Versión optimizada del endpoint de predicción que **evita la creación de `pd.DataFrame`** y pasa un arreglo NumPy directamente al pipeline. Esto reduce overhead para inferencia de una sola fila y es útil como demostración académica de serving optimizado.

Acepta el **mismo request body** que `/api/predict_intent` y devuelve el **mismo formato de respuesta**.

**Request body (JSON):** *(igual que `/api/predict_intent`)*

**Response body (JSON):** *(igual que `/api/predict_intent`)*

> **Nota:** Este endpoint es un *bonus* para demostración. Si el pipeline interno requiere nombres de columnas (por ejemplo, `ColumnTransformer` con nombres de features), el modelo de producción actual podría necesitar ajustes para soportar arrays planos.

### Lógica de Respuesta

- **`probability`**: Probabilidad de la clase positiva (`Revenue = True`), redondeada a 4 decimales.
- **`classification`**: Se determina comparando la probabilidad contra el **threshold óptimo** cargado desde `artifacts/model_config.json` (0.36 para el modelo balance-focused, ajustado durante el entrenamiento):
  - `"compra"` si `probability >= threshold`
  - `"no_compra"` si `probability < threshold`
- **`human_readable_message`** (niveles de confianza):
  - `>= 0.70`: "bastante probable"
  - `0.50 - 0.69`: "moderadamente probable"
  - `< 0.50`: "poco probable"

---

## Setup Local

### Requisitos

- Python 3.10+
- pip

### Instalación

```bash
# Clonar el repositorio
cd ecommerce-ai-model

# Instalar dependencias
pip install -r requirements.txt
```

### Entrenar el modelo

```bash
python src/train.py
```

Esto genera `artifacts/ecommerce_pipeline.pkl` con el pipeline entrenado.

### Ejecutar la API localmente

```bash
uvicorn src.api:app --host 0.0.0.0 --port 7860 --reload
```

La documentación interactiva (Swagger UI) estará disponible en:
- http://localhost:7860/docs

### Ejecutar tests

```bash
pytest tests/ -v
```

---

## Docker

### Construir la imagen

```bash
docker build -t ecommerce-ai-model .
```

### Ejecutar el contenedor

```bash
docker run -p 7860:7860 ecommerce-ai-model
```

La API estará expuesta en `http://localhost:7860`.

---

## Despliegue

### Render (Producción)

El servicio está desplegado en **Render** utilizando el `Dockerfile` incluido en el repositorio. Render construye la imagen automáticamente desde GitHub y expone el servicio en el puerto configurado.

1. Conectar el repositorio de GitHub a Render.
2. Seleccionar "Web Service" y apuntar al `Dockerfile`.
3. Render detecta automáticamente el puerto 7860 definido en el `EXPOSE` del Dockerfile.
4. El modelo serializado (`artifacts/ecommerce_pipeline.pkl`) está incluido en el repositorio y se empaqueta en la imagen durante el build.

### Hugging Face Spaces (Alternativo)

1. Crear un nuevo Space en Hugging Face seleccionando el SDK **Docker**.
2. Subir el contenido de este repositorio.
3. Hugging Face construye la imagen y expone el servicio en el puerto 7860.

> **Nota**: El artefacto serializado debe estar presente en el repositorio antes del despliegue. Tanto Render como Hugging Face Spaces lo incluyen en la imagen Docker para un *cold start* determinista.

---

## Decisiones de Arquitectura

| Decisión | Elección | Racional |
|----------|----------|----------|
| Modelo | `RandomForestClassifier` | Buen baseline para tablas mixtas; robusto a no linealidades y fácil de desplegar. |
| HPO | Búsqueda manual explícita (20 configs) | Transparente, reproducible y pedagógica; evita fugas de test durante la búsqueda. |
| Escalado numérico | `RobustScaler` | Protege contra outliers en duraciones y tasas. |
| Preprocesamiento | `ColumnTransformer` dentro de `Pipeline` | Elimina *train/serve skew* y *data leakage*. |
| División | Estratificada 70/15/15 | Preserva el desbalance ~84.5/15.5 en todos los conjuntos. |
| API | FastAPI + Pydantic | Validación automática de requests, documentación OpenAPI integrada y alto rendimiento. |
| Threshold Tuning | Búsqueda en [0.1, 0.9] paso 0.01 | Optimiza el punto de corte sin reentrenar; ganó ~3 puntos de F1 y 33 compradores detectados. |
| Serialización | `joblib` + `model_config.json` | Preserva pipeline + threshold óptimo + métricas para serving determinista. |

---

## Estructura del Proyecto

```
ecommerce-ai-model/
├── artifacts/
│   ├── ecommerce_pipeline.pkl              # Modelo principal serializado
│   └── model_config.json                   # Config con threshold óptimo y métricas
├── data/
│   └── dataset_shop.csv                # Dataset raw
├── src/
│   ├── __init__.py
│   ├── api.py                          # FastAPI app
│   ├── pipeline.py                     # ETL, preprocesamiento y splits
│   ├── train.py                        # Entrenamiento + HPO + serialización
│   ├── train_optuna.py                 # Experimento Optuna (30 trials TPE)
│   ├── train_smote.py                  # Experimento SMOTE (oversampling)
│   ├── train_smote_hpo.py              # Experimento SMOTE+HPO (oversampling + HPO)
│   ├── train_xgboost.py                # Experimento XGBoost (scale_pos_weight)
│   └── threshold_tuning.py             # Script de experimentación con threshold tuning
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py                # Tests unitarios del pipeline
│   ├── test_api.py                     # Tests de integración de la API
│   ├── test_train.py                   # Tests del entrenamiento
│   ├── test_docker.py                  # Tests del Dockerfile
│   └── test_docs.py                    # Tests de documentación
├── Dockerfile                          # Imagen para HF Spaces
├── .dockerignore
├── requirements.txt
├── README.md                           # Este documento
└── EXPERIMENTOS.md                     # Documentación de experimentos Optuna, SMOTE, SMOTE+HPO y XGBoost
```
