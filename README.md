# Sistema Predictivo de Intención de Compra E-commerce

Servicio de Machine Learning para predecir la intención de compra de sesiones de navegación en un sitio de e-commerce. Incluye un pipeline completo de entrenamiento offline con Random Forest, búsqueda manual de hiperparámetros, threshold tuning y una API REST con FastAPI para inferencia en tiempo real.

---

## Arquitectura

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  dataset_shop   │────▶│  src/pipeline.py │────▶│  src/train.py       │
│   (raw CSV)     │     │  Clean + Split   │     │  ManualSearch       │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                                                           │
                                                           ▼
                                              ┌─────────────────────┐
                                              │ artifacts/ecommerce │
                                              │   _pipeline.pkl     │
                                              └─────────────────────┘
                                                           │
                                                           ▼
                                              ┌─────────────────────┐
                                              │    src/api.py       │
                                              │   FastAPI + UVicorn │
                                              │    Puerto 7860      │
                                              └─────────────────────┘
```

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

### Preprocesamiento

- **División estratificada** 70/15/15 (train / validación / test) preservando la proporción de la clase positiva.
- **Escalado numérico**: `RobustScaler` (mediana + IQR) para robustez ante outliers en duraciones y tasas de rebote.
- **Codificación categórica**: `OneHotEncoder(handle_unknown="ignore")` para variables categóricas.
- Todo se encapsula en un único `sklearn.Pipeline` para evitar *data leakage* entre entrenamiento e inferencia.

### Manejo del Desbalance de Clases

El dataset presenta un desbalance aproximado de **84.5 % no compra / 15.5 % compra**. Para mitigarlo se utiliza `class_weight='balanced'` en el `RandomForestClassifier`, ajustando automáticamente los pesos inversamente proporcionales a la frecuencia de cada clase.

---

## Entrenamiento del Modelo

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

## Despliegue en Hugging Face Spaces

1. Crear un nuevo Space en Hugging Face seleccionando el SDK **Docker**.
2. Subir el contenido de este repositorio (incluyendo `artifacts/ecommerce_pipeline.pkl`).
3. Hugging Face construirá automáticamente la imagen usando el `Dockerfile` y expondrá el servicio en el puerto configurado (7860).
4. El artefacto serializado se incluye en la imagen para un *cold start* determinista y rápido.

> **Nota**: Para evitar tiempos de inicio lentos, el modelo debe entrenarse previamente y el archivo `artifacts/ecommerce_pipeline.pkl` debe estar presente en el repositorio antes del despliegue.

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
│   ├── model_config.json                   # Config con threshold óptimo y métricas
│   ├── ecommerce_pipeline_optuna.pkl       # Modelo Optuna (experimento)
│   ├── ecommerce_pipeline_smote.pkl        # Modelo SMOTE (experimento)
│   ├── ecommerce_pipeline_smote_hpo.pkl    # Modelo SMOTE+HPO (experimento)
│   └── ecommerce_pipeline_xgboost.pkl      # Modelo XGBoost (experimento)
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
