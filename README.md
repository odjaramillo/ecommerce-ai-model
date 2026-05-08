# Sistema Predictivo de Intención de Compra E-commerce

Servicio de Machine Learning para predecir la intención de compra de sesiones de navegación en un sitio de e-commerce. Incluye un pipeline completo de entrenamiento offline con Random Forest, Hyperparameter Optimization y una API REST con FastAPI para inferencia en tiempo real.

---

## Arquitectura

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  dataset_shop   │────▶│  src/pipeline.py │────▶│  src/train.py       │
│   (raw CSV)     │     │  Clean + Split   │     │  RandomizedSearchCV │
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

Se utiliza `RandomizedSearchCV` con:
- **30 iteraciones** (`n_iter=30`)
- **5 folds** de Cross-Validation
- **Métrica de scoring**: `roc_auc`

Espacio de búsqueda:

| Hiperparámetro | Valores evaluados |
|----------------|-------------------|
| `n_estimators` | 100, 200, 300, 400, 500 |
| `max_depth` | 10, 20, 30, `None` |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |

### Experimento: 30 vs 80 iteraciones

Como experimento de seguimiento se ejecutó una búsqueda ampliada a **80 iteraciones** incluyendo `max_features` y `bootstrap`. Resultados:

| Métrica | 30 iteraciones (PR#1) | 80 iteraciones | Conclusión |
|---------|----------------------|----------------|------------|
| Best CV ROC AUC | 0.9313 | 0.9333 | +0.0020 (marginal) |
| Test Accuracy | 0.8935 | 0.8757 | **-1.78 pp** |
| Test Precision | 0.6498 | 0.5765 | **-7.33 pp** |
| Test Recall | 0.6748 | 0.7378 | +6.30 pp |
| Test F1 | 0.6621 | 0.6472 | **-1.49 pp** |
| Test ROC AUC | 0.9258 | 0.9239 | **-0.0019** |

**Conclusión**: Aumentar a 80 iteraciones no mejoró la generalización. El modelo de 30 iteraciones presenta mejor equilibrio entre precisión y recall, y un ROC-AUC ligeramente superior en test. Por tanto, **se mantiene la configuración de 30 iteraciones** como óptima para este problema.

### Métricas Finales (Test Set — Modelo de 30 iteraciones)

| Métrica | Valor |
|---------|-------|
| Accuracy | **0.8935** |
| Precision | 0.6498 |
| Recall | 0.6748 |
| F1 Score | 0.6621 |
| ROC AUC | **0.9258** |

### Matriz de Confusión (Test Set)

|  | Pred: No | Pred: Sí |
|--|----------|----------|
| **Actual: No** | 1460 | 104 |
| **Actual: Sí** | 93 | 193 |

- **Verdaderos Negativos (1460)**: Sesiones que no compraron y el modelo acertó.
- **Falsos Positivos (104)**: Sesiones que no compraron pero el modelo predijo compra.
- **Falsos Negativos (93)**: Sesiones que compraron pero el modelo predijo no compra.
- **Verdaderos Positivos (193)**: Sesiones que compraron y el modelo acertó.

> **¿Por qué accuracy no es suficiente?**  
> Con un desbalance del 84.5 %, un modelo trivial que siempre predice "no compra" alcanzaría ~84.5 % de accuracy sin haber aprendido nada. Por eso **ROC-AUC es la métrica principal**: mide la capacidad de discriminación del modelo independientemente del umbral de decisión, y es robusta ante desbalance de clases.

### Experimentos Adicionales

Se implementaron tres experimentos comparativos documentados en [`EXPERIMENTOS.md`](EXPERIMENTOS.md):

| Experimento | Accuracy | Test ROC-AUC | Notas |
|-------------|----------|--------------|-------|
| **Principal** (Random Search 30 iter) | **0.8935** | **0.9258** | Modelo de producción |
| **Optuna** (TPE 30 trials) | 0.8859 | 0.9263 | Mejora marginal (+0.05%); mayor recall, menor precision |
| **SMOTE** (Oversampling) | 0.8978 | 0.9268 | Mayor precision pero menor recall |
| **SMOTE+HPO** (Oversampling + Random Search) | 0.8973 | 0.9250 | Mejor precision (0.68); menor recall (0.62) |
| **XGBoost** (Gradient Boosting) | 0.8622 | **0.9301** | Mayor ROC-AUC y recall; menor precision |

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
- **`classification`**:
  - `"compra"` si `probability >= 0.5`
  - `"no_compra"` si `probability < 0.5`
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
| HPO | `RandomizedSearchCV` (~30 iter) | Mejor cobertura/costo que GridSearch; suficiente para este dataset. |
| Escalado numérico | `RobustScaler` | Protege contra outliers en duraciones y tasas. |
| Preprocesamiento | `ColumnTransformer` dentro de `Pipeline` | Elimina *train/serve skew* y *data leakage*. |
| División | Estratificada 70/15/15 | Preserva el desbalance ~84.5/15.5 en todos los conjuntos. |
| API | FastAPI + Pydantic | Validación automática de requests, documentación OpenAPI integrada y alto rendimiento. |
| Serialización | `joblib` | Formato nativo de sklearn; preserva el pipeline completo (preprocesamiento + modelo). |

---

## Estructura del Proyecto

```
ecommerce-ai-model/
├── artifacts/
│   ├── ecommerce_pipeline.pkl              # Modelo principal serializado
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
│   └── train_xgboost.py                # Experimento XGBoost (scale_pos_weight)
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
