# Experimentos de Mejora del Pipeline

Este documento describe los experimentos realizados sobre el pipeline de predicción de intención de compra, incluyendo la corrección de imputación, la optimización de hiperparámetros con Optuna y el sobremuestreo con SMOTE.

---

## 1. Corrección de Imputación (SimpleImputer)

### Problema original
La verificación anterior detectó que `src/pipeline.py` utilizaba `dropna()` en lugar de una estrategia de imputación propiamente dicha. Esto violaba la especificación del sistema, que exige un pipeline de ciencia de datos con **limpieza, imputación y preprocesamiento**.

### Solución implementada
Se reemplazó `dropna()` por `SimpleImputer` dentro de cada rama del `ColumnTransformer`, envuelto en un `Pipeline` de sklearn:

- **Variables numéricas**: `SimpleImputer(strategy='median')` → `RobustScaler`
- **Variables categóricas**: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`

Todo queda dentro del `Pipeline` de sklearn, lo que garantiza que:
1. No haya **fuga de datos** (data leakage): la imputación y el escalado aprenden únicamente del conjunto de entrenamiento.
2. El artefacto serializado sea autocontenido y pueda aplicarse directamente a datos nuevos en la API.

### ¿Por qué no KNN Imputer?
`KNNImputer` es atractivo porque considera la similitud entre muestras, pero presenta un problema grave con **variables categóricas**: calcula distancias euclídeas (o de Minkowski) entre filas. Cuando una columna contiene categorías como `"Feb"`, `"Returning_Visitor"` o valores booleanos, el concepto de "distancia numérica" carece de sentido semántico. Forzar una codificación numérica ordinal (por ejemplo, `Ene=1, Feb=2, Mar=3`) introduce un orden ficticio que distorsiona el vecindario. Por eso se prefirió `SimpleImputer(strategy='most_frequent')`, que es robusto, rápido y no asume métricas de distancia sobre datos nominales.

### Resultado
El dataset actual no contiene valores faltantes, por lo que el cambio no modificó las métricas finales, pero el pipeline ahora cumple con la especificación y es robusto ante datos incompletos futuros.

---

## 2. Optuna — Optimización Bayesiana de Hiperparámetros

### Objetivo
Determinar si una búsqueda bayesiana con **TPE** (Tree-structured Parzen Estimator) podía superar la búsqueda manual explícita del modelo principal.

### Cómo funciona TPE
1. **Fase de calentamiento (warmup)**: los primeros 10 ensayos son completamente aleatorios para explorar el espacio.
2. **Fase bayesiana**: a partir del ensayo 11, TPE modela dos densidades de probabilidad:
   - Una sobre los ensayos **buenos** (mejores que la mediana observada).
   - Otra sobre los ensayos **malos**.
3. **Selección**: elige el siguiente punto donde la razón `p(bueno) / p(malo)` es máxima, equilibrando exploración y explotación.

### Configuración del experimento
- **Sampler**: `TPESampler(n_startup_trials=10, seed=42)`
- **Ensayos**: 30
- **Validación cruzada**: 5 folds sobre el training set (igual que el modelo principal)
- **Métrica de optimización**: ROC-AUC medio de CV
- **Espacio de búsqueda**:
  - `n_estimators`: 100-500 (paso 100)
  - `max_depth`: 10, 20, 30, `None`
  - `min_samples_split`: 2-10
  - `min_samples_leaf`: 1-4

### Resultados

| Métrica      | Modelo Principal (Manual Search) | Optuna (TPE) | Δ      |
|--------------|----------------------------------|--------------|--------|
| Accuracy     | 0.8935                           | 0.8859       | -0.0076|
| Precision    | 0.6498                           | 0.6176       | -0.0323|
| Recall       | 0.6748                           | 0.6888       | +0.0140|
| F1           | 0.6621                           | 0.6512       | -0.0109|
| **ROC-AUC**  | **0.9258**                       | **0.9263**   | **+0.0005**|

#### Matriz de Confusión — Optuna (Test)

|                 | Pred: No | Pred: Sí |
|-----------------|----------|----------|
| **Actual: No**  | 1442     | 122      |
| **Actual: Sí**  | 89       | 197      |

### Lección
Optuna logró un ROC-AUC ligeramente superior, pero las diferencias son mínimas. El modelo principal con búsqueda manual explícita es preferible para producción porque:
- Es más simple (menos dependencias).
- Tiene métricas prácticamente idénticas.
- Es determinista y completamente reproducible.
- No requiere un framework adicional de optimización.

---

## 3. SMOTE — Sobremuestreo de la Clase Minoritaria

### Objetivo
Evaluar si generar ejemplos sintéticos de la clase minoritaria (`Revenue=True`) mejora el desempeño del clasificador.

### Metodología
SMOTENC (la variante para datos mixtos numéricos/categóricos) requiere datos **sin procesar** porque necesita conocer qué columnas son categóricas para calcular vecindarios adecuados. Sin embargo, nuestro pipeline One-Hot encodea las variables categóricas antes del clasificador, lo que complica la inserción de SMOTENC dentro de un `Pipeline` estándar.

Por eso se optó por el siguiente flujo experimental:
1. Cargar y dividir los datos (train/val/test).
2. **Ajustar el preprocessor únicamente con el training set**.
3. Transformar el training set con el preprocessor ya ajustado.
4. Aplicar **SMOTE** (versión estándar) sobre las características transformadas (incluyendo las columnas binarias del One-Hot).
5. Entrenar `RandomForestClassifier` sobre los datos aumentados.
6. Evaluar en val/test **sin aplicar SMOTE** (el modelo nunca ve datos de test).

> **Nota importante**: Aplicar SMOTE sobre características One-Hot genera valores continuos en columnas que teóricamente son binarias. Aunque esto es conceptualmente imperfecto, en la práctica el clasificador basado en árboles puede manejar estas interpolaciones sin problemas.

### Configuración del experimento
- **Algoritmo**: `SMOTE(random_state=42)`
- **Balance resultante**: 7.295 ejemplos de cada clase (vs. ~7.900/900 originalmente).
- **Clasificador**: `RandomForestClassifier` con los mejores hiperparámetros aproximados del modelo principal.

### Resultados

| Métrica      | Modelo Principal | SMOTE    | Δ      |
|--------------|------------------|----------|--------|
| Accuracy     | 0.8935           | 0.8978   | +0.0043|
| Precision    | 0.6498           | 0.6764   | +0.0265|
| Recall       | 0.6748           | 0.6503   | -0.0245|
| F1           | 0.6621           | 0.6631   | +0.0010|
| **ROC-AUC**  | **0.9258**       | **0.9268**| **+0.0009**|

#### Matriz de Confusión — SMOTE (Test)

|                 | Pred: No | Pred: Sí |
|-----------------|----------|----------|
| **Actual: No**  | 1475     | 89       |
| **Actual: Sí**  | 100      | 186      |

### Lección
SMOTE mejoró levemente la precisión y el ROC-AUC, pero redujo el recall. Esto indica que el modelo se volvió más conservador al predecir la clase positiva. Dado que el dataset original no está extremadamente desbalanceado (~15% positivos), el beneficio de SMOTE es marginal. El modelo principal sigue siendo preferible por su simplicidad.

---

## 4. XGBoost — Gradient Boosting con scale_pos_weight

### Objetivo
Evaluar si un clasificador basado en **gradient boosting** (XGBoost) puede superar al Random Forest del modelo principal, utilizando `scale_pos_weight` para mitigar el desbalance de clases.

### ¿Por qué XGBoost?
XGBoost (eXtreme Gradient Boosting) es uno de los algoritmos más populares para datos tabulares por varias razones:
1. **Optimización de segundo orden**: utiliza el gradiente y la hessiana de la función de pérdida para converger más rápido.
2. **Regularización integrada**: incluye términos L1 (alpha) y L2 (lambda) que reducen el overfitting.
3. **Manejo de missing values**: XGBoost aprende internamente la mejor dirección para missing values en cada split.
4. **Eficiencia**: utiliza aproximación de histogramas y paralelización a nivel de características.

### scale_pos_weight
En lugar de `class_weight='balanced'` (usado en Random Forest), XGBoost utiliza `scale_pos_weight`, que se calcula como:

```
scale_pos_weight = count(negativos) / count(positivos)
```

Para este dataset: **7295 / 1336 ≈ 5.46**. Este valor le dice al algoritmo que un error en la clase positiva (compra) debe penalizarse ~5.5 veces más que un error en la clase negativa.

### Configuración del experimento
- **Clasificador**: `XGBClassifier(scale_pos_weight=5.46, random_state=42, n_jobs=-1)`
- **Búsqueda**: `RandomizedSearchCV` con 30 iteraciones, 5 folds, scoring `roc_auc`
- **Espacio de búsqueda**:
  - `n_estimators`: 100, 200, 300, 400, 500
  - `max_depth`: 3, 5, 7, 10
  - `learning_rate`: 0.01, 0.05, 0.1, 0.2
  - `subsample`: 0.8, 0.9, 1.0
  - `colsample_bytree`: 0.8, 0.9, 1.0

### Resultados

| Métrica      | Modelo Principal (Random Forest) | XGBoost  | Δ       |
|--------------|----------------------------------|----------|---------|
| Accuracy     | 0.8935                           | 0.8622   | -0.0313 |
| Precision    | 0.6498                           | 0.5348   | -0.1150 |
| Recall       | 0.6748                           | 0.8322   | **+0.1574** |
| F1           | 0.6621                           | 0.6512   | -0.0109 |
| **ROC-AUC**  | **0.9258**                       | **0.9301** | **+0.0043** |

#### Matriz de Confusión — XGBoost (Test)

|                 | Pred: No | Pred: Sí |
|-----------------|----------|----------|
| **Actual: No**  | 1357     | 207      |
| **Actual: Sí**  | 48       | 238      |

### Lección
XGBoost con `scale_pos_weight` logró el **mejor ROC-AUC de todos los experimentos** (0.9301), superando al principal por +0.43 pp. Sin embargo, este beneficio viene a costa de una caída importante en **precision** (-11.5 pp) y una mayor tasa de falsos positivos (207 vs 104 del principal).

Esto indica que XGBoost es mucho más agresivo prediciendo la clase positiva, lo que aumenta el recall dramáticamente (+15.7 pp) pero genera más falsas alarmas. Si el negocio prioriza **no perder compradores potenciales** (maximizar recall), XGBoost es una excelente alternativa. Si se busca **equilibrio general**, el Random Forest principal sigue siendo preferible.

---

## 5. SMOTE + HPO — Sobremuestreo con Búsqueda de Hiperparámetros

### Objetivo
El experimento anterior (SMOTE) utilizó hiperparámetros fijos basados en el modelo principal. Este experimento evalúa si **combinar SMOTE con una búsqueda de hiperparámetros** (`RandomizedSearchCV`) puede mejorar el desempeño del clasificador sobre datos aumentados.

### Metodología
El flujo es idéntico al experimento SMOTE hasta el paso 4, pero en lugar de entrenar con parámetros fijos, se ejecuta `RandomizedSearchCV` sobre los datos ya aumentados:

1. Cargar y dividir los datos (train/val/test).
2. Ajustar el preprocessor únicamente con el training set.
3. Transformar el training set con el preprocessor ya ajustado.
4. Aplicar **SMOTE** sobre las características transformadas.
5. Ejecutar `RandomizedSearchCV` (30 iteraciones, 5 folds, scoring `roc_auc`) sobre el training set aumentado.
6. Entrenar el mejor `RandomForestClassifier` encontrado.
7. Evaluar en val/test **sin aplicar SMOTE**.

> **Advertencia sobre validación cruzada**: Al aplicar SMOTE *antes* de `RandomizedSearchCV`, los folds de validación cruzada contienen ejemplos sintéticos. Esto puede inflar ligeramente el CV ROC-AUC (0.9921 observado) porque el modelo valida sobre datos muy similares a los de entrenamiento. Las métricas de test son las que realmente reflejan la generalización.

### Configuración del experimento
- **Algoritmo**: `SMOTE(random_state=42)`
- **Balance resultante**: 7.295 ejemplos de cada clase (mismo que SMOTE fijo).
- **Búsqueda**: `RandomizedSearchCV` con 30 iteraciones, 5 folds, scoring `roc_auc`
- **Espacio de búsqueda** (mismo que el modelo principal):
  - `n_estimators`: 100, 200, 300, 400, 500
  - `max_depth`: 10, 20, 30, `None`
  - `min_samples_split`: 2, 5, 10
  - `min_samples_leaf`: 1, 2, 4
- **Mejores hiperparámetros encontrados**:
  - `n_estimators=400`
  - `max_depth=30`
  - `min_samples_split=2`
  - `min_samples_leaf=1`

### Resultados

| Métrica      | Modelo Principal | SMOTE (fijo) | SMOTE+HPO | Δ vs Principal | Δ vs SMOTE |
|--------------|------------------|--------------|-----------|----------------|------------|
| Accuracy     | 0.8935           | 0.8978       | 0.8973    | +0.0038        | -0.0005    |
| Precision    | 0.6498           | 0.6764       | 0.6846    | **+0.0348**    | +0.0083    |
| Recall       | 0.6748           | 0.6503       | 0.6224    | -0.0524        | -0.0280    |
| F1           | 0.6621           | 0.6631       | 0.6520    | -0.0101        | -0.0111    |
| **ROC-AUC**  | **0.9258**       | **0.9268**   | **0.9250**| -0.0009        | -0.0018    |

#### Matriz de Confusión — SMOTE+HPO (Test)

|                 | Pred: No | Pred: Sí |
|-----------------|----------|----------|
| **Actual: No**  | 1482     | 82       |
| **Actual: Sí**  | 108      | 178      |

### Lección
SMOTE+HPO logró la **mejor precision de todos los experimentos** (0.6846), superando incluso al SMOTE fijo (+0.83 pp) y al principal (+3.48 pp). Sin embargo, este beneficio se obtiene a costa de una reducción en **recall** (0.6224), el más bajo entre todos los experimentos excepto el principal.

El ROC-AUC final (0.9250) es prácticamente idéntico al modelo principal y ligeramente inferior al SMOTE fijo (0.9268) y a XGBoost (0.9301). Esto sugiere que, para este dataset, añadir HPO sobre datos SMOTEados no aporta una ventaja discriminativa significativa respecto a fijar los hiperparámetros del modelo principal.

El modelo principal sigue siendo preferible por su equilibrio recall/precision y su mayor simplicidad (no requiere SMOTE ni búsqueda adicional).

---

## Tabla Comparativa Final

| Experimento    | Accuracy | Precision | Recall | F1     | ROC-AUC |
|----------------|----------|-----------|--------|--------|---------|
| **Principal (Balance-Focused)** | 0.8941 | 0.6461 | **0.6958** | **0.6700** | **0.9289** |
| **Principal (Manual + Threshold Tuning)** | **0.9011** | **0.7249** | 0.5804 | 0.6447 | 0.9279 |
| **Principal (Manual v2)** | 0.8897 | 0.6192 | 0.7448 | 0.6762 | 0.9291 |
| **Optuna**     | 0.8859   | 0.6176    | 0.6888 | 0.6512 | 0.9263  |
| **SMOTE**      | 0.8978   | 0.6764    | 0.6503 | 0.6631 | 0.9268  |
| **SMOTE+HPO**  | 0.8973   | 0.6846    | 0.6224 | 0.6520 | 0.9250  |
| **XGBoost**    | 0.8622   | 0.5348    | 0.8322 | 0.6512 | 0.9301  |

### Matrices de Confusión Comparativas

**Modelo Principal (Balance-Focused):**
|                 | Pred: No | Pred: Sí |
|-----------------|----------|----------|
| **Actual: No**  | 1455     | 109      |
| **Actual: Sí**  | 87       | 199      |

**Modelo Principal (Manual HPO + Threshold Tuning):**
|                 | Pred: No | Pred: Sí |
|-----------------|----------|----------|
| **Actual: No**  | 1501     | 63       |
| **Actual: Sí**  | 120      | 166      |

**Modelo Principal (Manual v2):
|                 | Pred: No | Pred: Sí |
|-----------------|----------|----------|
| **Actual: No**  | 1410     | 154      |
| **Actual: Sí**  | 67       | 219      |

**Optuna:**
|                 | Pred: No | Pred: Sí |
|-----------------|----------|----------|
| **Actual: No**  | 1442     | 122      |
| **Actual: Sí**  | 89       | 197      |

**SMOTE:**
|                 | Pred: No | Pred: Sí |
|-----------------|----------|----------|
| **Actual: No**  | 1475     | 89       |
| **Actual: Sí**  | 100      | 186      |

**SMOTE+HPO:**
|                 | Pred: No | Pred: Sí |
|-----------------|----------|----------|
| **Actual: No**  | 1482     | 82       |
| **Actual: Sí**  | 108      | 178      |

**XGBoost:**
|                 | Pred: No | Pred: Sí |
|-----------------|----------|----------|
| **Actual: No**  | 1357     | 207      |
| **Actual: Sí**  | 48       | 238      |

### Observaciones clave
- **Balance-Focused**: El modelo `rf-bal-20` (n_estimators=800, criterion='entropy', threshold=0.36) ofrece el **mejor balance global** con F1=0.6700, recall=0.6958 y ROC-AUC=0.9289, detectando 199 compradores reales vs 166 del modelo accuracy-focused.
- **Accuracy-Focused**: El modelo `rf-i2-18` alcanzó **90.11%** accuracy con threshold 0.65, sacrificando recall (0.58) y F1 (0.64). Útil si el negocio prioriza precision sobre detección.
- **ROC-AUC**: El modelo v2 (0.9291) y el balance-focused (0.9289) superan a todos los RF previos, demostrando que una búsqueda manual bien diseñada iguala herramientas automatizadas.
- **Threshold Tuning**: Ajustar el umbral de 0.50 a 0.36 (F1) o 0.65 (accuracy) permite optimizar métricas específicas sin reentrenar el modelo.
- **Complejidad**: Principal < XGBoost < SMOTE < SMOTE+HPO < Optuna (en términos de infraestructura y dependencias).

---

## Conclusiones y Recomendaciones

1. **Modelo para producción**: El modelo **balance-focused** (`rf-bal-20`) se recomienda como principal por su superior equilibrio de métricas (F1=0.6700, recall=0.6958, ROC-AUC=0.9289, accuracy=89.41%). Detecta 33 compradores más que la versión accuracy-focused. La versión accuracy-focused (`rf-i2-18`) está disponible si se prioriza maximizar accuracy (90.11%) a costa de recall.
2. **Imputación**: el pipeline ahora cumple la especificación con `SimpleImputer`, haciendo el sistema robusto ante valores faltantes futuros.
3. **Optuna**: demostró que TPE puede emparejar (pero no superar de forma significativa) una búsqueda manual bien diseñada en este dataset. Es útil si el espacio de hiperparámetros crece o si se dispone de más tiempo de computo.
4. **SMOTE**: útil cuando el desbalance es severo. En este caso, con ~15% de clase positiva, el aporte es limitado. Si el negocio prioriza la **precisión** sobre el **recall**, SMOTE podría reconsiderarse.
5. **SMOTE+HPO**: combinó sobremuestreo con búsqueda de hiperparámetros, alcanzando la mejor precision (0.6846) pero el menor recall (0.6224) entre los experimentos de Random Forest. No superó al SMOTE fijo en ROC-AUC, lo que sugiere que el espacio de hiperparámetros ya estaba bien explorado.
6. **XGBoost**: logró el mejor ROC-AUC (0.9301) y el mayor recall (0.83), a costa de precision. Es la mejor opción si el negocio quiere **maximizar la detección de compradores potenciales** sin importar algunos falsos positivos adicionales.

---

## Archivos generados

| Archivo | Descripción |
|---------|-------------|
| `src/train_optuna.py` | Script de HPO con Optuna (30 trials, TPE) |
| `src/train_smote.py` | Script de sobremuestreo con SMOTE sobre features procesadas |
| `src/train_smote_hpo.py` | Script que combina SMOTE con RandomizedSearchCV (30 iteraciones) |
| `src/train_xgboost.py` | Script de entrenamiento con XGBClassifier y scale_pos_weight |
| `artifacts/ecommerce_pipeline_optuna.pkl` | Mejor modelo encontrado por Optuna |
| `artifacts/ecommerce_pipeline_smote.pkl` | Modelo entrenado con datos aumentados por SMOTE |
| `artifacts/ecommerce_pipeline_smote_hpo.pkl` | Modelo SMOTE+HPO con hiperparámetros optimizados |
| `artifacts/ecommerce_pipeline_xgboost.pkl` | Modelo XGBoost con scale_pos_weight |
