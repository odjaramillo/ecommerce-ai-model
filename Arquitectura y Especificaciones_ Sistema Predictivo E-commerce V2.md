# **Arquitectura y Especificaciones: Sistema Predictivo E-commerce**

## **1\. Topología del Repositorio (Clean Architecture)**

El proyecto se estructura separando estrictamente el entorno de experimentación matemática del entorno de servicio, optimizando el ciclo de vida del software.

* **data/**: Directorio local (ignorado en .gitignore) para dataset\_shop.csv.  
* **notebooks/**: Entorno de experimentación para el Análisis Exploratorio de Datos (EDA), multicolinealidad y pruebas de asimetría vectorial.  
* **src/**  
  * **pipeline.py**: Módulo ETL. Particionamiento estratificado (70/15/15), mapeo tensorial y escalado robusto.  
  * **train.py**: Bucle de optimización de hiperparámetros (HPO) minimizando entropía cruzada y serialización.  
  * **api.py**: Capa de controladores REST (FastAPI), validadores de esquemas estables (Pydantic) y lógica de negocio.  
* **artifacts/**: Directorio de salida para el modelo binario unificado (ecommerce\_pipeline.pkl).  
* **requirements.txt**: Declaración determinista de dependencias.  
* **README.md**: Documentación científica y operativa de los endpoints.

## **2\. Flujo Lógico y Matemático (Data Science Pipeline)**

El sistema se divide en dos macro-procesos asíncronos y desacoplados:

### **A. Flujo de Entrenamiento (Offline)**

1. **Aislamiento Tensorial:** Extracción de la variable dependiente Revenue. Partición estratificada conservando la distribución asimétrica (84.5% vs 15.5%): Train (70%), Valid (15%), Test (15%). Bloqueo absoluto del tensor de prueba para evitar Data Leakage.  
2. **Transformación Ortogonal (ColumnTransformer):**  
   * Variables continuas (10): Escalado mediante mediana e IQR (RobustScaler) para mitigar gradientes explosivos de outliers.  
   * Variables nominales (8): Expansión del espacio vectorial vía codificación One-Hot, previniendo sesgos de magnitud escalar.  
3. **Optimización Predictiva:** Inyección de tensores al ensamblador (RandomForestClassifier) compensando la asimetría mediante ponderaciones algebraicas en la métrica de impureza de Gini (w\_j \= N / (k × N\_j)).  
4. **Serialización:** Congelamiento del transformador y el clasificador en un grafo de ejecución inmutable.

### **B. Flujo de Inferencia (Online)**

1. **Arranque (Cold Start):** Inicialización del worker y carga del artefacto binario unificado en memoria RAM L1/L2 (Inferencia O(1)).  
2. **Validación HTTP:** El endpoint intercepta el payload JSON y valida la integridad de los 18 atributos exigidos.  
3. **Evaluación Vectorial:** Transformación a vector 1xD, proyección geométrica estática y emisión de probabilidad continua P(y=1 | X).  
4. **Respuesta Heurística:** Formateo del output JSON aplicando umbrales de decisión (\> 0.5) para el human\_readable\_entry.

## **3\. Análisis de Trade-offs: Estrategias de Optimización (HPO)**

| Estrategia | Complejidad Temporal | Veredicto Arquitectónico |
| :---- | :---- | :---- |
| Grid Search | O(N × |H|) | Descartado. Costo computacional prohibitivo al escalar combinaciones de profundidad y estimadores en RandomForest. |
| Random Search | O(N × K) | Seleccionado. Muestreo estocástico uniforme. Establece línea base robusta sin saturar la RAM local. |
| Optuna (Bayesiana) | O(N × K) \+ Heurística | Óptimo. Modela función objetivo con Procesos Gaussianos. Excede el alcance requerido pero maximiza ROC-AUC iterativamente. |

## **4\. Especificación de la Capa de Presentación (REST API)**

Arquitectura stateless orientada a latencia O(1). Contrato de datos rígido para acoplamiento con ecosistemas frontend externos.

| Endpoint | Método | Lógica Funcional |
| :---- | :---- | :---- |
| /health | GET | Liveness probe. Verifica estado del loop asíncrono y disponibilidad del binario pre-cargado. |
| /api/predict\_intent | POST | Validación de tensor 1x18. Emite clase dicotómica, probabilidad continua y directiva legible (string). |

## **5\. Arquitectura de Despliegue (Docker & Cloud)**

Implementación orientada a mitigar errores Out Of Memory (OOM) y aislar dependencias del SO host.

* **Proveedor Cloud:** Hugging Face Spaces (Tier gratuito 16 GB RAM / 2 vCPU). Supera bloqueos monohilo de Render/Railway.  
* **Gestor ASGI:** Uvicorn configurado con 1 worker.  
* **Networking:** Binding dinámico al puerto 7860\.

### **Especificación Dockerfile**

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1  
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .  
RUN pip install \--no-cache-dir \-r requirements.txt

COPY . .

EXPOSE 7860

CMD \["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"\]  
