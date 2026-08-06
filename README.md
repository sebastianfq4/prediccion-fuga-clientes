# Predicción de Fuga de Clientes para una Entidad Bancaria

Modelo de clasificación para identificar clientes con riesgo de fuga (attrition), comparando un enfoque interpretable (GAM) frente a un modelo de alto rendimiento predictivo (LightGBM).

## Descripción

El attrition o fuga de clientes es uno de los principales desafíos del sector bancario: identificar de forma anticipada a los clientes con mayor probabilidad de abandonar los servicios permite implementar estrategias de retención focalizadas. Este proyecto desarrolla y compara dos modelos predictivos sobre un dataset de más de 70,000 clientes bancarios (caso BBVA).

## Lo que hice

- Desarrollé un modelo de clasificación en Python para identificar clientes con riesgo de fuga, utilizando información de más de 70,000 clientes y 53 variables.
- Realicé ingeniería de variables sobre el historial temporal de los clientes (6 meses de datos por indicador), construyendo 28 features agregadas: promedio, desviación estándar, tendencia lineal y delta.
- Apliqué balanceo de clases con SMOTE sobre el conjunto de entrenamiento para manejar el desbalance de la variable objetivo (~15.5% de tasa de fuga).
- Comparé modelos predictivos — un Modelo Aditivo Generalizado (GAM) y LightGBM — mediante métricas de clasificación (AUC-ROC, precision, recall, F1) y validación cruzada estratificada, para seleccionar la alternativa con mejor desempeño según el objetivo (interpretabilidad vs. poder predictivo).
- Generé un scoring de riesgo por cliente, segmentado en deciles, para apoyar estrategias de retención y toma de decisiones.

## Metodología

| Etapa | Técnica |
|---|---|
| Ingeniería de variables temporales | Promedio, desviación estándar, tendencia (regresión lineal), delta sobre 6 meses de historial |
| Balanceo de clases | SMOTE (solo en train) |
| Modelo interpretable | GAM (Generalized Additive Model) con splines cúbicos |
| Modelo de alto desempeño | LightGBM (gradient boosting) |
| Interpretabilidad | Gráficos de dependencia parcial (GAM), importancia de variables y SHAP (LightGBM) |
| Selección de umbral | Índice de Youden sobre la curva ROC |
| Validación | Validación cruzada estratificada (k=5) |

## Resultados principales

- Dataset: **70,000 clientes**, 53 variables originales, tasa de fuga ≈ **15.5%**.
- LightGBM: mayor capacidad predictiva, captura interacciones entre variables automáticamente, usa las 81 variables (originales + temporales agregadas).
- GAM: menor poder predictivo pero alta interpretabilidad mediante funciones suaves por variable, útil en contextos donde se requiere justificar decisiones ante reguladores o áreas de riesgo.
- Se generó un scoring de riesgo por decil, permitiendo priorizar campañas de retención sobre el segmento de mayor riesgo real.

## Datos

El dataset (`train_clientes.csv`, caso BBVA) contiene información bancaria de clientes y **no se incluye en este repositorio** por su naturaleza sensible.

## Contenido del repositorio

- `Comparacion_GAM_LightGBM.ipynb` — Notebook con el pipeline completo (EDA, ingeniería de features, preprocesamiento, entrenamiento de ambos modelos, interpretabilidad, evaluación y scoring).
- `informe_GAM_LightGBM.pdf` — Informe técnico detallado del proceso, decisiones metodológicas y resultados.

## Herramientas

Python · pandas · numpy · scikit-learn · pygam · lightgbm · shap · imbalanced-learn (SMOTE) · matplotlib / seaborn · Google Colab

## Autor

Michael Sebastian Flores Quispe (trabajo de curso)
