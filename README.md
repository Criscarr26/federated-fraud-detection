# Detección de fraude con aprendizaje federado

Proyecto de *Inteligencia Artificial Distribuida* (ITLA): detección de fraude en
transacciones comparando un **modelo centralizado** contra **aprendizaje federado**
(Flower + PyTorch), incluyendo experimentos con **privacidad diferencial**.

## Contenido

| Ruta | Qué es |
|---|---|
| [`notebooks/proyecto_deteccion_fraude.ipynb`](notebooks/proyecto_deteccion_fraude.ipynb) | Proyecto completo con análisis, entrenamiento y resultados (EDA, baseline centralizado, federado, gráficas) |
| [`resultados/`](resultados/) | Benchmarks finales: métricas globales y por cliente federado |
| [`docs/ensayo.pdf`](docs/ensayo.pdf) | Ensayo del proyecto |
| `client_pytorch.py` / `server_pytorch.py` | Cliente y servidor de Flower para el entrenamiento federado |
| `models.py` | Redes en PyTorch para clasificación de fraude |
| `privacy.py` | Utilidades de privacidad diferencial (ruido/clipping) |
| `data_utils.py` / `eval_utils.py` | Carga, partición por cliente y métricas de evaluación |
| `train_centralized.py` | Baseline centralizado para comparar |
| `experiments.py` | Orquestación de los experimentos |

`fraud_data.csv` es un placeholder mínimo del formato de datos; el notebook
documenta el dataset real utilizado.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate       # en Windows
pip install -r requirements.txt
```

Para `torch` con GPU, instala la variante adecuada desde https://pytorch.org/.

## Uso

- **Notebook**: abrir `notebooks/proyecto_deteccion_fraude.ipynb` — contiene el
  flujo completo con los resultados ya ejecutados.
- **Federado por consola**: iniciar el servidor y luego los clientes Flower:

```bash
python server_pytorch.py
# en otras terminales:
python client_pytorch.py
```

- **Baseline centralizado**:

```bash
python train_centralized.py
```

## Resultados

Los CSV de [`resultados/`](resultados/) comparan las configuraciones evaluadas
(centralizado vs federado, con y sin privacidad diferencial) con métricas
globales y desagregadas por cliente. El análisis está en el notebook y en el
ensayo.

## Licencia

[MIT](LICENSE)
