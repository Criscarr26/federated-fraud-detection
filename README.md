# federated_fraud

Proyecto de ejemplo para Detección Federada de Fraude (Flower + PyTorch + scikit-learn).

Contenido principal:

- data/fraud_data.csv — placeholder/sintético pequeño (reemplazar por dataset real).
- fed_fraud/ — paquete con código (data_utils, models, privacy, client/server, experiments, etc.).
- notebooks/01_eda_y_calidad_datos.ipynb — notebook de EDA y validación de calidad.

Instalación (ejemplo con PowerShell):

1) Crear y activar entorno virtual:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
2) Actualizar pip e instalar dependencias:
   pip install -U pip
   pip install -r requirements.txt

Notas:
- Para instalar `torch` elige la variante adecuada en https://pytorch.org/ si necesitas GPU.
- `flwr` (Flower) está incluido en requirements.txt.

Ejecutar experimentos:
   python -c "from fed_fraud.experiments import run_experiments; run_experiments(data_path='data/fraud_data.csv', output_csv='experiments_results.csv')"

Copiar el proyecto a C:\Users\TuNombre\Documents\federated_fraud (reemplaza TuNombre):

Opción simple (PowerShell):
   $dest = 'C:\Users\TuNombre\Documents\federated_fraud'
   New-Item -ItemType Directory -Force -Path $dest
   Copy-Item -Path .\federated_fraud\* -Destination $dest -Recurse -Force

Opción con robocopy (robusta):
   robocopy .\federated_fraud "C:\Users\TuNombre\Documents\federated_fraud" /MIR

Opción zip:
   Compress-Archive -Path .\federated_fraud\* -DestinationPath C:\Users\TuNombre\Documents\federated_fraud.zip -Force
   Expand-Archive -Path C:\Users\TuNombre\Documents\federated_fraud.zip -DestinationPath C:\Users\TuNombre\Documents\federated_fraud -Force

Verificar archivos:
   Get-ChildItem -Path 'C:\Users\TuNombre\Documents\federated_fraud' -Recurse

Siguientes mejoras sugeridas:
- Generar un dataset sintético grande para pruebas si no hay datos reales.
- Integrar callbacks de Flower para métricas por ronda y guardado automático de modelo federado.
