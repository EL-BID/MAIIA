# MAIIA

**M**apeo de **A**sentamientos **I**nformales basado en **I**nteligencia
**A**rtificial

## **¿Qué es?**

MAIIA es un algoritmo que permite mapear de forma automatizada la ubicación de
asentamientos urbanos informales mediante el análisis de imágenes satelitales.
Para facilitar su implementación, se distribuye mediante una imagen de sistema
operativo pre-configurada (vía [Docker](https://www.docker.com/)), junto a
scripts que permiten entrenar un modelo de detección y aplicarlo a imágenes
nuevas en sólo dos pasos.

**¿Por qué la estamos compartiendo al público?**

El algoritmo fue desarrollado con énfasis en su facilidad de implementación,
para bajar las barreras de acceso a herramientas de inteligencia artificial (IA)
y permitir que agencias de gobierno, investigadores y otros actores interesados
puedan aplicarlo a sus propios casos de uso.

## **¿Cómo se usa?**

Los requisitos básicos para usar MAIIA son:

- Sistema operativo Linux (en una instancia local o en la nube)
- Hardware dedicado para aceleración gráfica (NVIDIA GPU)
- Docker instalado y configurado para utilizar la GPU

El proceso de instalación y puesta en marcha se describe a continuación. Es
posible utilizar MAIIA en una laptop o computadora de escritorio propia que
cumpla con los requisitos. Aún así, se sugiere realizar la instalación y puesta
en marcha en una instancia en la nube, por el beneficio de contar con un sistema
estandarizado.

## Instalación

(estos pasos son requeridos sólo una vez)

1. Clonar o descargar el contenido de este repositorio.

2. Instalar
   [Docker](https://docs.docker.com/engine/install/ubuntu/#installation-methods)
   y
   [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Aclaración sobre *Docker*: al instalar docker, es necesario cumplir con los
[pasos de
post-instalación](https://docs.docker.com/engine/install/linux-postinstall/).

Aclaración sobre *nvidia-docker*: como indican los pre-requisitos, es necesario
tener instalado NVIDIA driver. Se puede descargar [el
instalador](https://www.nvidia.com/Download/index.aspx?lang=en-us) o seguir esta
[guía](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
para la instalación a través de paquete.

3.  Para asegurarse de que los containers *docker* puedan acceder a la GPU, editar o crear el archivo `/etc/docker/daemon.json` para que luzca así:

        {
            "default-runtime": "nvidia",
            "runtimes": {
                "nvidia": {
                    "path": "nvidia-container-runtime",
                    "runtimeArgs": []
                }
            }
        }

    Luego de editar el archivo, reiniciar el servicio *docker* para que el cambio tome efecto:

        $ sudo systemctl restart docker.service
        
4. Iniciar la imagen de docker con el código y dependencias. Desde el directorio
   donde se ha descargado MAIIA, ejecutar:

        docker compose build

    Es posible que se requiera la instalación de *docker-compose* **versión 1.28 o superior** al intentar
    ejecutar este comando. Para instalar la versión más reciente disponible, referirse a las instrucciones
    detalladas [aquí](https://docs.docker.com/compose/install/):

    El proceso tomará un buen rato la primera vez, ya que necesitará descargar y
    configurar varios componentes de software. Una vez completada la primera
    puesta en marcha, las subsiguientes serán casi instantáneas.

## Configuración

Estos pasos deben seguirse cada vez que se utiliza el sistema.  Iniciar el
contenedor de Docker, especificando el usuario y grupo correspondientes al
host.

    docker compose run -u $(id -u):$(id -g) maiia /bin/bash

o ejecutar:

    ./start

Si todo salió bien, debería ver lo siguiente:

```

________                               _______________
___  __/__________________________________  ____/__  /________      __
__  /  _  _ \_  __ \_  ___/  __ \_  ___/_  /_   __  /_  __ \_ | /| / /
_  /   /  __/  / / /(__  )/ /_/ /  /   _  __/   _  / / /_/ /_ |/ |/ /
/_/    \___//_/ /_//____/ \____//_/    /_/      /_/  \____/____/|__/


You are running this container as user with ID 1005 and group 1008,
which should map to the ID and group for your user on the Docker host. Great!

tf-docker /app >
```

Desde esta terminal, ejecutará los comandos de entrenamiento y predicción
detallados a continuación.

### Jupyter Notebook

Si desea utilizar Jupyter Notebooks en vez de trabajar por la terminal
solamente, ejecute el siguiente comando.  Por defecto el contenedor iniciará
una instancia de Jupyter Notebooks.

    docker compose run --service-ports -u $(id -u):$(id -g) maiia

o simplemente ejecutar:

    ./start_jupyter

**Importante**: Como la arquitectura consume la memoria de la GPU mientras está cargada, solo es posible ejecutar un notebook por kernel. En caso de ejecutar el notebook de predicción luego del notebook de entrenamiento, es necesario apagar el kernel del notebook de entrenamiento luego de ejecutarlo (menú de Running, click en Shutdown).

## Uso

### Entrenamiento

1.  Proveer insumos de entrenamiento:

    - Colocar en la carpeta `data/train/images/` uno o más archivos en formato
      GeoTIFF (.tif) de imágenes satelitales de alta resolución de una zona con
      áreas informales previamente identificadas.

    - Colocar en la carpeta `data/train/labels/` un archivo vectorial
      georreferenciado en formato *GeoPackage* con las máscaras o límites
      identificados manualmente de asentamientos informales en la zona cubierta
      por la imagen satelital.

      **Importante**: Los *features* del archivo vectorial deben contener una
      columna `class` de tipo *string*, con el valor `A`.

    - (Opcional) Colocar en la carpeta `data/train/areas/` un archivo vectorial
      georreferenciado en formato *GeoPackage* con polígonos de áreas de
      interés (AOIs) donde se desea realizar el entrenamiento.

      Para poder practicar, y como ejemplo, en estas carpetas ya se encuentran
      un archivo raster y uno de máscaras (*GeoPackage* con áreas informales),
      que permiten realizar de inmediato una prueba del sistema.

2.  Iniciar el algoritmo de entrenamiento. Desde la conexión al contenedor de
    docker, ejecutar:

        train

   Esto iniciará el entrenamiento de un modelo de reconocimiento de áreas
   informales utilizando los insumos provistos.  Al completarse, se creará un
   archivo en formato HDF5 (.h5) del modelo entrenado dentro de la carpeta
   `data/models`.

   Es posible modificar los parámetros de entrenamiento con el comando `train`.
   Para ver todas las opciones, ejecutar `train --help`. A continuación se
   muestran:

   ```
Usage: train [OPTIONS]

Options:
  --images-dir TEXT              Path to directory containing images
  --output-model-path TEXT       Path to output model (.h5)
  --labels-path TEXT             Path to labels vector file
  --aoi-path TEXT                Path to AOI vector file
  --size INTEGER                 Size of the extracted chips  [default: 600]
  --step-size INTEGER            Step size of the extracted chips  [default:
                                 100]
  --num-channels INTEGER         Number of channels of input images  [default:
                                 3]
  -E, --epochs INTEGER           Number of epochs to train model  [default:
                                 30]
  -s, --steps-per-epoch INTEGER  Number of steps per epoch  [default: 100]
  -B, --batch-size INTEGER       Batch size  [default: 16]
  -S, --seed INTEGER             Random seed  [default: 42]
  --temp-dir TEXT                Path to temporary directory, which will
                                 contain extracted chips
  --help                         Show this message and exit.
   ```

   Por ejemplo, si desea entrenar por 100 epochs, con un *batch size* de 4, ejecutar:

    train --epochs 100 --batch-size 4


### Predicción / Inferencia

1.  Proveer insumo de detección:

    - Dejar en la carpeta `data/predict/images/` uno o más archivos en formato
      GeoTIFF (.tif) con imágenes satelitales de alta resolución de la zona
      donde identificar áreas informales.

      En esta carpeta ya se encuentra un archivo raster (el mismo utilizado para
      entrenar) sólo con el fin de disponer de un ejemplo para testear el modelo
      entrenado en el paso anterior.

    - (Opcional) Colocar en la carpeta `data/predict/areas/` un archivo
      vectorial georreferenciado en formato *GeoPackage* con polígonos de áreas
      de interés (AOIs) donde se desea realizar la predicción.

2.  Iniciar el algoritmo de detección. Desde la conexión a la imagen de docker, ejecutar:

        predict

    Esto iniciará el reconocimiento de áreas informales en la imagen provista,
    utilizando el último modelo entrenado. Al completar el proceso, quedará
    disponible en la carpeta `data/results` un archivo raster georeferenciado con
    los límites de las áreas informales detectadas en la imagen.

   Es posible modificar los parámetros de entrenamiento con el comando `predict`.
   Para ver todas las opciones, ejecutar `predict --help`. A continuación se
   muestran:

   ```
Usage: predict [OPTIONS]

Options:
  --images-dir TEXT         Path to directory containing images
  --model-path TEXT         Path to trained model (HDF5 format, .h5)
  --output-path TEXT        Path to output vector file (GPKG format, .gpkg)
  --aoi-path TEXT           Path to AOI vector file
  --size INTEGER            Size of the extracted chips  [default: 100]
  -t, --threshold FLOAT     Threshold for filtering (between 0 and 1)
                            [default: 0.5]
  -min, --min-area INTEGER  Minimum area of detected polygons for filtering
                            (in meters)  [default: 500]
  --num-channels INTEGER    Number of channels of input images  [default: 3]
  -B, --batch-size INTEGER  Batch size  [default: 16]
  --temp-dir TEXT           Path to temporary directory, which will contain
                            extracted chips
  --help                    Show this message and exit.
   ```

   Por ejemplo, si desea predecir y filtrar por un umbral de 0.3 y area minima
   de 1000m ejecutar:

    predict --threshold 0.3 --min-area 1000


### Ajustes

En caso de que alguno de los procesos falle por falta de recursos (en
general, por no disponer de suficiente memoria RAM de la GPU) se pueden usar
ciertos parámetros que reducen la carga de procesamiento, a costa de una posible
reducción de la calidad de resultados.

Para entrenar, puede reducirse el parámetro "batch size" (que por defecto es
16), usando `--batch-size n`, donde "n" es un número menor. Por ejemplo:

    train --batch-size 2

Al inferir, puede reducirse el parámetro "window size" (que por defecto es
100), usando `--size n`, donde "n" es un número menor. Por ejemplo:

    predict --size 50

### Actualizaciones y Desarrollo

Si se realizan cambios en la implementación interna de MAIIA (o se actualiza el
código desde el repositorio de git), es necesario construir la imagen de Docker
nuevamente, ejecutando el comando ``` docker compose build```, como fue descripto
en la sección de *Instalación*.

## Contribuciones

Reportes de bugs y *pull requests* pueden ser reportados en la [página de
issues](https://github.com/dymaxionlabs/maiia) de este repositorio. Este
proyecto está destinado a ser un espacio seguro y acogedor para la colaboración,
y se espera que los contribuyentes se adhieran al código de conducta
[Contributor Covenant](http://contributor-covenant.org).

## Licencia

El código está licenciado bajo Apache 2.0. Refiérase a
[LICENSE.txt](LICENSE.txt).
