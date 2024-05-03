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

### [English] **What is it?**

MAIIA is an algorithm that allows the automated mapping of the 
location of informal urban settlements through 
the analysis of satellite images.
To facilitate its implementation, it is distributed via a pre-configured 
operating system image (via [Docker] (https://www.docker.com/)), together 
with scripts that allow training a detection model 
and applying it to new images in only two steps. 

### [Portugues] **¿Qué es?**

MAIIA é um algoritmo que permite o mapeamento automatizado da 
localização de aglomerados urbanos informais através da análise de imagens de satélite.
Para facilitar a sua implementação, é distribuído através de uma imagem 
pré-configurada do sistema operativo (via [Docker]()), juntamente com scripts que permitem treinar um modelo de detecção e 
aplicá-lo a novas imagens em apenas duas etapas.  


**¿Por qué la estamos compartiendo al público?**

El algoritmo fue desarrollado con énfasis en su facilidad de implementación,
para bajar las barreras de acceso a herramientas de inteligencia artificial (IA)
y permitir que agencias de gobierno, investigadores y otros actores interesados
puedan aplicarlo a sus propios casos de uso.  

[English] **¿Por qué la estamos compartiendo al público?**  

The algorithm was developed with an emphasis on ease of implementation, 
to lower the barriers of access to artificial intelligence (AI) tools and allow 
government agencies, researchers and other stakeholders to apply it to their own use cases.     

[Portugues] **Por que estamos compartilhando isso com o público?**  

O algoritmo foi desenvolvido com ênfase na facilidade de implementação,
para diminuir as barreiras de acesso às ferramentas de inteligência artificial (IA) 
e permitir que agências governamentais, pesquisadores e outros partes interessadas 
a apliquem em seus próprios casos de uso.

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

### [English] **How to use it?**

The basic requirements for using MAIIA are:

- Linux operating system (on a local instance or in the cloud)
- Dedicated hardware for graphics acceleration (NVIDIA GPU)
- Docker installed and configured to use the GPU.

The installation and setup process is described below. It is possible to use MAIIA 
on your own laptop or desktop computer that meets the requirements. Even so, it is 
suggested to perform the installation and setup on a cloud instance, for the benefit 
of having a standardized system.    

### [Portugues] **Como é utilizado?**

Os requisitos básicos para o uso do MAIIA são:

- Sistema operacional Linux (no local ou na nuvem)
- Hardware dedicado para aceleração gráfica (GPU NVIDIA)
- Docker instalado e configurado para usar a GPU

O processo de instalação e configuração é descrito abaixo. É possível usar MAIIA em seu 
próprio laptop ou computador desktop que atenda aos requisitos. compatível com laptop ou 
computador desktop. Entretanto, sugere-se realizar a instalação e a partida em uma 
instância na nuvem. na nuvem, para o benefício de ter um sistema padronizado. sistema padronizado.

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


### [English] **Installation**

1. Clone or download the content of this repository.

2. Install
   [Docker](https://docs.docker.com/engine/install/ubuntu/#installation-methods)
   and
   [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Clarification on *Docker*: when installing docker, you need to follow the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/).

Clarification on *nvidia-docker*: as the prerequisites indicate, you need to have NVIDIA driver installed. You can download [the
installer](https://www.nvidia.com/Download/index.aspx?lang=en-us) or follow this [guide](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) for installation via package.

3.  To ensure that *docker* containers can access the GPU, edit or create the `/etc/docker/daemon.json` file to look like this:

        {
            "default-runtime": "nvidia",
            "runtimes": {
                "nvidia": {
                    "path": "nvidia-container-runtime",
                    "runtimeArgs": []
                }
            }
        }


    After editing the file, restart the *docker* service for the change to take effect:

        $ sudo systemctl restart docker.service
        
4. Start the docker image with the code and dependencies. From the directory where MAIIA has been downloaded, run:

        docker compose build

    Installation of *docker-compose* **version 1.28 or higher** may be required when attempting to run this command. To install the most recent version available, refer to the detailed instructions [here](https://docs.docker.com/compose/install/):

    The process will take quite a while the first time, as you will need to download and configure several software components. Once the first start-up is complete, subsequent start-ups will be almost instantaneous.


### [Portugues] **Instalação**

(estas etapas são necessárias apenas uma vez)

1. Clonar ou baixar o conteúdo deste repositório.

2. Instalar
   [Docker](https://docs.docker.com/engine/install/ubuntu/#installation-methods)
   e
   [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Nota sobre *Docker*: Ao instalar o Docker, é necessário seguir o [pós-instalação](https://docs.docker.com/engine/install/linux-postinstall/).

Nota sobre *nvidia-docker*: como indicado nos pré-requisitos, você precisa ter Driver NVIDIA instalado. Você pode baixar [o
instalador](https://www.nvidia.com/Download/index.aspx?lang=en-us) ou siga este [guia](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) para instalação via pacote.

3.  Para garantir que os recipientes *docker* possam acessar a GPU, editar ou criar o arquivo `/etc/docker/daemon.json` para ter este aspecto:

        {
            "default-runtime": "nvidia",
            "runtimes": {
                "nvidia": {
                    "path": "nvidia-container-runtime",
                    "runtimeArgs": []
                }
            }
        }


    Depois de editar o arquivo, reinicie o serviço *docker* para que a mudança tenha efeito:

        $ sudo systemctl restart docker.service
        
4. Iniciar a imagem do estivador com o código e as dependências. A partir do diretório onde o MAIIA foi baixado, executado:

        docker compose build

    Você pode ser solicitado a instalar *docker-compose* **versão 1.28 ou superior*** ao tentar executar este comando. para executar este comando. Para instalar a última versão disponível, consulte as instruções detalhadas [aqui](https://docs.docker.com/compose/install/):

    O processo levará algum tempo na primeira vez, pois você precisará baixar e configurar vários componentes de software. Uma vez que o primeiro lançamento esteja completo, os lançamentos subseqüentes serão quase instantâneos.


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


### [English] **Configuration**

These steps must be followed each time the system is used.  
Start the Docker container, specifying the user and group corresponding to the host.

    docker compose run -u $(id -u):$(id -g) maiia /bin/bash

or run:

    ./start

If all went well, you should see the following:

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

From this terminal, you will execute the training and prediction commands detailed below

### [Portugues] **Configuração**

Estes passos devem ser seguidos cada vez que o sistema for utilizado.  
Iniciar o container Docker, especificando o usuário e o grupo correspondente ao host.

    docker compose run -u $(id -u):$(id -g) maiia /bin/bash

Ou executar:

    ./start

Se tudo correu bem, você deve ver o seguinte:

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

A partir deste terminal, você executará os comandos de treinamento e previsão detalhados abaixo.

### Jupyter Notebook

Si desea utilizar Jupyter Notebooks en vez de trabajar por la terminal
solamente, ejecute el siguiente comando.  Por defecto el contenedor iniciará
una instancia de Jupyter Notebooks.

    docker compose run --service-ports -u $(id -u):$(id -g) maiia

o simplemente ejecutar:

    ./start_jupyter

**Importante**: Como la arquitectura consume la memoria de la GPU mientras está cargada, solo es posible ejecutar un notebook por kernel. En caso de ejecutar el notebook de predicción luego del notebook de entrenamiento, es necesario apagar el kernel del notebook de entrenamiento luego de ejecutarlo (menú de Running, click en Shutdown).  


[English] If you want to use Jupyter Notebooks instead of working through the terminal only, run the following command.  By default the container will start an instance of Jupyter Notebooks.

    docker compose run --service-ports -u $(id -u):$(id -g) maiia

or simply run:

    ./start_jupyter

**Important**: As the architecture consumes GPU memory while loaded, it is only possible to run one notebook per kernel. In case of running the prediction notebook after the training notebook, it is necessary to shutdown the kernel of the training notebook after running it (Running menu, click on Shutdown).    

[Portugues] Se você quiser usar os cadernos Jupyter em vez de trabalhar somente através do terminal, execute o seguinte comando.  Por padrão, o recipiente iniciará uma instância de Jupyter Notebooks.

    docker compose run --service-ports -u $(id -u):$(id -g) maiia

ou simplesmente executar:

    ./start_jupyter

**Importante**: Como a arquitetura consome memória GPU enquanto carregada, só é possível rodar um notebook por kernel. No caso de executar o caderno de previsão após o caderno de treinamento, é necessário desligar o kernel do caderno de treinamento após executá-lo (menu Running, clique em Shutdown).  

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

### [English] Use

#### Training

1. Provide training inputs:

    - Place in the `data/train/images/` folder one or more files in GeoTIFF (.tif) format of high resolution satellite images of an area with previously identified informal areas.

    - Place in the `data/train/labels/` folder a geo-referenced vector file in *GeoPackage* format with manually identified masks or boundaries of informal settlements in the area covered by the satellite image.

      **Important**: The *features* of the vector file must contain a `class` column of type *string*, with the value `A`.

    - (Optional) Place in the folder `data/train/areas/` a georeferenced vector file in *GeoPackage* format with polygons of areas of interest (AOIs) where you want to perform the training.

      In order to practice, and as an example, these folders already contain a raster file and a mask file (*GeoPackage* with informal areas), which allow immediate testing of the system.

2. Start the training algorithm. From the connection to the docker container, run:

        train

This will initiate the training of an informal area recognition model using the inputs provided.  Upon completion, a file in HDF5 (.h5) format of the trained model will be created inside the `data/models` folder.

It is possible to modify the training parameters with the `train` command. To see all options, run `train --help`. They are shown below:


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

For example, if you want to train for 100 epochs, with a *batch size* of 4, run:

    train --epochs 100 --batch-size 4


#### Prediction / Inference


1. Provide detection input:

    - Leave in the folder `data/predict/images/` one or more files in GeoTIFF (.tif) format with high resolution satellite images of the area where to identify informal areas.

      In this folder there is already a raster file (the same used for training) just to have an example to test the model trained in the previous step.

    - (Optional) Place in the folder `data/predict/areas/` a georeferenced vector file in *GeoPackage* format with polygons of areas of interest (AOIs) where the prediction is to be made.

2. Start the detection algorithm. From the connection to the docker image, run:

        predict

    This will start the recognition of informal areas in the provided image, using the last trained model. When the process is completed, a georeferenced raster file with the boundaries of the informal areas detected in the image will be available in the `data/results` folder.

   It is possible to modify the training parameters with the `predict` command. To see all options, run `predict --help`. They are shown below:


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

For example, if you want to predict and filter by a threshold of 0.3 and minimum area of 1000m run:

    predict --threshold 0.3 --min-area 1000


### [Portugues] Use

#### Treinamento

1. Fornecer insumos de treinamento:

    - Colocar na pasta `data/train/images/`um ou mais arquivos no formato GeoTIFF (.tif) de imagens de satélite de alta resolução de uma área com áreas informais previamente identificadas.

    - Colocar na pasta `data/train/labels/` um arquivo vetorial geo-referenciado em formato *GeoPackage* com máscaras identificadas manualmente ou limites de assentamentos informais na área coberta pela imagem de satélite.

      **Importante**: As *características* do arquivo vetorial devem conter uma coluna `classe` do tipo *tring*, com o valor `A`.

    - (Opcional) Coloque na pasta `data/train/areas/` um arquivo vetorial georreferenciado em formato *GeoPackage* com polígonos de áreas de interesse (AOIs) onde você deseja realizar o treinamento.

    A fim de praticar, e como exemplo, um arquivo raster e um arquivo de máscara (*GeoPackage* com áreas informais) já estão nessas pastas, o que permite testar imediatamente o sistema.

2. Iniciar o algoritmo de treinamento. Da conexão ao contêiner portuário, correr:

        train

   Isto iniciará o treinamento de um modelo de reconhecimento de área informal utilizando os insumos fornecidos.  Ao final, um arquivo em formato HDF5 (.h5) do modelo treinado será criado dentro da pasta `data/models`.

   É possível modificar os parâmetros de treinamento com o comando "treinamento". Para ver todas as opções, execute `train --help'. Eles são mostrados abaixo:


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

Por exemplo, se você quiser treinar para 100 épocas, com um *batch tamanho* de 4, correr:

    train --epochs 100 --batch-size 4


#### Predição / Inferência

1. Fornecer dados de detecção:

    - Deixe na pasta `data/predict/images/`um ou mais arquivos no formato GeoTIFF (.tif) com imagens de satélite de alta resolução da área onde identificar áreas informais.

      Nesta pasta já existe um arquivo raster (o mesmo utilizado para treinamento) apenas para ter um exemplo para testar o modelo treinado na etapa anterior.

    - (Opcional) Colocar na pasta `data/predict/areas/` um arquivo vetorial georreferenciado em formato *GeoPackage* com polígonos de áreas de interesse (AOIs) onde a previsão deve ser feita.

2. Iniciar o algoritmo de detecção. Da conexão para a imagem do estivador, correr:

        predict

    Isto iniciará o reconhecimento de áreas informais na imagem fornecida, utilizando o último modelo treinado. Quando o processo for concluído, um arquivo raster georreferenciado com os limites das áreas informais detectadas na imagem estará disponível na pasta 'data/results'.

   É possível modificar os parâmetros de treinamento com o comando `predict`. Para ver todas as opções, execute `predict --help'. Estes são mostrados abaixo:


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


Por exemplo, se você quiser prever e filtrar por um limiar de 0,3 e uma área mínima de 1000m de execução:

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


#### [English] Settings

In case any of the processes fails due to lack of resources (usually due to insufficient GPU RAM), certain parameters can be used to reduce the processing load, at the cost of a possible reduction in the quality of the results.

For training, you can reduce the "batch size" parameter (which by default is 16), using `--batch-size n`, where "n" is a smaller number. For example:

    train --batch-size 2

Al inferir, puede reducirse el parámetro "window size" (que por defecto es 100), usando `--size n`, donde "n" es un número menor. Por ejemplo:

    predict --size 50


#### [Portugues] Ajustes

No caso de algum dos processos falhar por falta de recursos (geralmente devido à insuficiência de GPU RAM), certos parâmetros podem ser usados para reduzir a carga de processamento, ao custo de uma possível redução na qualidade dos resultados.

Para treinamento, você pode reduzir o parâmetro de tamanho do lote (que por padrão é 16), utilizando `--batch-size n`, onde "n" é um número menor. Por exemplo:

    train --batch-size 2

Ao inferir, você pode reduzir o parâmetro "tamanho da janela" (que por padrão é 100), utilizando `- tamanho n`, onde "n" é um número menor. Por exemplo:

    predict --size 50

### Actualizaciones y Desarrollo

Si se realizan cambios en la implementación interna de MAIIA (o se actualiza el
código desde el repositorio de git), es necesario construir la imagen de Docker
nuevamente, ejecutando el comando ``` docker compose build```, como fue descripto
en la sección de *Instalación*.

#### [English] Upgrades and Development

If changes are made to the internal MAIIA implementation (or code is updated from the git repository), it is necessary to build the Docker image again, by running the ```` docker compose build``` command, as described in the *Installation* section.

#### [Portugues] Atualizações e Desenvolvimento

Se você fizer alterações na implementação interna do MAIIA (ou atualizar o código do repositório git), você precisa construir a imagem do Docker novamente, executando o comando ```docker compose build'', como descrito na seção *Instalação*.

## Contribuciones

Reportes de bugs y *pull requests* pueden ser reportados en la [página de
issues](https://github.com/el-bid/maiia) de este repositorio. Este
proyecto está destinado a ser un espacio seguro y acogedor para la colaboración,
y se espera que los contribuyentes se adhieran al código de conducta
[Contributor Covenant](http://contributor-covenant.org).


### [English] Contributions

Bug reports and *pull requests* can be reported on the [issues page](https://github.com/el-bid/maiia) of this repository. This project is intended to be a safe and welcoming space for collaboration, and contributors are expected to adhere to the code of conduct [Contributor Covenant](http://contributor-covenant.org).

### [Portugues] Contribuições

Relatórios de erros e *pull requests* podem ser relatados na [página de edições](https://github.com/el-bid/maiia) deste repositório. Este projeto pretende ser um espaço seguro e acolhedor para colaboração, e espera-se que os colaboradores adiram ao código de conduta [Pacto de Contribuintes](http://contributor-covenant.org).


## Licencia

El código está licenciado bajo Apache 2.0. Refiérase a
[LICENSE.txt](LICENSE.txt).

### [English] License

The code is licensed under Apache 2.0. Refer to
[LICENSE.txt](LICENSE.txt).

### [Portugues] Licença

O código é licenciado sob o Apache 2.0. Consulte
[LICENSE.txt](LICENSE.txt).
