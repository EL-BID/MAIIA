# MAIIA

**M**apeo de **A**sentamientos **I**nformales basado en **I**nteligencia **A**rtificial

## **¿Qué es?**

MAIIA es un algoritmo que permite mapear de forma automatizada la ubicación de  asentamientos urbanos informales mediante el análisis de imágenes satelitales. Para facilitar su implementación, se distribuye mediante una imagen de sistema operativo pre-configurada (vía [docker](https://www.docker.com/)), junto a scripts que permiten entrenar un modelo de detección y aplicarlo a imágenes nuevas en sólo dos pasos.

MAIIA fue desarrollado por la empresa de tecnología geoespacial [GIM](https://www.gim.be/en) en colaboración con [la División de Vivienda y Desarrollo Urbano (HUD) del BID](https://www.iadb.org/es/sectores/desarrollo-urbano-y-vivienda/perspectiva-general). El desarrollo se realizó como parte de una asistencia técnica al [Departamento Nacional de Planeación de Colombia,](https://www.dnp.gov.co/DNPN/Paginas/default.aspx) con el fin de proveer a la institución de una herramienta que permita generar y actualizar mapas precisos de la ubicación y extensión de asentamientos informales en ciudades colombianas.

**¿Por qué la estamos compartiendo al público?**

El algoritmo fue desarrollado con énfasis en su facilidad de implementación, para bajar las barreras de acceso a herramientas de inteligencia artificial (IA) y permitir que agencias de gobierno, investigadores y otros actores interesados puedan aplicarlo a sus propios casos de uso.

## **¿Cómo se usa?**

Los requisitos básicos para usar MAIIA son:

-   Sistema operativo Linux (en una instancia local o en la nube) 

-   Hardware dedicado para aceleración gráfica (NVIDIA GPU)

-   Docker instalado y configurado para utilizar la GPU

El proceso de instalación y puesta en marcha se describe a continuación. Es posible utilizar MAIIA en una laptop o computadora de escritorio propia que cumpla con los requisitos. Aún así, se sugiere realizar la instalación y puesta en marcha en una instancia en la nube, por el beneficio de contar con un sistema estandarizado.

### Instalación

(estos pasos son requeridos sólo una vez)

1.  Clonar o descargar el contenido de este repositorio. 

2.  Instalar [docker](https://docs.docker.com/engine/install/ubuntu/#installation-methods) y [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). 

Aclaración sobre *docker*: al instalar docker, es necesario cumplir con los [pasos de post-instalación](https://docs.docker.com/engine/install/linux-postinstall/).  
Aclaración sobre *nvidia-docker*: como indican los pre-requisitos, es necesario tener instalado NVIDIA driver. Se puede descargar [el instalador](https://www.nvidia.com/Download/index.aspx?lang=en-us) o seguir esta [guía](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) para la instalación a través de paquete. 

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

4.  Iniciar la imagen de docker con el código y dependencias. Desde el directorio donde se ha descargado MAIIA, ejecutar:

        $ docker-compose up -d

    Es posible que se requiera la instalación de *docker-compose* al intentar ejecutar este comando. En ese caso, ejecutar: 
        $ sudo apt install docker-compose

    El proceso tomará un buen rato la primera vez, ya que necesitará descargar y configurar varios componentes de software. Una vez completada la primera puesta en marcha, las subsiguientes serán casi instantáneas.

### Configuración

(estos pasos deben seguirse cada vez que se utiliza el sistema)

Habiendo iniciado la instancia de docker con:

    $ docker-compose up -d

1.  Conectar a la instancia:

        $ docker exec -it gim_cv_container /bin/bash

2.  Activar el ambiente de trabajo Python:

        $ conda activate gim_cv_gpu

3.  Instalar el paquete de Python con los scripts de reconocimiento de imagen:

        $ pip install -e .

### Uso

#### Entrenamiento

1.  Proveer insumos de entrenamiento:

    -   Dejar en la carpeta `TRAIN/rasters/` uno o más archivos con imagen satelital de alta resolución de una zona con áreas informales previamente identificadas 

    -   Dejar en la carpeta `TRAIN/masks/` un archivo georreferenciado (en formato *shapefile* o *raster*) con los límites identificados manualmente de asentamientos informales en la zona cubierta por la imagen satelital.

  Para poder practicar y como ejemplo, en estas carpetas ya se encuentran un archivo raster y uno de máscara (*shapefile* con áreas informales), que permiten realizar de inmediato una prueba del sistema.

2.  Iniciar el algoritmo de entrenamiento. Desde la conexión a la imagen de docker, ejecutar:

        $ python bin/train_segmentalist.py -d train_tif -tsr 0.4 -sag -ds -pp -dcbam -ecbam -ot -l dice_coeff_loss -ep 200

   Esto iniciará el entrenamiento de un modelo de reconocimiento de áreas informales utilizando los insumos provistos. Al completarse, los archivos del modelo quedarán almacenados en la carpeta `MODELS`

#### Predicción (detección de áreas informales en imagen nueva)

1.  Proveer insumo de detección:

    -   Dejar en la carpeta `INFER` uno o más archivos con imagen satelital de alta resolución de la zona donde identificar áreas informales

  En esta carpeta ya se encuentra un archivo raster (el mismo utilizado para entrenar) sólo con el fin de disponer de un ejemplo para testear el modelo entrenado en el paso anterior.

2.  Iniciar el algoritmo de detección. Desde la conexión a la imagen de docker, ejecutar:

        $ python bin/run_inference_segmentalist.py -td train_tif -d infer_tif -l dice_coeff_loss

  Esto iniciará el reconocimiento de áreas informales en la imagen provista, utilizando el último modelo entrenado. Al completar el proceso, quedará disponible en la carpeta PREDICTIONS un archivo raster georeferenciado con los límites de las áreas informales detectadas en la imagen.
  
  
  
NOTA: En caso de que alguno de los procesos falle por falta de recursos (en general, por no disponer de suficiente memoria RAM de la GPU) se pueden usar ciertos parámetros que reducen la carga de procesamiento, a costa de una posible reducción de la calidad de resultados.

Para entrenar, puede reducirse el parámetro "batch size" (que por defecto es 4), usando `-bs n`, donde "n" es un número menor. Por ejemplo:

        $ python bin/train_segmentalist.py -bs 2 -d train_tif -tsr 0.4 -sag -ds -pp -dcbam -ecbam -ot -l dice_coeff_loss -ep 200
        
Al inferir, puede reducirse el parámetro "window size" (que por defecto es 1024), usando `-w n`, donde "n" es un número menor. Por ejemplo:

        $ python bin/run_inference_segmentalist.py -w 512 -td train_tif -d infer_tif -l dice_coeff_loss
