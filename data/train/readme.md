## CARPETA **TRAIN**

Esta carpeta se divide en tres subcarpetas:  

1. IMAGES  

    - En esta carpeta hay que colocar uno más archivos en formato GeoTIFF *(.tif)* con imágenes satelitales de alta resolución de una zona con áreas informales ya identificadas.   

    - A modo de ejemplo, en esta carpeta se encuentra la imagen satelital de Medellín, Colombia (el mismo que se encuentra en la carpeta de predicción).    


2. LABELS  

    - En esta carpeta hay que colocar un archivo vectorial georreferenciado en formato GeoPackage *(.gpkg)* con las máscaras o límites identificados manualmente de asentamientos informales en la zona cubierta por la imagen satelital.   
    **Importante**: Los *features* del archivo vectorial deben contener una columnas `class` de tipo *string*, con el valor `A`


3. AREAS  

    - Opcionalmente, se puede colocar en esta carpeta un archivo vectorial georreferenciado en formato GeoPackage *(.gpkg)* con polígonos de áreas de interés (AOIs) donde realizar el entrenamiento. 



