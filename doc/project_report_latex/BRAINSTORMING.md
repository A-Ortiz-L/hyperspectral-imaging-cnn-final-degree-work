##possible mentions

- Rapid earthquake damage detection
- Time is a  critical task such as performing the damage assessment,
providing immediate delivery of relief assistance. 
- Very High Resolution (VHR) Satellite Imagery offers sub-meter resolution
##Model
- To maximize the accuracy, deep convolution neural network (CNN) model is designed especially for the earthquake damage detection using remote sensing data and implemented using high performance GPU without compromising with the execution time.
- GPU K80 High Performance Computing (HPC) platform.
- Five convolution layers each followed by max pooling layer are added. Convolution is initiated by invoking 64 filters of size 3x3 and max pooling by size 3x3.
## Usos
- podemos miminizar los daños y repartir los recursos sabiendo
las zonas que han sido más afectadas y las que necesitan más ayuda en cada
momento.
- el tiempo se convierte en una métrica esencial para poder reaccionar 
de manera correcta. Los requisitos de tiempo se acercan al procesamiento
en tiempo real de los datos.
- Aprovechar la capacidad de los satélites para la visualización de grandes
espacios de terreno.
- Es clave poder consumir los datos que produce el satélite en tiempo real
- poder producir un heatmap del área según el daño en tiempo real

# tech stack
- tensorflow
- flask
- fastapi
- openvino
- tensorflow serve
- docker
- google cloud 
  * cloud function
  * pub/sub
  * storage
  * compute engine
- python3.7
- uvicorn
- gunicorn
- opencv
- docker-compose
- github
- pycharm
- texmaker

