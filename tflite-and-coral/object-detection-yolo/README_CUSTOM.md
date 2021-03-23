1) fare il training custom con il progetto "darknet" ottenendo:
   tiny-yolo-cfg.cfg 
   darknet-weights.weights
   (seguire le istruzioni per il training custom)

2) convertire i pesi darknet in pesi cerai 
   con il comando del progetto "keras-yolo3"
   python convert.py tiny-yolo-cfg.cfg darknet-weights.weights keras-model.h5

3) convertire i pesi cerai in un modello tflite quantizzato, con il comando:
   python keras_to_tflite_quant.py keras-model.h5 quantized.tflite
   (usando la libreria TF 2.0 nightly: pip install tf-nightly)

4) usare il compilatore corale per convertire il file tflite in file edgetpu
   https://coral.ai/docs/edgetpu/compiler/.
   edgetpu_compiler quantized.tflite

5) usare inference.py per fare inferenza (utilizzando Tensorflow 1.15.0):
   python inference.py --model quantized_edgetpu.tflite 
		      --anchors tiny_yolo_anchors.txt 
     		      --classes coco.names 
		      --quant --edge_tpu --cam



usage: python inference.py [-h] --model MODEL 
			       --anchors ANCHORS 
			       --classes CLASSES
                                [-t THRESHOLD] [--edge_tpu]
                                [--quant] [--cam] [--image IMAGE]
                                [--video VIDEO]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model to load.
  --anchors ANCHORS     Anchors file.
  --classes CLASSES     Classes (.names) file.
  -t THRESHOLD, --threshold THRESHOLD
                        Detection threshold.
  --edge_tpu            Whether to delegate to Edge TPU or run on CPU.
  --quant               Indicates whether the model is quantized.
  --cam                 Run inference on webcam.
  --image IMAGE         Run inference on image.
  --video VIDEO         Run inference on video.