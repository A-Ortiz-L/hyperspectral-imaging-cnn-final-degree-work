La serialización del modelo de keras solo funciona en 1.14.0
keras usa un formato h5. tensorflow pb

h5 -> pb -> bin + xml ( openvino )



comando para convertir openvino : mo.py --input_model ./data/xml_model/6/saved_modelss.pb -b 1


gsutil notification create -t openvino -f json -e OBJECT_FINALIZE gs://tfg-andrew


tensorflow_model_server --rest_api_port=8501 --model_name=model --model_base_path=/app/data/models &
