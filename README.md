# A Spatial-temporal Attention Approach for Traffic Prediction

This is a TensorFlow1.0 implementation of APTN.

## Requirements
- python>=3.5
- tensorflow>=1.12.0
- numpy


Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Model Training
```bash
# PeMSD4
python train.py --config_filename=configs/pems04.yaml

# PeMSD8
python train.py --config_filename=configs/pems08.yaml
```


## Model Evaluating
```bash
# PeMSD4
python eval.py --config_filename={saved_model_config_filename}

# PeMSD8
python eval.py --config_filename={saved_model_config_filename}
```