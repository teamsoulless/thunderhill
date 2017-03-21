# Info about models

Training is integrated with TensorBoard:
```
tensorboard --logdir=logs/
```

## model_000_NVIDIA_trained_on_sim001_sim002_yt_polysync

Model trained on datasets:
- dataset_polysync_1464466368552019
- dataset_sim_001_km_320x160
- dataset_sim_002_km_320x160_recovery
- dataset_session_5

Validation:
- dataset_sim_001_km_320x160

Loss:

![Loss](model_000_NVIDIA_trained_on_sim001_sim002_yt_polysync/loss.png)

Validation loss:

![Validation loss](model_000_NVIDIA_trained_on_sim001_sim002_yt_polysync/val_loss.png)

Files:
- model.json
- weights.228-0.071.hdf5

### Training

```
python3 model.py --dataset ../../../../thunderhill_data/ --output all
```

### Running

```
python3 drive.py model_000_NVIDIA_trained_on_sim001_sim002_yt_polysync/model.json model_000_NVIDIA_trained_on_sim001_sim002_yt_polysync/weights.228-0.071.hdf5

```

### Results

Set point 50 MPH (YouTube video):

[![50 mph](https://img.youtube.com/vi/y-UKbBN6RX8/0.jpg)](https://www.youtube.com/watch?v=y-UKbBN6RX8)

Set point 70 MPH (YouTube video):

[![70 mph](https://img.youtube.com/vi/Ap2kFm1Kis8/0.jpg)](https://www.youtube.com/watch?v=Ap2kFm1Kis8)





## model_001_NVIDIA_trained_on_sim002_sim003_yt_poly0_poly2

Model trained on datasets:
- dataset_polysync_1464466368552019
- dataset_polysync_1464552951979919
- dataset_sim_002_km_320x160_recovery
- dataset_sim_003_km_320x160
- dataset_session_5

Validation:
- dataset_sim_001_km_320x160

Loss:

![Loss](model_001_NVIDIA_trained_on_sim002_sim003_yt_poly0_poly2/loss.png)

Validation loss:

![Validation loss](model_001_NVIDIA_trained_on_sim002_sim003_yt_poly0_poly2/val_loss.png)

Files:
- model.json
- weights.0076-0.128.hdf5


### Training

```
cd thunderhill/models/karolmajek/model_001_NVIDIA_trained_on_sim002_sim003_yt_poly0_poly2/src
python3 model.py --dataset ../../../../../thunderhill_data/ --output result
```

### Running

```
cd thunderhill/models/karolmajek/model_001_NVIDIA_trained_on_sim002_sim003_yt_poly0_poly2/src
python3 drive.py ../model.json ../weights.0076-0.128.hdf5
```

### Results

Set point 70 MPH (YouTube video):

[![70 mph](https://img.youtube.com/vi/6LA_rW6Der8/0.jpg)](https://www.youtube.com/watch?v=6LA_rW6Der8)
