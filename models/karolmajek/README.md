# Info about models

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

Set point 50 MPH:

[![50 mph](https://img.youtube.com/vi/y-UKbBN6RX8/0.jpg)](https://www.youtube.com/watch?v=y-UKbBN6RX8)

Set point 70 MPH:

[![70 mph](https://img.youtube.com/vi/Ap2kFm1Kis8/0.jpg)](https://www.youtube.com/watch?v=Ap2kFm1Kis8)
