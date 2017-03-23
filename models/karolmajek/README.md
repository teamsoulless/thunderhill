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







## model_005_small

```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 80, 160, 3)    0                                            
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 38, 78, 24)    1824        input_1[0][0]                    
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 38, 78, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 37, 36)    21636       elu_1[0][0]                      
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 17, 37, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 17, 48)     43248       elu_2[0][0]                      
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 7, 17, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 15, 64)     27712       elu_3[0][0]                      
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 5, 15, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 13, 64)     36928       elu_4[0][0]                      
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 3, 13, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2496)          0           elu_5[0][0]                      
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           249700      flatten_1[0][0]                  
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           elu_6[0][0]                      
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]                  
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           elu_7[0][0]                      
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]                  
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          elu_8[0][0]                      
====================================================================================================
Total params: 386,619
```

Model trained on datasets:
- dataset_polysync_1464466368552019
- dataset_polysync_1464552951979919
- dataset_sim_002_km_320x160_recovery
- dataset_sim_003_km_320x160
- dataset_session_5

Validation:
- dataset_sim_001_km_320x160

Loss:

![Loss](model_005_small/loss.png)

Validation loss:

![Validation loss](model_005_small/val_loss.png)

Files:
- model.json
- weights.0803-0.064.hdf5


### Training

```
cd thunderhill/models/karolmajek/model_005_small/src
python3 model.py --dataset ../../../../../thunderhill_data/ --output result
```

### Running

```
cd thunderhill/models/karolmajek/model_005_small/src
python3 drive.py ../model.json ../weights.0803-0.064.hdf5
```

### Results

Set point 70 MPH (YouTube video):

[![70 mph](https://img.youtube.com/vi/_YcGCOoxK2U/0.jpg)](https://www.youtube.com/watch?v=_YcGCOoxK2U)








## model_006_small_2inputs_3outputs

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 80, 160, 3)    0                                            
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 38, 78, 24)    1824                                         
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 38, 78, 24)    0                                            
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 17, 37, 36)    21636                                        
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 17, 37, 36)    0                                            
____________________________________________________________________________________________________
conv2d_3 (Conv2D)                (None, 7, 17, 48)     43248                                        
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 7, 17, 48)     0                                            
____________________________________________________________________________________________________
conv2d_4 (Conv2D)                (None, 5, 15, 64)     27712                                        
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 5, 15, 64)     0                                            
____________________________________________________________________________________________________
conv2d_5 (Conv2D)                (None, 3, 13, 64)     36928                                        


____________________________________________________________________________________________________
speed_input (InputLayer)         (None, 8)             0                                            
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 3, 13, 64)     0                                            
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           900                                          


____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2496)          0                                            
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 100)           0                                           
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 2596)          0                                            


____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 100)           259700                                       
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 100)           0                                            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0                                            
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 50)            5050                                         
____________________________________________________________________________________________________
elu_9 (ELU)                      (None, 50)            0                                            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0                                            
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 10)            510                                          
____________________________________________________________________________________________________
elu_10 (ELU)                     (None, 10)            0                                            
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 3)             33                                           
====================================================================================================
Total params: 397,541.0

```

Model trained on datasets:
- dataset_sim_002_km_320x160_recovery
- dataset_sim_003_km_320x160
- dataset_sim_004_km_320x160_cones_brakes
- dataset_session_5

Validation:
- dataset_sim_001_km_320x160

Loss:

![Loss](model_006_small_2inputs_3outputs/loss.png)

Validation loss:

![Validation loss](model_006_small_2inputs_3outputs/val_loss.png)

Files:
- model.json
- weights.0723-0.137.hdf5


### Training

```
cd thunderhill/models/karolmajek/model_006_small_2inputs_3outputs/src
python3 model.py --dataset ../../../../../thunderhill_data/ --output result
```

### Running

```
cd thunderhill/models/karolmajek/model_006_small_2inputs_3outputs/src
python3 drive.py ../model.json ../weights.0723-0.137.hdf5
```

### Results

Set point 70 MPH (YouTube video):

[![70 mph](https://img.youtube.com/vi/Y36FVlPdfJk/0.jpg)](https://www.youtube.com/watch?v=Y36FVlPdfJk)
