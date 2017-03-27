#Main function that brings all data sets and models together
#Import all modules
import config
from generators_funcs import generators_funcs
from models import models
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import pandas as pd

class rambo_main:
    
    """
    
    Author: Kiarie Ndegwa
    Team Udacity - Self Racing League 1-2nd April 2017
    Date: 24/3/2017
    
    """   
    def main(self):
        #Import all configurations constants 
        
        ROWS = config.ROWS
        COLS = config.COLS
        CH = config.CH
        
        PATH_FILE = config.PATH_FILE
        BATCH_SIZE = config.BATCH_SIZE
        EPOCHS = config.EPOCHS
        LOG = config.LOG
       
        DATA = pd.read_csv(config.PATH_FILE+config.LOG)
        
        #get generators
        gen = generators_funcs()
        #import models
        model = models("Rambo")
        model = model.model
        
        #Split data into train and test sets, that is 90% training 10% validation
        train, test = train_test_split(DATA, test_size = 0.10)
        n_train_samples = len(train)
        n_val_samples = len(test)
        
        gen_train = gen.train_generator(train, BATCH_SIZE)
        gen_val = gen.val_generator(test, BATCH_SIZE)

        model.fit_generator(generator = gen_train,
                            samples_per_epoch = n_train_samples,
                            validation_data = gen_val,
                            nb_val_samples = n_val_samples,
                            nb_epoch = EPOCHS,
                                    verbose = 1)   
        
        filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        #Save final model in formats as specified in Udacity ruberic
        model_json = model.to_json()
        
        with open("rambo_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save('rambo_model.h5') 
        print("Saved model to disk")
            
if __name__ == "__main__": 
    rambo = rambo_main()
    rambo.main()