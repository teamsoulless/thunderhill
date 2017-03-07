## PolySync installation

1. [Requirements](http://docs.polysync.io/articles/overview/installation/basic-system-requirements/)
2. [Download](http://docs.polysync.io/downloads/)
3. [Install](http://docs.polysync.io/flows/getting-started/)
4. Activate license (***only do this once or you will use multiple licenses!***): <br/>
 `polysync-license-tool -a b5b3-eb9e-6f3c-4ff9-b0e5-dd22-d90a-f308`
5. Create a symlink to the libcanlib.so <br/>
`sudo ln –s /usr/local/polysync/vendor/lib/libcanlib.so /usr/local/polysync/vendor/lib/libcanlib.so.1`


## Using sample data

1. Download [PolySync dataset](https://www.dropbox.com/s/exjh3y0d9q4t5a3/polysync-self-racing-cars-2016-dataset.tar.gz?dl=0)
2. Download [psync.sdf file](https://drive.google.com/open?id=0B0ZFAnBlExjMSXVGa0xJdTNXaGc)
3. Go to downloaded dataset folder (_polysync-self-racing-cars-data/1464470620356308_) and replace existing psync.sdf file with the one from step 2
4. Go to _~/.local/share/polysync/config_ folder and replace existing psync.sdf file with the one from step 2
5. Start PolySync Manager: <br/>
`sudo service polysync-core-manager start`
6. Open PolySync Configurator: <br/>
`polysync-core-sdf-configurator` <br/>
Make sure that you can see a configuration for PolySync Kia Soul, close the configurator
7. Open PolySync Studio: <br/>
`polysync-core-studio`
8. [Import session](1464470620356308_) from _1464470620356308_ folder and replay

## Troubleshoot sample data
Sample data may not replay immediately

1. Make sure that there is enough disk space to import sample data
2. "The system is in inconsistent state" error:
  - if appears immediately after opening studio, press Standby to resolve
  - if appears after an attempt to play the recorded session - no action needed, the error may be present, but the data will replay after some time
  - may have to close the studio, stop the manager, restart the manager and reopen the studio to get the session to run
