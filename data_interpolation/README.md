# Using Velocity to Interpolate Throttle and Braking
  
On our first day at Thunderhill, the first order of business was to collect as much data as we possibly could to train our model. As the first couple of laps of data started to come in, we noticed that the throttle values were not being recorded. They were all zero! At this point, we were trying to have our models output the steering, throttle, and brake values, which is pretty difficult without some ground truth data to train against :) We **really** didn't want to just throw away half of our data for the day, so we decided to attempt to interpolate the throttle using the velocity vectors that were recorded.
  
The notebook in this folder is a walkthrough of the method we used to estimate the throttle values. Note that the provided dataset contains the actual ground-truth values for throttle and breaking so we can judge the accuracy of our method.
  
If you have any questions about the method, feel free to reach out to me at jpthalman@gmail.com. 
  
## Usage

Install the following packages:

- Numpy
- Pandas
- Matplotlib
- Jupyter

Start a notebook in this directory:

```
cd path/to/this/dir
jupyter notebook
```

Open `thunderhill-throttle-brake-interpolation.ipynb` and enjoy!
  
