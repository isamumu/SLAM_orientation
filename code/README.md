# ECE 276 - PR #1

### How to Run

* make sure that the dataset path relative to the code folder should be "../data/
* make sure that python3 and Autograd are installed on the machine
* to run type "python3 <filename>" to the terminal from the code folder in the terminal

### File Descriptions
* orientation.py: runs the gradient descent algorithm and plots the orientations of the VICON data, estiamted quaternions, and optimized quaternions
* panoramaTest.py: only applies to datasets 1,2,8, and 9. Produces panorama images stitched together using VICON data
* full_pipeline.py: runs gradient descent algorithm, plots orientations of optimized quaternions against estimates, and stitches together panorama using optimized quaternions

#### How to Change datasets 
* For orientation.py: change the dataset number on line 172
* For panoramaTest.py: change the dataset number on line 8
* For full_pipeline.py: change the dataset number on line 172

#### NOTE
* gradient descent will take between 30s to a minute to run
* panorama stitching will take around a minute to run
* orientation.py and full_pipeline.py also produce the loss graphs for reference