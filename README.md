**I**ntelligent **T**est **S**elector **for** **S**elf-**D**riving **C**ars **(ITS4SDC)** 

The dataset of the test cases can be found using the following link:
https://zenodo.org/records/14599223
The dataset contains 10000 test cases. A test case contains [(x,y), (x,y), ..., (x,y)] road coordinates as a list and interpolated road coordinates to create texture as an asphalt plane in the simulator.
BeamNG.tech v24 is used as a simulation environment to label data. As a bridge between road coordinates and BeamNG.tech, SDC-Scissor is used.

To run machine-learning test case predictors, use SDC-Scissor with the following link:

https://sdc-scissor.readthedocs.io/en/latest/user_documentation/machine_learning.html

**main.py** file is the ITS4SDC competition file. It uses the **.onnx** model to predict new unseen test cases.

**model_full_10000.h5** is the trained model. The model was trained using 10000 test cases. 

**its4sdc.onnx** is the trained model. The model was trained using 10000 test cases. Due to the heaviness of the model with the *.h5* file, the *.onnx* format is used to speed up the process.

**extract_road_characteristics.py** extracts segment angles and segment lengths into a JSON file.

**LSTM_train_and_test.py** If you want to train and evaluate the model, use this script. It uses k-fold cross-validation. 

