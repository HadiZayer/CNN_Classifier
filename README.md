Hadi Alzayer
ha366@cornell.edu

Dependencies:
The code requires PyTorch and numpy

usage: main files are train.py and predict.py, and both use argparse to get inputs. Use --help to display description of the parameters

models.py:
Includes two model. The first model, basic_cnn, is just a simple cnn with 3 convolutions, maxpooling and batch norm. The second model, is vgg but modified to support MNSIT. VGG is definitly overkill, and slower, but could be useful when every fraction of accuracy matters

dataloaders.py:
Two main functions, the first, load_training_data, loads data in a structure similar to the training data where the label is the folder name. It also samples part of the data for validation based on provided probability. The second, load_test_data, just reads all the .png files within the directory and stores the file name as ID without '.png' extension

train.py:
initalizes model, and load it if applicable, then train with given hyperparameters. Uses crossentropy loss.
Two functions: one for training a neural network and the other to determine accuracy.

predict.py:
loads provided model and runs prediction on given files and outputs csv that includes classification.