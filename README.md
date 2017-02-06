# TensorFlow Transfer Learning on CIFAR-10
Trains a softmax regression model on CIFAR-10 using CNN pool_3 weights from inception-v3.
Forked from https://github.com/sthomp/tensorflow_transfer_cifar10<br />
Made changes to do training in batches.<br />

This is the code that supplements the original [blog post](https://medium.com/@st553/using-transfer-learning-to-classify-images-with-tensorflow-b0f3142b9366)

## Setup
1. Download and extract [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) dataset to resources/datasets/
2. Download  and extract the [Inception v3](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz) model to resources/

## Using Pretrained Model
1. **First generate and save bottleneck features.**<br />
To generate the bottleneck features set the flag DO_SERIALIZATION in transfer_cifar10_softmax.py to True:<br />
```python
#flag to generate and save bottleneck features
DO_SERIALIZATION = False
```
2. Run script transfer_cifar10_softmax.py
This would run the input images through the trained Inception V3 network and save the output of the pool_3 layer.

3. The previous step would generate .npy files in the project root directory. The files will be
    * X_train.npy
    * X_test.npy
    * Y_train.npy
    * Y_test.npy
   These files store the oputput of layer pool3 of the Inception model and the corresponding labels. Training for the new task will use these as inputs.

## Training for new task
1. Add desired layers by modifying function ```add_final_training_ops()``` in transfer_cifar10_softmax.py.

2. Set flag DO_SERIALIZATION to False.

3. run script transfer_cifar10_softmax.py.

## Note
function ```load_CIFAR10()``` in data_utils.py can be modified to return only 1000 training images and 100 test images from the CIFAR-10 data. This will speed up the
generation of bottlenecks for a quick demo.

## Todo Tasks
- [x] Fix to train in batches
- [ ] Code Cleanup
- [ ] Create methods and classes
- [ ] Fix tsne plotting code

