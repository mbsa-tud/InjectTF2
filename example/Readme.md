# InjectTF2 example


This example illustrates the basic usage of the fault injection framework. A simple neural network is trained to classify digits using the MNIST dataset. Afterward a fault injection experiment is conducted.

### Usage:

> Consider using `docker` for an easier setup.<br/>[This](https://hub.docker.com/r/nvaitc/ai-lab) image is quite large, but can be considered as an all-in-one development environment.  

Using the docker image mentioned above, execute `run_docker.sh` in this directory to open a shell inside the docker container. You should now be in the root directory of the repository.

Now `cd` into `/example`. `ls` should list the following files:
```
example_config_2.yml
example_config.yml
inject_example_random_bit.py
inject_example_specific_bit.py
mnist.py
Readme.md (this file)
run_docker.py
```
Run
```shell
$ python mnist.py
```
to train a simple model on the MNIST data set.

Afterward, a new file called `mnist_model.h5` can be found in the current directory.


The summary of the model is as follows:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
_________________________________________________________________
flatten (Flatten)            (None, 6272)              0
_________________________________________________________________
dense (Dense)                (None, 128)               802944
_________________________________________________________________
dropout (Dropout)            (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
_________________________________________________________________
softmax (Softmax)            (None, 10)                0
=================================================================
Total params: 804,554
Trainable params: 804,554
Non-trainable params: 0
_________________________________________________________________
```

Within the configuration file `example_config.yml` the fault injection framework can be configured.

Its content is as follows:
```yaml
inject_layer:
  layer_name: dense_1
  fault_type: BitFlip
  bit_flip_type: RandomBit
  probability: 1.0
```

Here the layer with name `dense_1` is selected for injection. The fault type is `BitFlip`, which is currently the only supported fault type. However, the framework is designed in a modular way so that new fault types can be easily added. The entry `bit_flip_type` accepts the values `RandomBit` and `SpecificBit`, which correspond to the injection of a random or a specific bit of a random element respectively. In the latter case the bit position that should be injected has to be specified. It is zero indexed (i.e. bit position 31 is a sign flip), refer to the configuration file `example_config_2.yml` for an example. The entry `probability` specifies the probability for fault injection in `[0.0, 1.0]`.

Run
```shell
$ python inject_example_random_bit.py
```
to start the fault injection experiment using the configuration file `example_config.yml`.

The experiment will print the classification accuracy of the network on the MNIST test subset with fault injection. You should see a decrease in performance compared to the classification accuracy without faults.

To start the fault injection experiment for a specific bit position run
```shell
$ python inject_example_specific_bit.py
```
