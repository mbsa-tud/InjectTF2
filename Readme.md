# InjectTF2 - a fault injection framework for TensorFlow 2
----

InjectTF2 is a Python 3 framework for fault injection into TensorFlow models. It is capable of injecting faults into the output of layers of a neural network. Currently the framework supports sequential models and can inject the following faults:

* Random bit flip into a random element of the output tensor of a layer
* Specific bit flip into a random element of the output tensor of a layer

For the injection of faults into models that have been designed with the low level TensorFlow 1 API, please take a look at [InjectTF](https://github.com/mbsa-tud/InjectTF).

----
### Overview
- [Working principle](#working-principle)
- [Usage](#usage)
----
### Working principle

When initializing the framework the model is executed up to the layer where errors are to be injected, using the provided data set. The output values of the injected layer are collected and stored. Afterward, the fault injection experiment can be started from the selected layer onward using the gathered values.

During the execution of the fault injection experiment, errors are injected into the stored values of the selected layer according to the parameters specified in the configuration file.

Splitting the model and executing the two resulting parts separately drastically reduces the execution time of the experiments, since the network is not executed from bottom to top each time.

----

### Usage

Please refer to the example in the `example` folder for details on how to use the framework.

----
