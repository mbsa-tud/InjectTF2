#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import logging

import tensorflow as tf

from inject_tf2.config_manager import ConfigurationManager
from inject_tf2.model_manager import ModelManager


class InjectionManager:
    def __init__(self):
        logging.debug("initialized InjectionManager")

    def inject(self, output_val, layer_config):
        return output_val  # TODO
