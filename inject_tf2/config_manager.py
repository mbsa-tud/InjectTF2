#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import logging
import yaml

import inject_tf2.string_res as str_res


class ConfigurationManager:
    def __init__(self, path_to_config_file="./config/InjectTF2_conf.yml"):
        """InjectTF2 configuration class.

        This class handles reading and parsing the yaml configuration file.

        Args:
            path_to_config_file (str): Path to the configuration file. Defaults to
                "./config/InjectTF2_conf.yml".
        """

        self.__config_data = self.__read_config(path_to_config_file)

    def __read_config(self, file):

        logging.debug("Reading configuration file {0}".format(file))

        file = (
            file if (file.endswith(".yml") or file.endswith(".yaml")) else file + ".yml"
        )

        try:
            with open(file, "rb") as f:
                data = yaml.safe_load(f)
        except IOError as error:
            print("Can not open file: ", file)
            raise

        logging.debug("Done reading config file.\nData is:\n{0}".format(data))

        return data

    def get_data(self):
        """Returns a dictionary containing the complete data of the configuration file."""
        return self.__config_data

    def get_selected_layer(self):
        """Returns the name of the selected layer."""
        return self.__config_data[str_res.inject_layer_str][str_res.layer_name_str]
