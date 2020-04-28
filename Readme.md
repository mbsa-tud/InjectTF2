# Experiment results

Data for graphics in '.yaml' format can be read by using pyyaml library in python

## Example

> import yaml
> out_file = './class_distributions_vgg16_plot_data.yml'

> with open(out_file, "r") as f:
>    plot_data = yaml.load(f, Loader=yaml.UnsafeLoader)
