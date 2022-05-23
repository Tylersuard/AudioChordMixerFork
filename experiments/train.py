import os
import sys
import yaml
import logging
import argparse
import torch
from models.utils import get_logger

parser = argparse.ArgumentParser(description="experiments")
parser.add_argument("--problem_class", type=str, default='adding')
parser.add_argument("--problem", type=str, default='200')
parser.add_argument("--model", type=str, default='chordmixer')
parser.add_argument("--logger", type=str, default='file')

args = parser.parse_args()

print(args)
print(args.problem_class)

stream = open("config.yaml", 'r')
cfg_yaml = yaml.load(stream)[args.problem_class][args.problem]
training_config = cfg_yaml['training']
model_config = cfg_yaml['models'][args.model]

print('training config:', training_config)
print('model config:', model_config)

loginf = get_logger(args, training_config, model_config)
        
loginf('training_config')
loginf(training_config)
loginf('model_config')
loginf(model_config)



