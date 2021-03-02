import os
import torch
import pathlib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")
google_cloud = False

base_data_path = os.path.join(pathlib.Path().absolute(), 'Data')
base_model_path = os.path.join(pathlib.Path().absolute(), 'Models')
base_output_path = os.path.join(pathlib.Path().absolute(), 'Output')

if google_cloud:
    base_data_path = os.path.join(r'/home/amitudedhia/DrugDiscovery/Code', 'Data')
    base_model_path = os.path.join(r'/home/amitudedhia/DrugDiscovery/Code', 'Models')
    base_output_path = os.path.join(r'/home/amitudedhia/DrugDiscovery/Code', 'Output')

max_sequence_length = 45+1
noise_vector_length = 5


def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    return torch.rand(size, noise_vector_length)


def ones_target(size, label_smoothing=False):
    '''
    Tensor containing ones, with shape = size
    '''
    if label_smoothing:
        a = 0.8
        b = 1
        return torch.rand(size) * (b - a) + a
    else:
        return torch.ones(size)


def zeros_target(size, label_smoothing=False):
    '''
    Tensor containing zeros, with shape = size
    '''
    if label_smoothing:
        a = 0
        b = 0.2
        return torch.rand(size) * (b - a) + a
    else:
        return torch.zeros(size)
