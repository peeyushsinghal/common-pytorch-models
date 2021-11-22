import torch
import random
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torchsummary import summary


def get_device():
  '''
  provide cuda (GPU) if available, else CPU
  '''
  cuda = torch.cuda.is_available()
  if cuda == True:
    return torch.device("cuda")
  else:
    return torch.device("cpu")


def seed_all(seed_value : int):
  '''
  set seed for all, this is required for reproducibility and deterministic behaviour
  '''
  random.seed(seed_value)
  np.random.seed(seed_value)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  else:
    torch.manual_seed(seed_value)


def get_mean_std_dev(dataset_name):
  '''
  get mean and std deviation of dataset
  reference : https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
  '''
  if dataset_name == "CIFAR10" :
    dataset = datasets.CIFAR10(
      root = './',# directory where data needs to be stored
      train = True, # get the training portion of the dataset
      download = True, # downloads
      transform = transforms.ToTensor()# converts to tensor
      )
    data = dataset.data / 255 # data is numpy array

    mean = data.mean(axis = (0,1,2)) 
    std = data.std(axis = (0,1,2))
    # print(f"Mean : {mean}   STD: {std}") #Mean : [0.491 0.482 0.446]   STD: [0.247 0.243 0.261]
    return tuple(mean), tuple(std)

  return (0,0,0),(0,0,0)



def model_summary(model, input_size):
    """
    Summary of the model.
    """
    summary(model, input_size=input_size) 