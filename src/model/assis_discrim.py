from torchvision.models import alexnet
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
from data_loader import FoodDataset
import torch.nn.functional as F


batch_size = 16
lr = 0.00001
beta_1 = 0.5


if __name__ == '__main__':
  model = alexnet(num_classes=2)

  dataset = FoodDataset(transform=transforms.Compose([ToTensor()]))
  data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0, drop_last=True)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta_1, .999))
  
