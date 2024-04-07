import numpy as np
import torch
from captum.attr import IntegratedGradients, Saliency


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

