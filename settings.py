import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models
import clip
import numpy as np
import random
from collections import defaultdict, Counter
from copy import deepcopy
import math
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
from numpy.linalg import pinv
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
from typing import List, Set, Dict, Tuple, Optional

seed = 1234 # 1, 12, 123, 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
