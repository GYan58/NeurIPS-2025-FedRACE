from settings import *
import models as MLs
 
def load_model(Dname: str) -> nn.Module:
    num_classes = {"Cifar100": 100, "ImageNet": 200, "Food": 101}
    output_dim = num_classes[Dname]
    return MLs.HStatNet(output_dim)

class CustomImageDataset(Dataset):
    def __init__(self, inputs: np.ndarray, labels: np.ndarray, transforms=None):
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, label = self.inputs[index], self.labels[index].item()
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self) -> int:
        return self.inputs.shape[0]

def get_Dataset(Dname: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path_train = f"{Dname}_train_dataset.pth"
    path_test = f"{Dname}_test_dataset.pth"
    TrainX, TrainY = torch.load(path_train)
    TestX, TestY = torch.load(path_test)
    return TrainX, TrainY, TestX, TestY

def check_labels(labels: List) -> np.ndarray:
    if isinstance(labels[0], list):
        new_labels = [label[0] if label else -1 for label in labels]
        return np.array(new_labels)
    return np.array(labels)

def find_labels(labels: List, key: int) -> np.ndarray:
    if isinstance(labels[0], list):
        return np.array([i for i, label in enumerate(labels) if key in label])
    return np.where(labels == key)[0]

def split_data(data: np.ndarray, labels: List, n_clients: int, alpha: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
    original_labels = deepcopy(labels)
    labels = check_labels(labels)
    data = np.array(data)

    client_data_idx = [[] for _ in range(n_clients)]
    unique_labels = np.unique(labels)
    avg_num = len(labels) / n_clients

    for label in unique_labels:
        idx_k = find_labels(labels, label)
        dirichlet_dist = np.random.dirichlet([alpha] * n_clients)
        dirichlet_dist /= np.sum(dirichlet_dist)
        data_counts = (dirichlet_dist * len(idx_k)).astype(int)

        if any(len(client_data_idx[ky]) + data_counts[ky] > 5 * avg_num for ky in range(n_clients)):
            dirichlet_dist = np.ones(n_clients) / n_clients
            data_counts = (dirichlet_dist * len(idx_k)).astype(int)

        data_counts[-1] = len(idx_k) - sum(data_counts[:-1])

        start = 0
        for i, count in enumerate(data_counts):
            client_data_idx[i].extend(idx_k[start:start + count])
            start += count

    clients_split = []
    original_labels = np.array(original_labels)
    TotalNum = 0
    SplitNum = []

    for idxs in client_data_idx:
        unique_idxs = np.unique(idxs)
        if len(unique_idxs) < 20:
            additional = np.random.choice(len(data), 20 - len(unique_idxs), replace=True)
            unique_idxs = np.concatenate([unique_idxs, additional])
        Ls = original_labels[unique_idxs]
        Ds = data[unique_idxs]
        TotalNum += len(unique_idxs)
        SplitNum.append(len(unique_idxs))
        clients_split.append((Ds, Ls))

    return clients_split

def loadData(Dname: str, Clients: int, Dtype: Optional[str] = None, alpha: float = 0.5, batchsize: int = 128) -> Tuple[DataLoader, List[DataLoader], List[DataLoader]]:
    TrainX, TrainY, TestX, TestY = get_Dataset(Dname)
    splits_tr = split_data(TrainX, TrainY, Clients, alpha)
    splits_te = split_data(TestX, TestY, Clients, alpha)

    client_trloaders = [DataLoader(CustomImageDataset(x, y), batch_size=batchsize, shuffle=True, drop_last=True) for x, y in splits_tr]
    client_teloaders = [DataLoader(CustomImageDataset(x, y), batch_size=batchsize, shuffle=False, drop_last=False) for x, y in splits_te]

    test_loader = DataLoader(CustomImageDataset(TestX, TestY), batch_size=256, shuffle=False, drop_last=False)

    return test_loader, client_trloaders, client_teloaders

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.1, noise_std: float = 0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.noise_std = noise_std

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        batch_size = embeddings.size(0)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        pairwise_dist = self.pairwise_distance(embeddings)
        labels = labels.unsqueeze(1)
        mask_positive = torch.eq(labels, labels.T).float()
        mask_negative = 1 - mask_positive
        mask_positive = mask_positive - torch.eye(batch_size, device=device)
        mask_positive = torch.clamp(mask_positive, min=0)

        all_same = torch.all(labels == labels[0])
        all_unique = torch.all(torch.sum(mask_positive, dim=1) == 0)

        if all_same:
            rand_mask = torch.rand(batch_size, batch_size, device=device) < 0.5
            mask_positive = rand_mask.float() * (1 - torch.eye(batch_size, device=device))
        elif all_unique:
            noise = torch.randn_like(embeddings) * self.noise_std
            embeddings = embeddings + noise
            embeddings = F.normalize(embeddings, p=2, dim=1)
            pairwise_dist = self.pairwise_distance(embeddings)
            mask_positive = torch.eye(batch_size, device=device).float()

        anchor_positive_dist = pairwise_dist * mask_positive
        anchor_positive_dist += (1 - mask_positive) * 1e9
        hardest_positive_dist, _ = anchor_positive_dist.min(dim=1)

        anchor_negative_dist = pairwise_dist + (mask_positive) * 1e9
        hardest_negative_dist, _ = anchor_negative_dist.max(dim=1)

        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin).mean()
        return triplet_loss

    @staticmethod
    def pairwise_distance(embeddings: torch.Tensor) -> torch.Tensor:
        dot_product = torch.matmul(embeddings, embeddings.T)
        square_sum = torch.diagonal(dot_product)
        distances = square_sum.unsqueeze(1) - 2 * dot_product + square_sum.unsqueeze(0)
        distances = F.relu(distances)
        distances = torch.sqrt(distances + 1e-8)
        return distances

def avgParas(params_list: List[Dict[str, torch.Tensor]], weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
    if weights is None:
        weights = np.ones(len(params_list))
    sum_weights = np.sum(weights)
    avg_params = {key: torch.zeros_like(param) for key, param in params_list[0].items()}
    for i, params in enumerate(params_list):
        weight_factor = weights[i] / sum_weights
        for key in params:
            avg_params[key] += params[key] * weight_factor
    return avg_params

def getGrad(local_params: Dict[str, torch.Tensor], global_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: local_params[key] - global_params[key] for key in local_params}

def getDist(w0: Dict[str, torch.Tensor], w1: Dict[str, torch.Tensor]) -> float:
    dist = sum(torch.norm(w0[key].cpu() - w1[key].cpu())**2 for key in w0)
    return dist.sqrt().item()

def getSim(w0: Dict[str, torch.Tensor], w1: Dict[str, torch.Tensor]) -> float:
    norm0 = torch.tensor(0.0, device=next(iter(w0.values())).device)
    norm1 = torch.tensor(0.0, device=next(iter(w1.values())).device)
    dots = torch.tensor(0.0, device=next(iter(w0.values())).device)
    for key in w0:
        v0, v1 = w0[key], w1[key]
        norm0 += torch.norm(v0)**2
        norm1 += torch.norm(v1)**2
        dots += torch.sum(v0 * v1)
    sim = dots / (torch.sqrt(norm0 * norm1) + 1e-8)
    return sim.item()

def getDirc(paras: List[Dict[str, torch.Tensor]], re_para: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    grad = getGrad(avgParas(paras), re_para)
    return {key: torch.sign(grad[key]) for key in grad}

def minusParas(param1: Dict[str, torch.Tensor], scale: float, param2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: param1[key] - scale * param2[key] for key in param1}

def calculate_entropy(data_list: List[float], epsilon: float = 1e-5) -> float:
    data_array = np.clip(np.array(data_list, dtype=np.float32), a_min=epsilon, a_max=None)
    entropy = np.sum(data_array * np.log(data_array + epsilon))
    return entropy

def find_optimal_threshold(data: List[float]) -> float:
    data = sorted(data)
    n = len(data)
    optimal_threshold = None
    min_value = float('inf')

    for i in range(1, n-1):
        group_A = data[:i]
        group_B = data[i:]
        mu_A = np.mean(group_A)
        mu_B = np.mean(group_B)
        sigma_A = np.var(group_A, ddof=1)
        sigma_B = np.var(group_B, ddof=1)
        if (mu_A - mu_B) ** 2 > 0:
            value = max(sigma_A, sigma_B) / (mu_A - mu_B) ** 2
            if value < min_value:
                min_value = value
                optimal_threshold = (data[i - 1] + data[i]) / 2

    if optimal_threshold is None:
        optimal_threshold = np.median(data)
    return optimal_threshold
