import numpy as np
from settings import *
from utils import *
from attacks import Attacker
from defenses import Defense

class Client:
    def __init__(self, client_id: int, dataset_name: str, dataloader: DataLoader, model: nn.Module, learning_rate: float, num_epoch: int, is_malicious: bool = False):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.dataloader = dataloader
        self.is_malicious = is_malicious
        self.model = deepcopy(model).to(device)
        self.representation_model = deepcopy(self.model)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.contrastive_criterion = TripletLoss()
        self.num_epoch = num_epoch
    
    def freeze_statistical_net(self):
        for param in self.model.statistical_net.parameters():
            param.requires_grad = False

    def freeze_task_net(self):
        for param in self.model.task_net.parameters():
            param.requires_grad = False

    def unfreeze_statistical_net(self):
        for param in self.model.statistical_net.parameters():
            param.requires_grad = True

    def unfreeze_task_net(self):
        for param in self.model.task_net.parameters():
            param.requires_grad = True

    def local_train(self, global_model_state: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        self.model.load_state_dict(global_model_state)
        self.representation_model.load_state_dict(global_model_state)
        loss, acc = self.train_model(self.model, self.dataloader, self.criterion)
        return loss, acc

    def train_model(self, model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        # Step 1: Train Task Net
        self.freeze_statistical_net()
        self.unfreeze_task_net()
        optimizer_task = optim.AdamW(
            filter(lambda p: p.requires_grad, model.task_net.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-6
        )
        model.train()
        for epoch in range(self.num_epoch):
            for feat, labels in dataloader:
                feat, labels = feat.to(device), labels.to(device)
                optimizer_task.zero_grad()
                zi = model.statistical_net(feat)
                outputs = model.task_net(zi)
                loss_ce = criterion(outputs, labels)
                loss_ce.backward()
                optimizer_task.step()

        # Step 2: Train Statistical Net
        self.freeze_task_net()
        self.unfreeze_statistical_net()
        optimizer_statistical = optim.AdamW(
            filter(lambda p: p.requires_grad, model.statistical_net.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-6
        )
        model.train()
        for epoch in range(self.num_epoch):
            for feat, labels in dataloader:
                feat, labels = feat.to(device), labels.to(device)
                optimizer_statistical.zero_grad()
                zi = model.statistical_net(feat)
                loss = self.contrastive_criterion(zi, labels)
                loss.backward()
                optimizer_statistical.step()
        return loss.item(), (outputs.argmax(dim=1) == labels).float().mean().item()

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self.model.state_dict())

    def update_lr(self, lr: float):
        self.learning_rate = lr

    def get_representations_per_class(self) -> Dict[int, np.ndarray]:
        num_classes = {"Cifar100": 100, "ImageNet": 200, "Food": 101}
        num_classes = num_classes[self.dataset_name]
        self.representation_model.statistical_net.eval()
        class_reps = defaultdict(list)
        with torch.no_grad():
            for feat, labels in self.dataloader:
                feat = feat.to(device)
                reps = self.representation_model.statistical_net(feat)
                labels = labels.cpu().numpy()
                for rep, label in zip(reps, labels):
                    class_reps[label].append(rep.cpu().numpy())

        sample_counts = [len(reps) for reps in class_reps.values() if len(reps) > 0]
        K = int(np.median(sample_counts)) if sample_counts else 1
        K = max(K, 1)

        balanced_class_reps = {}
        for c in range(num_classes):
            reps = class_reps.get(c, [])
            if len(reps) >= K:
                selected_reps = random.sample(reps, K)
            elif len(reps) > 0:
                selected_reps = [random.choice(reps) for _ in range(K)]
            else:
                feature_dim = self.representation_model.statistical_net.features[-1].out_features
                selected_reps = [np.zeros(feature_dim) for _ in range(K)]
            balanced_class_reps[c] = np.mean(selected_reps, axis=0)
        return balanced_class_reps

class Server:
    def __init__(self, dataset_name: str, model: nn.Module, num_clients: int, malicious_clients: Set[int], learning_rate: float = 0.01):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.malicious_clients = malicious_clients
        self.global_model = deepcopy(model).to(device)
        self.global_model.train()
        self.previous_global_model = deepcopy(self.global_model.state_dict())
        self.criterion = nn.CrossEntropyLoss()
        self.mu = 0.1
        self.learning_rate = learning_rate
        self.lr_decay = 0.98

    def get_lr(self) -> float:
        return self.learning_rate

    def update_params(self, params: Dict[str, torch.Tensor]):
        self.global_model.load_state_dict(params)
        self.previous_global_model = deepcopy(params)
 
    def detect_malicious_clients(self, clients: List[Client], malicious_ids: List[int]) -> Set[int]:
        num_classes = {"Cifar100": 100, "ImageNet": 200, "Food": 101}
        get_class = num_classes[self.dataset_name]

        client_class_reps = {client.client_id: client.get_representations_per_class() for client in clients}

        num_clients = len(clients)
        client_ids = [client.client_id for client in clients]
        vote_counts = Counter()
        detect_iteration = 0
        self.fedrace_iterations = int(num_clients / 2)

        cutoffs = []
        delta_benign = []
        delta_malicious = []
        thresholds = []
        optimal_thresholds = []

        while detect_iteration < self.fedrace_iterations:
            selected_client_ids = random.sample(client_ids, k=int(num_clients * 0.5))
            aggregated_class_reps = {}
            for c in range(get_class):
                reps_c = [client_class_reps[cid][c] for cid in selected_client_ids if client_class_reps[cid][c] is not None]
                if len(reps_c) > 2:
                    aggregated_class_reps[c] = np.median(reps_c, axis=0)
                else:
                    aggregated_class_reps[c] = None
            valid_classes = [c for c, rep in aggregated_class_reps.items() if rep is not None]

            if not valid_classes:
                detect_iteration += 1
                continue

            delta_i_j_max = {}
            for client in clients:
                client_id = client.client_id
                collects_j = []
                client_task_net = client.model.task_net.to(device)
                client_task_net.eval()

                for c in valid_classes:
                    r_c = aggregated_class_reps[c]
                    r_c_tensor = torch.tensor(r_c, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        y_hat = client_task_net(r_c_tensor)
                        probs = torch.softmax(y_hat, dim=1).cpu().numpy()[0]
                    true_label = c
                    loss_ce = -np.log(np.clip(probs[true_label], 1e-5, 1.0))
                    delta_ijc = np.sqrt(2 * loss_ce)
                    collects_j.append(delta_ijc)

                if collects_j:
                    delta_i_j_max[client_id] = calculate_entropy(collects_j)
                else:
                    delta_i_j_max[client_id] = float('inf')

            sorted_clients = sorted(delta_i_j_max.items(), key=lambda x: x[1])
            sorted_residuals = [res for _, res in sorted_clients]
            optimal_threshold = find_optimal_threshold(sorted_residuals)

            final_detection = []
            for cid, value in sorted_clients:
                if value > optimal_threshold:
                    final_detection.append(cid)

            for cid in final_detection:
                vote_counts[cid] += 1

            detect_iteration += 1

        adaptive_threshold = self.fedrace_iterations // 2
        detected_malicious = {cid for cid, count in vote_counts.items() if count >= adaptive_threshold}
        
        return detected_malicious

    def aggregation(self, detected_malicious: Set[int], client_params: List[Dict[str, torch.Tensor]], clients: List[Client]) -> Dict[str, torch.Tensor]:
        benign_params = [params for i, params in enumerate(client_params) if clients[i].client_id not in detected_malicious]
        if not benign_params:
            return self.global_model.state_dict()
        new_global_state = avgParas(benign_params)
        self.global_model.load_state_dict(new_global_state)
        self.previous_global_model = deepcopy(new_global_state)
        return new_global_state

    def evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for feat, labels in dataloader:
                feat, labels = feat.to(device), labels.to(device)
                outputs = model.task_net(model.statistical_net(feat))
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * feat.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
  