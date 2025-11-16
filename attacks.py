from settings import *
from utils import *
from torch.utils.data import DataLoader
 
class Attacker:
    def __init__(self, attack_type: str, inner_product_scaling: float = 10.0):
        self.attack_type = attack_type
        self.inner_product_scaling = inner_product_scaling
        self.unified_poisoned_params = None
        self.transform = get_feature_transform()

    def perform_model_attack(self, clients: List['Client'], previous_global_model: Dict[str, torch.Tensor] = None, global_model: Dict[str, torch.Tensor] = None, malicious_clients: Optional[List[int]] = None):
        if "inner_product" in self.attack_type:
            self.inner_product_attack(clients, previous_global_model, global_model)
        if "minmax" in self.attack_type:
            self.minmax_attack(clients, previous_global_model, malicious_clients)

    def minmax_attack(self, clients: List['Client'], previous_global_model: Dict[str, torch.Tensor], malicious_clients: Optional[List[int]]):
        malicious_params = [client.get_parameters() for client in clients if client.client_id in malicious_clients]
        if not malicious_params:
            return
        poisoned_params = attkMinMax(previous_global_model, malicious_params, len(malicious_clients))
        self.unified_poisoned_params = poisoned_params[0]
        for client in clients:
            if client.is_malicious:
                client.model.load_state_dict(self.unified_poisoned_params)

    def inner_product_attack(self, clients: List['Client'], previous_global_model: Dict[str, torch.Tensor], global_model: Dict[str, torch.Tensor]):
        for client in clients:
            if client.is_malicious:
                client_params = client.get_parameters()
                previous_global_params = previous_global_model
                global_params = deepcopy(global_model)

                update_direction = {key: client_params[key] - previous_global_params[key] for key in client_params}
                update_vector = torch.cat([v.view(-1) for v in update_direction.values()])

                grad_global = {key: global_params[key] - previous_global_params[key] for key in global_params}
                grad_vector = torch.cat([v.view(-1) for v in grad_global.values()])
                grad_direction = grad_vector.detach().clone()
                grad_norm = torch.norm(grad_direction)
                if grad_norm > 0:
                    grad_direction_normalized = grad_direction / grad_norm
                else:
                    continue
                
                current_inner_product = torch.dot(update_vector, grad_direction_normalized).item()
                target_inner_product = current_inner_product + self.inner_product_scaling

                delta = (target_inner_product - current_inner_product) * grad_direction_normalized
                adjusted_update_vector = update_vector + delta

                adjusted_update_params = {}
                current_position = 0
                for key in update_direction:
                    param_length = update_direction[key].numel()
                    adjusted_update_params[key] = adjusted_update_vector[current_position:current_position + param_length].view(client_params[key].shape).to(device)
                    current_position += param_length

                poisoned_params = {key: previous_global_params[key] + adjusted_update_params[key] for key in client_params}
                client.model.load_state_dict(poisoned_params)


def attkMinMax(previous_global_model: Dict[str, torch.Tensor], local_models: List[Dict[str, torch.Tensor]], num_malicious: int) -> List[Dict[str, torch.Tensor]]:
    if num_malicious <= 1:
        return [deepcopy(previous_global_model)] * num_malicious

    global_params = deepcopy(previous_global_model)
    direction = getDirc(local_models, global_params)
    grads = [getGrad(local_model, global_params) for local_model in local_models]
    avg_grad = avgParas(grads)

    max_dist = max(getDist(g1, g2) for i, g1 in enumerate(grads) for g2 in grads[i+1:])

    gamma = 0.01
    while True:
        new_grad = minusParas(avg_grad, gamma, direction)
        maxdist = max(getDist(new_grad, grad) for grad in grads)
        if maxdist < max_dist or gamma < 1e-6:
            find_grad = new_grad
            break
        gamma *= 0.5

    malicious_update = minusParas(global_params, -1, find_grad)
    return [malicious_update] * num_malicious
 