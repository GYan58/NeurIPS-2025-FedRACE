from utils import *
from settings import *
 
class MultiKrum:
    @staticmethod
    def apply(params: List[Dict[str, torch.Tensor]], frac: float, uids: Optional[List[int]] = None) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        n = len(params)
        num = n - int(frac) - 2
        distances = defaultdict(dict)
        keys = params[0].keys()

        for i in range(n):
            param1 = params[i]
            for j in range(i, n):
                if i == j:
                    distances[i][j] = 0.0
                    continue
                distance_sq = sum(torch.norm(param1[key] - params[j][key]).item() ** 2 for key in keys)
                distance = np.sqrt(distance_sq)
                distances[i][j] = distance
                distances[j][i] = distance

        if num == 1:
            min_score = float('inf')
            selected_id = -1
            for i in range(n):
                dist = sorted(distances[i].values())
                score = sum(dist[:num])
                if score < min_score:
                    min_score = score
                    selected_id = i
            return params[selected_id], [uids[selected_id] if uids else selected_id]
        else:
            scores = {i: sum(sorted(dist.values())[:num]) for i, dist in distances.items()}
            sorted_clients = sorted(scores.items(), key=lambda x: x[1])
            good_params = []
            good_ids = []
            for i in range(num):
                idx = sorted_clients[i][0]
                good_ids.append(uids[idx] if uids else idx)
                good_params.append(params[idx])
            bad_ids = sorted(set(uids) - set(good_ids)) if uids else []
            return avgParas(good_params), bad_ids

class TrimMean:
    @staticmethod
    def apply(params: List[Dict[str, torch.Tensor]], frac: int) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        n = len(params)
        k = min(frac, n // 2 - 1)
        aggregated_params = {}

        for key in params[0].keys():
            all_params = torch.stack([param[key] for param in params])
            sorted_params, _ = torch.sort(all_params, dim=0)
            trimmed_params = sorted_params[k:n - k]
            aggregated_params[key] = torch.mean(trimmed_params, dim=0)
        return aggregated_params, []

class Defense:
    def __init__(self):
        pass

    def apply_defense(self, defense_type: str, **kwargs) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        if defense_type == "MultiKrum":
            return MultiKrum.apply(
                params=kwargs.get("params"),
                frac=kwargs.get("frac"),
                uids=kwargs.get("uids"),
            )
        elif defense_type == "TrimMean":
            return TrimMean.apply(
                params=kwargs.get("params"),
                frac=kwargs.get("frac"),
            )
        else:
            raise ValueError(f"Unknown defense type: {defense_type}")
 