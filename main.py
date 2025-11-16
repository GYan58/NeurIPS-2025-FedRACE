from settings import *
from utils import *
from simulator import Client, Server
from attacks import Attacker
from defenses import Defense
 
class Processing:
    def __init__(self, configs: Dict):
        self.num_clients = configs["num_clients"]
        self.num_participant = int(configs["num_clients"] * configs["participant_ratio"])
        self.malicious_clients = list(range(configs["num_malicious"]))
        self.train_model = deepcopy(load_model(Dname=configs["dataset_name"])).to(device)

        self.test_loader, self.clients_train_loader, self.clients_test_loader = loadData(
            Dname=configs["dataset_name"],
            Clients=configs["num_clients"],
            Dtype=configs["dataset_type"],
            alpha=configs["dirichlet_alpha"],
            batchsize=configs["batch_size"]
        )

        self.clients = []
        for client_id in range(configs["num_clients"]):
            is_malicious = client_id in self.malicious_clients
            client = Client(
                client_id=client_id,
                dataset_name=configs["dataset_name"],
                dataloader=self.clients_train_loader[client_id],
                model=self.train_model,
                learning_rate=configs["learning_rate"],
                num_epoch=configs["num_epoch"],
                is_malicious=is_malicious
            )
            self.clients.append(client)

        self.server = Server(
            dataset_name=configs["dataset_name"],
            model=self.train_model,
            num_clients=configs["num_clients"],
            malicious_clients=set(self.malicious_clients),
            learning_rate=configs["learning_rate"]
        )

        num_classes = {"Cifar100": 100, "ImageNet": 200, "Food": 101}
        self.get_class = num_classes[configs["dataset_name"]]

        self.attacker = Attacker(
            attack_type=configs["attack_type"],
            inner_product_scaling=10.0
        )

        self.defense_type = configs["defense_type"]
        self.sota_defense = Defense()

    def main(self):
        for rnd in range(1, configs["num_rounds"] + 1):
            print(f"\n=== Federated Learning Round {rnd} ===")

            previous_global_model = deepcopy(self.server.previous_global_model)

            selected_ids = set(random.sample(range(self.num_clients), self.num_participant))
            selected_clients = [client for client in self.clients if client.client_id in selected_ids]

            selected_malicious_clients = [client for client in selected_clients if client.is_malicious]
            selected_malicious_ids = [client.client_id for client in selected_malicious_clients]

            for client in selected_clients:
                client.update_lr(self.server.get_lr())
                loss, acc = client.local_train(previous_global_model)

            available_params = [client.get_parameters() for client in selected_clients if client.is_malicious]
            est_global_params = avgParas(available_params) if available_params else self.server.global_model.state_dict()

            self.attacker.perform_model_attack(
                clients=selected_clients,
                previous_global_model=previous_global_model,
                global_model=est_global_params,
                malicious_clients=selected_malicious_ids
            )

            client_params = [client.get_parameters() for client in selected_clients]
            if self.defense_type == "FedRACE":
                detected_malicious = self.server.detect_malicious_clients(selected_clients, selected_malicious_ids)
                self.server.aggregation(detected_malicious, client_params, selected_clients)
            else:
                aggregated_model, detected_malicious = self.sota_defense.apply_defense(
                    defense_type=self.defense_type,
                    params=client_params,
                    frac=len(selected_malicious_ids),
                    uids=[client.client_id for client in selected_clients]
                )
                self.server.update_params(aggregated_model)

            self.server.learning_rate *= self.server.lr_decay

if __name__ == '__main__':
    configs = {
        "dataset_name": "Cifar100",
        "defense_type": "FedRACE",
        "attack_type": "minmax",
        "num_clients": 64,
        "num_malicious": 16,
        "participant_ratio": 0.25,
        "learning_rate": 0.001,
        "dirichlet_alpha": 0.5,
        "batch_size": 128,
        "num_epoch": 3,
        "num_rounds": 200,
    }

    processor = Processing(configs)
    processor.main()
 