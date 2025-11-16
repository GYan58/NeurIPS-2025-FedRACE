# FedRACE: A Hierarchical and Statistical Framework for Robust Federated Learning
©2025. Official PyTorch implementation accompanying our NeurIPS 2025 paper. This code is released strictly for non-commercial research purpose.
For commercial or extended use, please contact the authors for permission.

---

## Prerequisites

- Python 3.7 or higher
- PyTorch
- torchvision

Install via pip:

    pip3 install torch torchvision

(Optional) Using conda:

    conda create -n hstatnet python=3.9
    conda activate hstatnet
    pip3 install torch torchvision

---

## Repository Structure

1. settings.py: Configures the environment, including random seeds and device settings.
2. utils.py: Contains utility functions for loading models, data preprocessing, splitting data among clients, and various mathematical operations used throughout the system.
3. attacks.py: Implements attack strategies that malicious clients can perform to manipulate the global model.
4. defenses.py: Implements defense mechanisms to detect and mitigate malicious client contributions.
5. models.py: Defines the HStat-Net architectures used in the system.
6. simulator.py: Defines the Client and Server classes, managing local training, model parameter handling, and malicious client detection.
7. main.py: The entry point of the simulation. Initializes configurations, sets up clients and server, and orchestrates the federated learning process.

---

## Configuration

All configuration parameters are defined in a dictionary inside main.py. Key parameters include:

1. dataset_name: Name of the dataset (e.g., "Cifar100", "ImageNet", "Food")
2. phi_type: Feature extractor type (e.g., "ResNet", "CLIP")
3. defense_type: Defense mechanism to apply (e.g., "FedRACE", "MultiKrum", "TrimMean")
4. attack_type: Type of attack to perform (e.g., "minmax", "inner_product")
5. num_clients: Total number of clients in the federated learning setup
6. num_malicious: Number of malicious clients controlled by the adversary
7. participant_ratio: Fraction of clients participating in each training round
8. learning_rate: Learning rate for model training
9. dirichlet_alpha: Parameter for data distribution among clients
10. batch_size: Batch size for training
11. num_epoch: Number of local training epochs per client
12. num_rounds: Total number of federated learning rounds

---

## Running the Simulation

1. To execute the algorithms, run the ./main.py file using the following command:
```
   python3 ./main.py
```
2. Adjust the parameters and configurations within the ./main.py file to suit your specific needs.

---

## Citation
If you use the simulator or some results in our paper for a published project, please cite our work by using the following bibtex entry
```
@inproceedings{yan2025fedrace,
  title     = {FedRACE: A Hierarchical and Statistical Framework for Robust Federated Learning},
  author    = {Gang Yan, Sikai Yang and Wan Du},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```
