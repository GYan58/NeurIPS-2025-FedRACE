from settings import *

class HStatNet(nn.Module):
    def __init__(self, output_dim: int = 100):
        super(HStatNet, self).__init__()
        
        self.clip_model, _ = clip.load('ViT-B/32', device)

        for param in self.clip_model.parameters():
            param.requires_grad = False P

        self.statistical_net = nn.Sequential(
            nn.Linear(512, 256),
        )
            
        self.task_net = nn.Sequential(
            nn.Linear(256, output_dim)
        )

    def forward(self, image_tensors: torch.Tensor):
        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensors)

        features = self.statistical_net(features)

        return self.task_net(features)
    
    def get_representations(self, image_tensors: torch.Tensor):
        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensors)
            return self.statistical_net(features)

 