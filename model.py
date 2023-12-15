import torch
import torchvision.models as models

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.base_model = models.resnet34(pretrained=False)
        self.base_model.fc = torch.nn.Linear(self.base_model.fc.in_features, 18)
        self.base_model.load_state_dict(torch.load('./fairface_weights/fairface_weights_resnet34.pt'))
        self.base_model.fc = torch.nn.Linear(self.base_model.fc.in_features, 512)
        
        self.dropout = torch.nn.Dropout(0.1)
        self.dense = torch.nn.Linear(512, 256)
        self.output = torch.nn.Linear(256, 1)  
        self.Lrelu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.Lrelu(self.dense(x))
        x = self.output(x)

        return x