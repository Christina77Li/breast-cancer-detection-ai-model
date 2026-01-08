import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=1):
    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(checkpoint_path, device, num_classes=1):
    model = get_model(num_classes)
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected: 
        print("[load_model] Warning: state_dict mismatch")
        if missing:
            print(" Missing keys:", missing)
        if unexpected: 
            print(" Unexpected keys:", unexpected)
    model.to(device)
    return model

