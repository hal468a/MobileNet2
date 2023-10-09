import torch, os
import torch.nn as nn
import torch.functional as F
import torchvision.models as models

CLASSES = os.listdir("./Dataset")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss' : loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch: {epoch+1} train_loss: {result['train_loss']:.4f} val_loss: {result['val_loss']:.4f} val_acc: {result['val_acc']:.4f}")

class myMobileNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()

        self.network = models.mobilenet_v2()
        self.network.classifier[1] = nn.Linear(in_features=1280, out_features=len(CLASSES))
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))