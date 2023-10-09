import torch, os
from model import myMobileNet, CLASSES, DEVICE
from PIL import Image
from torchvision import transforms

brone_and_weights = torch.load("./model/ModileNetV2_all.pth")

model = myMobileNet()
model.load_state_dict(brone_and_weights['state_dict'])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('./Test_Dataset/trash14.jpg')
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(input_batch)

_, preds = torch.max(output, 1)
result = CLASSES[preds[0].item()]
print(CLASSES)
print("Predicted class:", result)