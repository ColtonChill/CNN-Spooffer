# Colton Hill
from torchvision import models
from spooffer import CNN_Spooffer
from utils import get_params, torch, functional, class_table

# Feel free to change it to another model! Alexnet and resnets work well.
pretrained_model = models.alexnet(pretrained=True)

# Ender your starting & ending class indices (see 'class_table.py')
init_class_idx = 150  # Sea-lion
target_class_idx = 420  # banjo
preprocessed_img, files = get_params(init_class_idx,target_class_idx)

# initial classification
prediction = functional.softmax(pretrained_model(preprocessed_img), dim=1)
top_class = torch.argmax(prediction).data.item()
confidence = prediction[0][top_class].data.item()
initial_str = "\n-----\n"+\
    'Initial classification:\n\tClass:' + class_table[top_class] + '\n'+\
    '\tConfidence:'+ format(confidence,'.3%')

# spoof the image
min_confidence = 0.99
print("Starting Gradient Ascent...")
spooffer = CNN_Spooffer(pretrained_model, preprocessed_img, target_class_idx, min_confidence, files)
spooffed_img = spooffer.gradAscent()

# final classification
prediction = functional.softmax(pretrained_model(spooffed_img), dim=1)
top_class = torch.argmax(prediction).data.item()
confidence = prediction[0][top_class].data.item()
finial_str = "\n-----\n"+\
    'Final classification:\n\tClass:' + class_table[top_class] + '\n'+\
    '\tConfidence:'+ format(confidence,'.3%')

print(initial_str)
print(finial_str)