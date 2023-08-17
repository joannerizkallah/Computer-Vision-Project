import torchvision as tv
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as transforms

img_path = "./dataset/images-20230811T075637Z-001/images/"
list_img=[img for img in os.listdir(img_path) if img.endswith('.png')==True]

random_auto1_transform = tv.transforms.AutoAugment(
    tv.transforms.AutoAugmentPolicy.CIFAR10
)

for i in range(1, 100):
    torch_image = Image.open(img_path + list_img.pop())
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(torch_image)
    new_image = random_auto1_transform(torch_image)
    plt.subplot(10,10, i)
    plt.imshow(new_image)
    new_image.save("./Images/% s" % i + ".png")
plt.show()

