I used autoaugmentation provided by the torch library with CIFAR10 policy since:

changing hue and contrast is generally good for the model to recognize the dolly and wheels
erasing is good to help the model spot parts of the objects
rotation was beneicial as well

All in all, data augmentation helped the model detect better.

Labels were generated with makesense.ai
