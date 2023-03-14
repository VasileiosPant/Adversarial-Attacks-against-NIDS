# Adversarial Machine Learning attacks against Network Intrusion Detection Systems
---
## Attacks implemented:

### Fast Gradient Sign Method (FGSM)
FGSM is a gradient-based attack that adds small perturbations to input images to fool a neural network into misclassifying the image. [FGSM_attack.py](https://github.com/VasileiosPant/Adversarial-Attacks-against-NIDS/blob/main/Adversarial%20Attacks/FGSM_attack.py)

### Jacobian-based Saliency Map Attack (JSMA)
JSMA is a white-box attack that modifies the input image by iteratively adding or subtracting small values to maximize the difference in the output between the correct class and the predicted class. [JSMA_attack.py](https://github.com/VasileiosPant/Adversarial-Attacks-against-NIDS/blob/main/Adversarial%20Attacks/JSMA_attack.py)

### DeepFool
DeepFool is an iterative attack that computes the minimal perturbation needed to change the output of a neural network.
[DeepFool_attack.py](https://github.com/VasileiosPant/Adversarial-Attacks-against-NIDS/blob/main/Adversarial%20Attacks/DeepFool_attack.py)

#### Supervised Learning models implemented:
1. Decision Tree
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors

#### Unsupervised Learning models implemented:
1. K-Means


## Dependencies
1. Tensorflow 2.10
2. adversarial-robustness-toolbox 1.13.0
3. cleverhans 3.1.0
4. scikit-learn 1.1.3

#### Dataset used: [IoTID20](https://sites.google.com/view/iot-network-intrusion-dataset/home)
