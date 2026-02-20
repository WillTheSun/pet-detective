# Project Goals
Complete the following goals:

Dataset Preparation: Use the Oxford-IIIT Pet Dataset, which contains images of 37 pet breeds (cats and dogs) with annotations. This dataset is balanced and provides a moderate challenge.

Model Architecture Selection: Choose from one of the following pre-trained models for transfer learning:

MobileNetV2 / V2: A lighter-weight option designed for mobile and resource-constrained environments, making it suitable for running on laptops.

Transfer Learning Implementation: Fine-tune your chosen pre-trained model on the pet dataset, leveraging existing weights to achieve high accuracy with limited training time.

Extra Credit
Export and wrap your model in a web-app (e.g. a game that allows users to compete with the bot in multiple-choice pet-guessing game).
Fine tune an image segmentation model using the data included with the Oxford-IIIT dataset.

You did it
Architecture choice affects deployment constraints. AlexNet offers simplicity for learning, ResNet balances performance with efficiency, and MobileNet prioritizes resource constraints over raw accuracy.

Balanced datasets simplify training. The Oxford-IIIT Pet Dataset's 37 balanced classes eliminate class imbalance issues that plague many real-world projects.

Transfer learning works across domains. ImageNet-pretrained models successfully adapt to pet classification despite being trained on general objects, demonstrating feature transferability.

Model deployment drives architecture decisions. Web apps favor lighter models like MobileNet, while high-accuracy applications justify ResNet's computational overhead.

Real World Applications
Custom classification with limited data: Apply pretrained vision models to niche domains (manufacturing defects, satellite imagery, medical scans) where collecting training data is expensive

Research acceleration: Use pretrained models as baselines to focus research efforts on novel architectures or training techniques

Cost-effective AI deployment: Leverage big tech's training investments to build competitive products without massive compute budgets
