cv_project
==============================

This project is an assingment for Computer Vision course at Wroclaw Univ. of Science and Technology. 

Main project goal was cars detection in aerial images using sliding window and two different approaches for classification:
- image descriptors (Histogram of Oriented Gradients, Local Binary Patterns) and SVM classifiers,
- image preprocessing (normalization, histogram equalization, Gaussian blur) and pretrained convolutional neural network classifier (transfer learning on ResNet18 and ResNet50).

Datasets
------------
- aerial
- vedai
- prepared

Outcome
------------
Table below presents best results gained among all exepriments. For dataset prepared by authors models trained on aerial-cars-dataset have been used.

![Table](./assets/table.png)

Most interesting aoutcome is neural network good knowledge generalization on prepared dataset. Images below shows sample results of detections on this dataset:

![Sample_detections](./assets/images.png)

Full report covers all experiments and cocnlusions is available [here]() (polish language only). 

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
