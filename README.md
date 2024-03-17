# Solar Eye 🌞👀

## Project Scope and Objectives 🎯

A multi-class classification problem in which the scope of the project is to classify whether solar panels whether are clean, dusty, cracked, damaged, bird dropping, and snow using deep learning. This project is inspired and referenced by a journal article called [Fault Detection in Solar Energy Systems using Deep Learning](https://www.mdpi.com/2079-9292/12/21/4397). The solareye project falls under the Computer Vision category in Deep Learning and will implement CNN model architecture.

The journal article uses 19 pre-trained CNN models such as `EfficientB0`, `ResNet`, `AlexNet`, etc. Along with that, `SVM` and `NCA` was used to classify 12 different classes of solar thermal images of around 20,000 images that was captured by UAV systems.

The overall conclusion of the article mentioned the use of `Efficientb0` pre-trained CNN model and this project will utilize CNN models as well.

The project objectives would be to:

  1. Get the dataset - https://www.kaggle.com/code/madenenivamsikrishna/fault-detection-using-resnet50-with-83-accuracy/input
  2. Perform EDA (Exploratory Data Analysis)
  3. Data Preprocessing & feature engineering
  4. Create a CNN model
  5. Use a pre-trained model and adjust it to our dataset (optional)
  6. Train the model
  7. Evaluate the model
  8. Test the model by predicting it on a random image.

## Project Applications 🌏

1) Solar Panel Maintenance System:

  * Develop a system that automatically assesses the condition of solar panels, providing insights into whether they are clean, damaged, dirty, etc.
  * Implement a notification system to alert maintenance personnel when issues are detected.

2) Energy Production Optimization:

  * Use the classification results to optimize the cleaning schedule for solar panels, ensuring they operate at peak efficiency.
  * Implement an intelligent system that recommends optimal cleaning frequencies based on historical data.

3) Automated Inspection Drones:

  * Integrate the image classification model with drones equipped with cameras for automated aerial inspections of large solar farms.
  * Enable the drones to identify and document areas that require attention.

4) Remote Monitoring Dashboard:

  * Develop a web-based dashboard that allows users to remotely monitor the status of solar panels in real-time.
  * Provide visualization tools and historical data to track the performance of individual panels or entire solar arrays.

5) Predictive Maintenance:

  * Implement predictive maintenance based on the classification results and historical data.
  * Use machine learning algorithms to predict potential issues before they occur, allowing for proactive maintenance.

6) Performance Analytics:

  * Provide detailed analytics on the performance of solar panels over time.
  * Include metrics such as energy production, efficiency, and degradation rates.

7) Integration with Smart Grids:

  * Explore integration possibilities with smart grid systems to optimize energy distribution based on the health of individual solar panels.

8) Educational Tools:

  * Develop educational tools or resources that explain the importance of solar panel maintenance and how the image classification system works.

9) Research Collaboration:

  * Collaborate with research institutions or environmental organizations to contribute data for broader research on the health and performance of solar energy systems

## Project Resources 📚

  * Project Dataset - https://www.kaggle.com/code/madenenivamsikrishna/fault-detection-using-resnet50-with-83-accuracy/input
  * TensorFlow - https://www.tensorflow.org/
  * Similar Computer Vision Problem - https://dev.mrdbourke.com/tensorflow-deep-learning/03_convolutional_neural_networks_in_tensorflow/
  * Convolutional Neural Network - https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
  * Basic Explanation of CNN - https://www.upgrad.com/blog/basic-cnn-architecture/
  * Solar Cell Image Classification Steps - https://umairrafiq.medium.com/solar-cells-image-classification-using-dnn-deep-neural-networks-10479ed02c16
  
**Journal Articles:**

  1) Fault Detection in Solar Energy Systems: A Deep Learning Approach - https://www.mdpi.com/2079-9292/12/21/4397#B14-electronics-12-04397

  2) Deep Learning fault detection of solar pv using thermal images by UAV drones - https://dergipark.org.tr/en/download/article-file/2337359

  3) Automatic Inspection of Photovoltaic Power Plants Using Aerial
Infrared Thermography - https://pdfs.semanticscholar.org/8748/6796371016c0d41496deaa02f5d4fdb9c1ec.pdf?_gl=1*dylbop*_ga*MTYzMTUwNDM3Ny4xNzA2MjYzNjA2*_ga_H7P4ZT52H5*MTcwNjI2MzYwNS4xLjEuMTcwNjI2NDM1Ny4yMS4wLjA.

  4) Computer Vision Tool for Detection, Mapping and Fault
Classification of PV Modules in Aerial IR Videos - https://arxiv.org/pdf/2106.07314.pdf
