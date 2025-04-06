# Solar Eye 🌞👀

## Project Scope and Objectives 🎯

A multi-class classification problem to classify the faults of solar pv modules using UAV drones and deep learning model.

This project aims to develop an AI-driven system that utilizes drone technology to capture images or videos of solar photovoltaic (PV) modules for the purpose of predicting and classifying faults. The integration of AI with drone imagery can enhance the efficiency and effectiveness of solar panel inspections, leading to improved maintenance and operational reliability.

The solareye project falls under the Computer Vision category in Deep Learning and will implement CNN model architecture.

The study employs computer vision techniques to detect various errors in solar cell manufacture usingthe YOLOv8 architecture; this new technique can address the drawbacks of conventional detection techniques,including their rigidity, slowness, and reduced accuracy. The experimental findings prove that the proposed approach can effectively and accurately detect numerous defaults that are present on solar cell surfaces, short-circuit, fracture, and horizontal dissolution.

The objectives of the project would be:

**Automate Inspections:** Replace manual inspections with drone-based automated systems, reducing labor costs and time.

**Enhance Accuracy:** Achieve >90% accuracy in fault detection and classification using AI models.

**Increase Efficiency:** Reduce inspection time by 50% compared to traditional methods.

**Provide Actionable Insights:** Deliver detailed reports to maintenance teams for timely repairs.

The project is referenced by a journal article called [CNN-based automatic detection of photovoltaic solar module
anomalies in infrared images: a comparative study]
](https://www.researchgate.net/publication/383036389_CNN-based_automatic_detection_of_photovoltaic_solar_module_anomalies_in_infrared_images_a_comparative_study)


## Key Project Components 🛠️

A. Drone Setup

Drone Selection:
Model: DJI Matrice 300 RTK or similar industrial-grade drones.
Payload Capacity: Must support thermal and visible spectrum cameras.
Flight Time: Minimum 30 minutes for large solar farms.
Camera Specifications:
Thermal Camera: FLIR Vue Pro or similar, with a resolution of 640x512 pixels and a thermal sensitivity of <50mK.
Visible Spectrum Camera: 20MP or higher resolution for detailed visual inspection.
Flight Planning:
Software: Use DJI Pilot or Pix4D for automated flight planning.
Parameters: Altitude of 10-20 meters, speed of 2-3 m/s, and 70% image overlap for comprehensive coverage.

B. Data Collection

Image/Video Capture:
Frequency: Weekly or bi-weekly inspections, depending on the size of the solar farm.
Conditions: Capture images during clear weather and at consistent times (e.g., midday) to minimize variations.
Annotation:
Tools: Use LabelImg or CVAT for annotating images with bounding boxes and labels (e.g., cracks, hot spots, soiling).
Dataset Size: Aim for at least 10,000 annotated images for robust model training.

C. AI Model Development

Model Selection:
YOLOv8: For real-time object detection and classification.
EfficientNet-B4: For high-accuracy image classification.
Transfer Learning: Use pre-trained models on ImageNet and fine-tune them on your dataset.
Data Preprocessing:
Resizing: Resize images to 416x416 pixels for YOLO or 224x224 pixels for EfficientNet.
Augmentation: Apply techniques like rotation, flipping, and brightness adjustment to increase dataset diversity.
Normalization: Normalize pixel values to [0, 1] for better model convergence.
Training:
Framework: TensorFlow or PyTorch.
Hyperparameters: Learning rate of 0.001, batch size of 32, and 50-100 epochs.
Metrics: Monitor accuracy, precision, recall, and F1 score during training.

D. Integration and Deployment

Real-Time Processing:
Edge Computing: Use NVIDIA Jetson Xavier or similar hardware for onboard processing.
Cloud Integration: Store and process data in the cloud (e.g., AWS or Google Cloud) for scalability.
User Interface:
Dashboard: Develop a web-based dashboard using React.js or Angular for real-time monitoring.
Mobile App: Create an app using Flutter or React Native for field technicians.
Alerts and Reporting:
Notifications: Use SMS or email alerts for detected faults.
Reports: Generate PDF reports with fault locations, classifications, and recommended actions.

## Project Approach 🪜

Step-by-Step Approach

Define Requirements:
Identify the size of the solar farm, types of faults to detect, and user needs.

Select Technology:
Choose drones, cameras, and AI frameworks based on project requirements.

Data Collection:
Capture and annotate images/videos of solar panels.

Model Development:
Preprocess data, train the AI model, and evaluate performance.

Integration:
Integrate the AI model with the drone system and develop the user interface.

Testing:
Conduct field tests to validate the system’s performance.

Deployment:
Deploy the system for real-world use and train end-users.

Monitoring and Maintenance:
Continuously monitor the system and update the model as needed.


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

  Resources Needed

  Hardware:
  Drones: DJI Matrice 300 RTK (2-3 units).
  Cameras: FLIR Vue Pro (thermal) and Sony Alpha (visible spectrum).
  Edge Computing: NVIDIA Jetson Xavier for onboard processing.

  Software:
  AI Frameworks: TensorFlow, PyTorch, or YOLOv8.
  Annotation Tools: LabelImg or CVAT.
  UI Development: React.js for the dashboard and Flutter for the mobile app.
  
  Human Resources:
  Data Scientists: For model development and training.
  Drone Operators: For flight planning and execution.
  Software Developers: For UI and system integration.

  Additional Resources:
  
  * Project Dataset - https://github.com/RaptorMaps/InfraredSolarModules
  * TensorFlow - https://www.tensorflow.org/
  * YOLOV8 Model - https://github.com/ultralytics/ultralytics
  * Similar Computer Vision Problem - https://dev.mrdbourke.com/tensorflow-deep-learning/03_convolutional_neural_networks_in_tensorflow/
  * Convolutional Neural Network - https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
  * Basic Explanation of CNN - https://www.upgrad.com/blog/basic-cnn-architecture/
  * Solar Cell Image Classification Steps - https://umairrafiq.medium.com/solar-cells-image-classification-using-dnn-deep-neural-networks-10479ed02c16
  * Load and preprocess input images - https://www.tensorflow.org/tutorials/load_data/images
  
**Journal Articles:**

  1) Automatic detection of solar cell surface defects in electroluminescence images based on YOLOv8 algorithm - https://www.researchgate.net/publication/376124810_Automatic_detection_of_solar_cell_surface_defects_in_electroluminescence_images_based_on_YOLOv8_algorithm

 2) Image based surface damage detection of renewable energy installations using a unified deep learning approach - https://www.sciencedirect.com/science/article/pii/S2352484721005102?ref=pdf_download&fr=RR-2&rr=8802fa115cdb126e

