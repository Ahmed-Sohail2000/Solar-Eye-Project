# Contains All of the functions created in the notebook for the project

# create a function to view random images
def view_random_image(target_dir, target_class):
  '''
  Views a random image from the target directory of the target class (Clean, Crack, Dirty, Bird dropping)
  '''

  # Visualize images from the train data
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg    # import matplotlib module to work with images
  import random, os
  
  # Setup target directory
  target_folder = target_dir + target_class   # target folder will be "solar_panels_classes/train" + "Clean" (or Damaged or Dirty)

  # get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)    # gets 1 random sample image from the list of files in target folder
  print(random_image)

  # Read in the image and plot it using matplotlib
  image = mpimg.imread(target_folder + '/' + random_image[0])   # reads the first element from the random selected target folder. Rmoves the [] in ['20170818_105127_jpg.rf.670a60792f66c43d4644ee469b6c2f3b.jpg']
  plt.imshow(image)
  plt.title(target_class)
  plt.axis('Off');

  print(f'Image shape: {image.shape}')   # show the shape of the image

  return image

# create a function to preprocess our data
def preprocess_image(train_path, valid_path, test_path):

  '''
  Preprocesses images by normalizing (values betweem 0 & 1) and resizing (224,224) it into batches of data (32)
  '''

  # import ImageDataGenerator
  import tensorflow as tf
  from tensorflow.keras.preprocessing.image import ImageDataGenerator

  # setup random seed
  tf.random.set_seed(42)

  # create train, valid, and test datagen. Preprocess data (get all the of pixel values between 0 & 1, also called scaling/normalization)
  train_datagen = ImageDataGenerator(rescale = 1./255)
  valid_datagen = ImageDataGenerator(rescale = 1./255)
  test_datagen = ImageDataGenerator(rescale = 1./255)

  # import data from directories and turn it into batches
  train_data = train_datagen.flow_from_directory(
      directory = train_path,                   # target directory
      target_size = (224, 224),                 # target size for the images (height, width)
      color_mode = 'rgb',                       # color channel = 3 (RGB)
      class_mode = 'categorical',               # creates a 2d numpy array of one hot encoded labels. Supports multi-label output.
      batch_size = 32,
      shuffle = True,
      seed = 42)

  valid_data = valid_datagen.flow_from_directory(
      directory = valid_path,
      target_size = (224, 224),
      color_mode = 'rgb',
      class_mode = 'categorical',
      batch_size = 32,
      shuffle = False,
      seed = 42)

  test_data = test_datagen.flow_from_directory(
      directory = test_path,
      target_size = (224, 224),
      color_mode = 'rgb',
      class_mode = 'categorical',
      batch_size = 32,
      shuffle = False,
      seed = 42)

  return train_data, valid_data, test_data

# let's create a function that will view the training and validation accuracy and loss curves
def view_loss_accuracy(History):
  '''
  plots the model accuracy and loss of training and validation
  '''

  # import library
  import matplotlib.pyplot as plt
  
  # setup loss and accuracy
  loss = History.history['loss']
  val_loss = History.history['val_loss']

  accuracy = History.history['accuracy']
  val_accuracy = History.history['val_accuracy']

  # setup epochs
  epochs = range(len(loss)) # how many epochs did it run for?

  # plot the model accuracy
  plt.plot(epochs, accuracy, label = 'train accuracy')
  plt.plot(epochs, val_accuracy, label = 'valid accuracy')
  plt.title('Model Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()

  # plot model loss
  plt.figure();
  plt.plot(epochs, loss, label = 'train loss')
  plt.plot(epochs, val_loss, label = 'valid loss')
  plt.title('Model Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

# Lets create a function that will take a custom image and be processed to then be able to make predictions by our trained model
def preprocess_custom_image(image_path):
  '''
  Takes a custom image and preprocess it by:
  1. Setting the target size to (224, 224, 3) -> (Width, Height, Color Channels)
  2. Normalizing the image (values between 0 & 1)
  3. Expanding the dimension of the image (1, 224, 224, 3) -> (batch_size, width, height, color channels)
  '''

  # import the libraries
  import tensorflow as tf
  import matplotlib.image as mpimg
  import matplotlib.pyplot as plt

  # read the image
  local_image_path = tf.keras.utils.get_file(origin = image_path)

  # read the image from the image path and set the target size
  image = tf.keras.preprocessing.image.load_img(local_image_path, target_size=(224, 224))

  # Convert the image to a NumPy array
  image = tf.keras.preprocessing.image.img_to_array(image)

  # 2. reshape the image to [1, 224, 224, 3] -> [batch_size, width, height, color channels]
  image = tf.expand_dims(image, axis = 0)

  # 3. normalize the image (values between 0 & 1)
  image = image / 255.

  # plot the image
  plt.imshow(image[0])  # reduce the dimension to (224, 224, 3)
  plt.axis(False)
  print(f'Image shape: {image.shape}')

  return image

# create a function to plot the predicted image against the actual image
def pred_and_plot(model, image_path):
  '''
  Imports an image located at image path, makesa prediction with the model
  and plots the image with the predicted class as the title
  '''

  # import library
  import tensorflow as tf
  import numpy as np
    
  # import the target image and preprocess it by calling the function (preprocess_custom_image())
  image = preprocess_custom_image(image_path)

  # make a prediction
  pred = model.predict(image)

  # get the predicted class
  pred_class = class_names[int(tf.round(np.argmax(pred)))]

  # plot the image and the predicted class
  # import matplotlib
  import matplotlib.pyplot as plt
  
  plt.imshow(image[0])
  plt.title(f'Prediction: {pred_class}')
  plt.axis(False);

def create_callbacks(dir_name, experiment_name):
  '''
  Creates a tensorboard callback, earlystoppingcallback, model checkpoint, and learning rate scheduler
  '''

  import datetime
  import tensorflow as tf
  import os

  # create a directory to store the experiments of the models as logs with the current date and time
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
  print(f'Saving TensorBoard Log Files to: {log_dir}')

  # use the earlystoppingcallback
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', verbose = 0, mode = 'min', restore_best_weights = True)
  
  # create the model checkpoint
  model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(dir_name, experiment_name),
                                                        monitor = 'val_loss',
                                                        verbose = 1, 
                                                        save_best_only = True, 
                                                        mode = 'min',
                                                        save_weights_only = True)
  
  # create learning rate scheduler function
  def schedule(epoch, lr):
    '''
    creates a learing rate schedule and decreases it based on the learning rate and epoch
    '''

    if epoch < 5:
      return lr
    
    else:
      return lr * tf.math.exp(-0.1)

  # call the learning rate method
  learning_rate = tf.keras.callbacks.LearningRateScheduler(schedule)

  return tensorboard_callback, early_stopping, model_checkpoint, learning_rate

# Let's make a create_model() function to create a model from a URL
def create_model(model_url, num_classes = 3, IMAGE_SHAPE = (224, 224)):
  '''
  Takes a TensorFlow Hub model and creates a Keras Sequential model with it.

  Args:
  model_url (str): A TensorFlow Hub feature extraction URL.
  num_classes (int): Number of output neurons in the output layer,
  should be equal to number of target classes, default to 3.

  Returns:
  An uncompiled Keras Sequential model with model_url as feature extractor layer
  and Dense output layer with num_classes output neurons.
  '''

  # import dependencies
  import tensorflow as tf
  import tensorflow_hub as hub
  from tensorflow.keras import layers

  # Download the pretrained model and save it as a Keras layer
  feature_extraction_layer = hub.KerasLayer(model_url,
                                            trainable = False, # freeze the already learned patterns from ImageNet images.
                                            name = 'feature_extraction_layer',
                                            input_shape = IMAGE_SHAPE + (3,)) # (224, 224, 3)

  # Create our own model
  model = tf.keras.Sequential([
      feature_extraction_layer,
      layers.Dense(num_classes, activation = 'softmax', name = 'output_layer')
      ])

  # compile the model
  model.compile(loss = 'CategoricalCrossentropy',
                optimizer = 'Adam',
                metrics = ['accuracy'])

  return model
