# Google - Isolated Sign Language Recognition

Attempt at Google's competitiom on [Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs) hosted on Kaggle.

Download the datasets from following links:

[1] [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/data)

[2] [GISLR Extended Train Dataframe](https://www.kaggle.com/datasets/dschettler8845/gislr-extended-train-dataframe)

[3] [ASL Pretrained](https://www.kaggle.com/datasets/bishwashk/asl-pretrained)

The goal of the competition was to classify isolated American Sign Language (ASL) signs. We were create a TensorFlow Lite model trained on labeled landmark data extracted using the MediaPipe Holistic Solution.

I first attempted to create a model with pytorch. The training of the model worked well but there was error while converting the model into tensorflow lite model.

Then I made use of `ConvLSTM` implemented in Tensorflow since we have a 3D data of shape $[37, 77, 3]$. The modelling and inference were successful and tflite model was also created.

