
Implementation for Age-related Factor guided Joint Task Modeling Convolutional Neural Network on Tensorflow
====
This is a TensorFlow implementation of the face recognizer described in the paper "Age-related Factor guided Joint Task Modeling Convolutional Neural Network for Cross-Age Face Recognition".
Training data: The CACD dataset ([http://bcsiriuschen.github.io/CARC/]), MORPF Album 2 dataset([http://www.faceaginggroup.com/morph/]) and the CASIA-WebFace dataset ([http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html]) have been used for training.

Requirements
----
1.[Tensorflow r1.2](http://tensorflow.org).

2.[Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). A Matlab/Caffe implementation can be found [here](https://github.com/kpzhang93/MTCNN_face_detection_alignment)

Model
----
./src/pre_model.py: the model with just identity softmax and center loss.

./src/afjt_model.py: the multiloss model for AFJTCNNs.

Training
----
./src/pretrain.py : Pretraining CNN with identity label.

./src/finetune_afjt.py: finetune in a AFJTCNN way.

./src/finetune_multiloss.py: finetune the multiloss CNN without joint task factor analysis.

./src/test.py: test the EER of different checkpoints.

License
----
This code is distributed under MIT LICENSE
