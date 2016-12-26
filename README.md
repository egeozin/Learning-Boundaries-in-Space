6.884 Machine Learning Graduate Project
Ege Ozgirin egeozin@mit.edu
Cagri Hakan Zaman zaman@mit.edu


## Learning Boundaries in Space Using Convolutional Neural Networks

Recent discoveries in neuroscience suggest that a variety of cell groups in the hippocampus (i.e. place cells) and entorhinal cortex (i.e. grid cells, head direction cells and boundary cells) provide a representation of space that act as a cognitive map. This representation combines allocentric sensory cues with proprioceptive information, allowing the animal to capture invariant features of the environments and robustly navigate between di↵erent places. Our project explores machine learning techniques that can represent this architecture and to develop a working model in the context of robotic perception and navigation.In this paper we introduce a biologically inspired architecture that allows for robust visual inference and navigation in novel environments using convolutional neural networks (CNN) After an online training phase with a series of distance sensors, our model predicts activation patterns for the boundary cells when presented an image. Using predicted boundary cell activations the robot is able to navigate in the simulated environment while avoiding obstacles with a very high success rate.

![alt tag](https://github.mit.edu/egeozin/Learning-Boundaries-in-Space/blob/master/final_CNN.jpg)

<img src="https://github.mit.edu/egeozin/Learning-Boundaries-in-Space/blob/master/final_CNN.jpg" width="300">

