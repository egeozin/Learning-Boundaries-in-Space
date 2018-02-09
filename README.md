6.884 Machine Learning Graduate Project
Ege Ozgirin,
Cagri Hakan Zaman

	This project is designed with:
	
	Python as framework + 
	SciPy stack for vector computation + 
	Keras for Convolutional Neural Network implementation + 
	MORSE library in Python/Blender for generating robot and environment simulation.

## Learning Boundaries in Space Using Convolutional Neural Networks

Recent discoveries in neuroscience suggest that a variety of cell groups in the hippocampus (i.e. place cells) and entorhinal cortex (i.e. grid cells, head direction cells and boundary cells) provide a representation of space that act as a cognitive map. This representation combines allocentric sensory cues with proprioceptive information, allowing the animal to capture invariant features of the environments and robustly navigate between different places. Our project explores machine learning techniques that can represent this architecture and to develop a working model in the context of robotic perception and navigation. In this paper we introduce a biologically inspired architecture that allows for robust visual inference and navigation in novel environments using convolutional neural networks (CNN). After an online training phase with a series of distance sensors, our model predicts activation patterns for the boundary cells when presented an image. Using predicted boundary cell activations the robot is able to navigate in the simulated environment while avoiding obstacles with a very high success rate.


<p align="center"><img src="https://github.com/egeozin/Learning-Boundaries-in-Space/blob/master/final_CNN.jpg" width="600"></p>


### License 

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
