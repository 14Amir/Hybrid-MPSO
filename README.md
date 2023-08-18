# Hybrid-MPSO
Implementing the article "Multi-level Particle Swarm optimized hyperparameters of Convolutional Neural Network" 
https://doi.org/10.1016/j.swevo.2021.100863  
Our goal in this article is to implement the algorithm described in the article in order to optimize the parameters of Convolutional Neural Networks (CNN).
First, we briefly explain convolutional neural networks:
Convolutional Neural Network (CNN) is a type of deep neural network that is widely used in the fields of image processing and pattern recognition. A CNN usually includes several Convolutional Layers, Feature Extractor Layers, Pooling Layers, and Fully Connected Layers.
In the convolutional neural layer, the network moves on the input image using a small and movable filter and extracts various features by applying convolution to the image. These filters increase the ability to detect certain patterns in the image. In these layers, activation function like ReLU is also used to add nonlinearity to the network.
In the feature analyzer layer, the features extracted in the convolutional layer are combined and aggregated to extract more and more complex features. In these layers, the activation function is also used.
In the combining layer, the dimensions of the image are reduced to reduce the number of network parameters and reduce the computational load. These layers are typically done by applying non-linear operations such as Max Pooling.
In the fully connected layer, the features extracted in the previous layers are connected to each other and used to make the final decision about image classification. Activation functions such as ReLU and Softmax are used in these layers.
By using convolutional neural network, it is possible to automatically identify different patterns and features in images and achieve very high accuracy in many image processing problems.
In the following, it should be said that the MPSO algorithm is the same as the PSO algorithm, but two nested colonies are used, and in fact, it uses the PSO algorithm twice in a nested manner, which will be briefly explained in the next chapter.
Now, in the mentioned article, the hyperparameters of the CNN neural network are considered as the location of the particles, and with the help of the MPSO algorithm, these particles update their location, and finally, the best location among the particles will be output, which is the criterion of being the best.
The accuracy of the CNN network with these hyperparameters (position of particles) is As a result, the output of this method is considered as the desired hyperparameters for CNN neural network training, and then by obtaining the corresponding weights, the accuracy of the network is measured on the test data.
In this article, the accuracy on different data sets is obtained with the help of this method and compared with the values obtained from previous methods as well as the random selection of hyperparameter values, and the superiority of this method is obvious in many comparisons.
In this article, we have implemented the method described in the article after describing the similar and inspiring algorithms of the authors in the next chapter, and our target dataset in this article was the MNIST dataset.
* For more information contact me with my email " Amirazad1380@gmail.com "
