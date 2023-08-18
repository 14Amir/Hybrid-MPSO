import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Define search space for hyperparameters at swarm level-1
nC_range = [1, 5]
nP_range = [1, 5]
nF_range = [1, 5]

p1min = np.array([1, 1, 1])
p1max = np.array([5, 5, 5])

# Define search space for hyperparameters at swarm level-2
c_nf_range = [1, 64]
c_fs_range = [1, 13]
c_pp_range = [0, 1]
c_ss_range = [1, 5]
p_fs_range = [1, 13]
p_ss_range = [1, 5]
p_pp_range = [0, 1]
op_range = [1, 1024]

def p2max(nC):
    p2max = np.array([])
    for i in range(nC):
        p2max[i] = [64, 13, 1, 5, 13, 5, 1, 1024]
    return p2max


def p2min(nC):
    p2min = np.array([])
    for i in range(nC):
        p2min[i] = [1, 1, 0, 1, 1, 1, 0, 1]
    return p2min

def train_CNN_model(position1, position2):  
    # Load the train dataset
    (X_train, y_train), _ = mnist.load_data()
    # Normalize the pixel values to be between 0 and 1
    X_train = X_train / 255.0
    # Reshape the input data to be a 4D tensor
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

    # Define the CNN model architecture with given hyperparameters
    model = tf.keras.models.Sequential()

    nC = position1[0]
    nP = position1[1]
    nF = position1[2]

    for i in range(nC):
        if i == 0:
            model.add(tf.keras.layers.Conv2D(filters=position2[i][0], kernel_size=position2[i][1] , strides=position2[i][3], padding=position2[i][2], activation='relu', input_shape=(28, 28, 1)))
        else:
            model.add(tf.keras.layers.Conv2D(filters=position2[i][0], kernel_size=position2[i][1] , strides=position2[i][3], padding=position2[i][2], activation='relu'))
        if i < nP:
            model.add(tf.keras.layers.MaxPooling2D(pool_size=position2[i][4], strides=position2[i][5], padding=position2[i][6]))
    model.add(tf.keras.layers.Flatten())
    for i in range(nF):
        model.add(tf.keras.layers.Dense(units=position2[i][7], activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the CNN model for 5 epochs
    model.fit(X_train, y_train, epochs=5, batch_size=128)

    return model

def CNN_train_Accuracy(position1, position2):
    model = train_CNN_model(position1, position2)
    # Load the train dataset
    (X_train, y_train), _ = mnist.load_data()
    # Normalize the pixel values to be between 0 and 1
    X_train = X_train / 255.0
    # Reshape the input data to be a 4D tensor
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    accuracy = model.evaluate(X_train, y_train)[1]
    return accuracy


# Define Particle class for swarm level-1 particles
class Particle:
    def __init__(self):
        self.position = [np.random.randint(nC_range[0], nC_range[1]), 
                          np.random.randint(nP_range[0], nP_range[1]),
                          np.random.randint(nF_range[0], nF_range[1])]
        self.velocity = [0, 0, 0]
        self.best_position = self.position.copy()
        self.fitness = float('inf')
        self.best_fitness = float('inf')
        self.gbest_position = np.zeros((self.position[0], 8))
        self.gbest_fitness = float('inf')

    def update_velocity(self, global_best_position, j, t_max1 = 5):
        alpha = 0.9
        temp = (10 * j - 2 * t_max1) / t_max1
        if j < alpha * t_max1:
            omega = 0.9
        else:
            omega =1/(1 + math.exp(temp))
        w = omega
        c1 = 2
        c2 = 2
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        vmin = p1min - np.array(self.position) 
        Vmax = p1max - np.array(self.position) 
        for i in range(3):
            temp1 = w * self.velocity[i] + c1 * r1 * (self.best_position[i] - self.position[i]) + c2 * r2 * (global_best_position[i] - self.position[i])
            if temp1 > Vmax[i]:
                self.velocity[i] = Vmax[i]
            elif temp1 < vmin[i]:
                self.velocity[i] = vmin[i]
            else:
                self.velocity[i] = temp1
       

    def update_position(self):
        self.position[0] += round(self.velocity[0])
        self.position[1] += round(self.velocity[1])
        self.position[2] += round(self.velocity[2])

    def evaluate_fitness(self):
        self.gbest_position, self.gbest_fitness = run_swarm2(5, self)
        self.fitness = self.gbest_fitness

        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

# Define Swarm class for swarm level-1
class Swarm:
    def __init__(self, num_particles):
        self.particles = []
        self.global_best_position = [0, 0, 0]
        self.global_best_fitness = float('inf')
        self.num_particles = num_particles
        self.global_best_particle_index = None
    
    def create_particles(self):
        for i in range(self.num_particles):
            particle = Particle()
            self.particles.append(particle)
    
    def update_particles(self, i):
        for j, particle in enumerate(self.particles):
            particle.update_velocity(self.global_best_position, i)
            particle.update_position()
            particle.evaluate_fitness()

            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
                self.global_best_particle_index = j

# Define Particle class for swarm level-2 particles
class Particle2:
    def __init__(self, parent_particle):
        self.parent_particle = parent_particle
        nC = self.parent_particle.position[0]
        nP = self.parent_particle.position[1]
        nF = self.parent_particle.position[2]
        position = np.zeros((nC,8))
        for i in range(nC):
            position[i][0] = np.random.randint(c_nf_range[0], c_nf_range[1])
            position[i][1] = np.random.randint(c_fs_range[0], c_fs_range[1])
            position[i][2] = np.random.randint(c_pp_range[0], c_pp_range[1])
            position[i][3] = np.random.randint(c_ss_range[0], c_ss_range[1])
        for i in range(nP):
            position[i][4] = np.random.randint(p_fs_range[0], p_fs_range[1])
            position[i][5] = np.random.randint(p_ss_range[0], p_ss_range[1])
            position[i][6] = np.random.randint(p_pp_range[0], p_pp_range[1])
        for i in range(nF):
            position[i][7] = np.random.randint(op_range[0], op_range[1])

        self.position = position
        self.velocity = np.zeros((nC,8))
        self.best_position = self.position.copy()
        self.fitness = float('inf')
        self.best_fitness = float('inf')
        

    def update_velocity(self, global_best_position, j, t_max2 = 5):
        alpha = 0.9
        temp = (10 * j - 2 * t_max2) / t_max2
        if j < alpha * t_max2:
            omega = 0.9
        else:
            omega =1/(1 + math.exp(temp))
        w = omega
        c1 = 2
        c2 = 2
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        nC = self.parent_particle.position[0]
        vmin = p2min(nC) - np.array(self.position)
        vmax = p2max(nC) - np.array(self.position)
        for i in range(len(self.velocity)):
            temp1 = w * self.velocity[i] + c1 * r1 * (self.best_position[i] - self.position[i]) + c2 * r2 * (global_best_position[i] - self.position[i])
            if temp1 > vmax[i]:
                self.velocity[i] = vmax[i]
            elif temp1 < vmin[i]:
                self.velocity[i] = vmin[i]
            else:
                self.velocity[i] = temp1
        
    def update_position(self):
        for i in range(len(self.position)):
            self.position[i] += round(self.velocity[i])
        
    def evaluate_fitness(self):
        self.fitness = CNN_train_Accuracy(self.parent_particle.position, self.position)

        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()
        
# Define Swarm class for swarm level-2
class Swarm2:
    def __init__(self, num_particles, parent_particle):
        self.parent_particle = parent_particle
        nC = self.parent_particle.position[0]
        self.particles = []
        self.global_best_position = np.zeros((nC,8))
        self.global_best_fitness = float('inf')
        self.num_particles = num_particles
        
    
    def create_particles(self):
        for i in range(self.num_particles):
            particle = Particle2(self.parent_particle)
            self.particles.append(particle)
    
    def update_particles(self, i):
        for particle in self.particles:
            particle.update_velocity(self.global_best_position, i)
            particle.update_position()
            particle.evaluate_fitness()

            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()

def run_swarm2(num_particles, parent_particle):
    Swarm_2 = Swarm2(num_particles, parent_particle)
    Swarm_2.create_particles()
    t_max2 = 5
    for i in range(t_max2):
        Swarm_2.update_particles(i)
    gbest_position = Swarm_2.global_best_position
    CNN_gbest = Swarm_2.global_best_fitness
    return gbest_position, CNN_gbest

def run_swarm(num_particles):
    Swarm_1 = Swarm(num_particles)
    Swarm_1.create_particles()
    t_max1 = 5
    for i in range(t_max1):
        Swarm_1.update_particles(i)
    gbest_position = Swarm_1.global_best_position
    CNN_gbest = Swarm_1.global_best_fitness
    gbesti_position = Swarm_1.particles[Swarm_1.global_best_particle_index].gbest_position
    return gbest_position, gbesti_position, CNN_gbest


gbest_position, gbesti_position, CNN_gbest = run_swarm(5)

# create a CNN model with gbest_position parameters and evaluate it on test data
model = train_CNN_model(gbest_position, gbesti_position)
_ , (X_test, y_test) = mnist.load_data()
# Normalize the pixel values to be between 0 and 1
X_test = X_test / 255.0
# Reshape the input data to be a 4D tensor
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
test_accuracy = model.evaluate(X_test, y_test)[1]
