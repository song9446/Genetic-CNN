from scipy.stats import bernoulli
from deap import base, creator, tools, algorithms
import numpy as np

STAGE_NUM = 3 # S
NODE_NUM = [3,4,5] # K
FILTER_NUM = [8, 8, 16] 
KERNEL_SIZE = [3, 3, 3]
BATCH_SIZE = 32
EPOCHS = 10
L = sum(n*(n-1)//2 for n in NODE_NUM) # total # of bit in a gene
i = 0
L_cumsum = [None] * STAGE_NUM
for s in range(STAGE_NUM):
    L_cumsum[s] = [0] * (NODE_NUM[s]-1)
    for n in range(NODE_NUM[s]-1):
        L_cumsum[s][n] = i
        i += n+1


INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

import tensorflow as tf

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.reshape(-1, *INPUT_SHAPE)
x_test = x_test.reshape(-1, *INPUT_SHAPE)
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

def Conv2DReluBatchNorm(filters, kernel_size, X):
    return tf.keras.layers.Activation(activation='relu')(
               tf.keras.layers.BatchNormalization()(
                    tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(X)))
def ishead(gene, stage, node):
    if node==0: return True
    i=L_cumsum[stage][node-1]
    return 0==sum(gene[i:i+node])
def isleaf(gene, stage, node):
    if node==NODE_NUM[stage]-1: return True
    return 0==sum(gene[L_cumsum[stage][i]+node] for i in range(node, NODE_NUM[stage]-1))
def InputLayer(input_shape, X):
    return tf.keras.layers.Conv2D(input_shape=input_shape, filters=8, padding='same', kernel_size=3, activation="relu")(X) # input conv
def OutputLayer(output_shape, X):
    return tf.keras.layers.Dense(output_shape, activation='softmax')(tf.keras.layers.Flatten()(tf.keras.layers.Dropout(0.5)(X)))
def expression(gene):
    tf.reset_default_graph()
    convs = [None]*sum(NODE_NUM)
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    input_layer = InputLayer(INPUT_SHAPE, inputs)
    last_layer = input_layer
    for s in range(STAGE_NUM):
        for n in range(NODE_NUM[s]): # stage s node n
            if ishead(gene, s, n):
                convs[s*STAGE_NUM + n] = Conv2DReluBatchNorm(FILTER_NUM[s], KERNEL_SIZE[s], last_layer) 
            else:
                cumsum = L_cumsum[s][n-1]
                input_nodes = [convs[s*STAGE_NUM + i] for i in range(n) if gene[cumsum+i]>0]
                add = tf.keras.layers.Add()(input_nodes) if len(input_nodes)>1 else input_nodes[0]
                convs[s*STAGE_NUM + n] = Conv2DReluBatchNorm(FILTER_NUM[s], KERNEL_SIZE[s], add) 
        leafs = [convs[s*STAGE_NUM + n] for n in range(NODE_NUM[s]) if isleaf(gene, s, n)]
        add = tf.keras.layers.Add()(leafs) if len(leafs)>1 else leafs[0]
        pool = tf.keras.layers.MaxPool2D()(add)
        last_layer = pool
    output_layer = OutputLayer(NUM_CLASSES, last_layer)
    model = tf.keras.Model(inputs=inputs, outputs=output_layer)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adadelta(lr=0.01), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    return model

def evaluate(gene):
    model = expression(gene)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("loss: {}, acc: {}".format(loss, accuracy))
    return accuracy,


population_size = 20
num_generations = 3

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list , fitness = creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("binary", bernoulli.rvs, 0.5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.binary, n = L)
toolbox.register("population", tools.initRepeat, list , toolbox.individual)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb = 0.8)
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", evaluate)

population = toolbox.population(n = population_size)
result = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.05, ngen = num_generations, verbose = True)

# print top-3 optimal solutions 
best_individuals = tools.selBest(population, k = 3)
for bi in best_individuals:
    print(bi)
for bi in best_individuals:
    evaluate(bi)


# --------------------------------------------------------------------------------------------------------------------
