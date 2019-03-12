#### K Means clustering algorithm                                ####
#### Created by: Tanasorn Chindasook and Prateek Kumar Choudhary ####
#### Created on: 08/03/2019                                      ####
#### Last updated: 12/03/2019                                    ####

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

# Initialize the parameters for folder and datafile location
dataFolder = '/home/prateek/Documents/DE/Data Engineering - Jacobs/Sem 2/Machine Learning/Mini_Project1/'
dataFile = 'mfeat-pix.txt'

# Load the main data file
data = np.loadtxt(dataFolder+dataFile)


# Function to visualize the number, takes two parameters
# data : image vector data of len 240
# imgNme : name with which the image should be saved
def plot_figure(data, imgNme):
    row = np.array(data)
    shape = (16, 15)
    rows_matrix = row.reshape(shape)
    plt.imshow(rows_matrix, cmap=plt.get_cmap('gray_r'))
    plt.imsave(imgNme, rows_matrix, cmap=plt.get_cmap('gray_r'))


# Function to assign each data point to a random cluster, takes two arguments
# df : the two dimnestional data array of image vectors
# k : number of clusters to which the data should be assigned
def random_assign_class(df, k):
    # Ensure that all classes are initialised at least once
    x = [x for x in range(k)]
    shuffle(x)
    # Fill in the other points that havent been initialised randomly
    missing_index = len(df) - len(x)
    missing_values = np.random.randint(0, k, missing_index)
    # join the two random initialisations together
    final_initialisation = x + missing_values.tolist()
    groups = np.zeros((len(final_initialisation), 1))
    for indx, item in enumerate(final_initialisation):
        groups[indx] = item
    data = np.append(df, groups, axis=1)
    return data


# Function to run K-Means clustering, takes two arguments
# data : ndarray of data points that need to be clustered
# clusterSize : number of clusters
def kmeans(data, clusterSize):
    numData = len(data)         # Number of data points
    dimSize = len(data[0])      # Dimension of each data point
    # Assign each point to a cluster randomly
    data = random_assign_class(data, clusterSize)

    # Keep running the kmeans algorithm while the classes keep changing
    classChange = True
    while classChange == True:
        # Store the old class values for all the image vectors
        oldClasses = data[:, -1].copy()
        # Calculate the centroid for each group and store it in a numpy array
        # the centroid is just an average of all the points in the group
        centroid = np.zeros((clusterSize, dimSize))
        for i in range(len(centroid)):
            centroid[i] = np.sum(data[data[:, -1] == i][:, :-1], axis=0)/len(data[data[:, -1] == i])

        # Calculate the distance of each point from all the centroids
        distances = np.zeros((numData, clusterSize))
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                distances[i, j] = np.linalg.norm(data[i, :-1]-centroid[j])
        # Re-assign the groups based on closest centroid
        for i in range(len(distances)):
            data[i, -1] = np.argmin(distances[i])
        # Store the new class values for the image vector
        newClasses = data[:, -1].copy()

        # Compare if the old and new classes are the same
        if sum(newClasses != oldClasses) == 0:
            classChange = False

    return centroid


# We are going to be using the 200 vectors for digit '2' for
# K-means clustering with cluster sizes 1,2,3 and 200.
clusterSizes = [1, 2, 3, 200]
twos = data[400:600][:]

for size in clusterSizes:
    centroids = kmeans(twos, size)
    c = 1
    for centroid in centroids:
        plot_figure(centroid, 'Number=2,k='+str(size)+',codebookVector='+str(c)+'.jpg')
        c = c + 1

# Now we will run K-Means to differentiate between the 10 numbers
centroids = kmeans(data, 10)
c = 1
for centroid in centroids:
    plot_figure(centroid, 'fullset_K=10,codebook_vector='+str(c)+'.jpg')
    c = c + 1

# Next we will iterate through the zeroes data set 5 times in order to explore the effects of random initialisation
zeroes = data[0:200][:]
for i in range(5):
    centroids = kmeans(zeroes, 3)
    c = 1
    for centroid in centroids:
        plot_figure(centroid, str(i)+'Run_Zeros(K=3)_CB='+str(c)+'.jpg')
        c = c + 1
