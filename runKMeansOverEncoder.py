import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

(x_train_d, y_train_d), (x_test_d, y_test_d) = mnist.load_data()
# split train data to train/val
x_train, x_val, y_train, y_val = train_test_split(x_train_d, y_train_d, test_size=0.15, random_state=48)
train_sz = y_train.shape[0]
val_sz = y_val.shape[0] # assert tot sz = 60000
x_train = x_train.reshape(train_sz, 784) / 255
x_val = x_val.reshape(val_sz, 784)/255
x_test = x_test_d.reshape(10000, 784) / 255
y_test = y_test_d
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)

# plot data sample
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
num_classes = len(classes)
samples_per_class = 7

for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(x_train[idx].reshape(28,28).astype('float64'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


# --- autoencoder model ---
# Model architecture include 6 simple feedforward layers.
# Enocoding into 10 features (bottleneck layer) and decoding back to the orig image
# Loss function of MSE (reduce the error between the net output to the orig input), Adam optimizer

range_n_clusters = [5, 6, 7, 8, 9, 10] # numbers of clusters to check

for n_clusters in range_n_clusters:
    autoencoder = Sequential()
    autoencoder.add(Dense(512,  activation='elu', input_shape=(784,)))      # encoding
    autoencoder.add(Dense(128,  activation='elu'))
    autoencoder.add(Dense(n_clusters,    activation='linear', name="bottleneck"))
    autoencoder.add(Dense(128,  activation='elu'))                          # decoding (reconstruction)
    autoencoder.add(Dense(512,  activation='elu'))
    autoencoder.add(Dense(784,  activation='sigmoid'))
    autoencoder.compile(loss='mean_squared_error', optimizer = Adam())
    # train
    batch_sz = 1024
    epochs_num = 13
    trained_model = autoencoder.fit(x_train, x_train, batch_size=batch_sz, epochs=epochs_num, verbose=1, validation_data=(x_val, x_val))
    encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)
    encoded_data = encoder.predict(x_train)  # bottleneck representation - clustering to 10 classes
    decoded_output = autoencoder.predict(x_train)        # reconstruction
    encoding_dim = n_clusters

    # --- KMeans Classifier ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(encoded_data)
    encoded_pred = kmeans.labels_
    pred_clusters = kmeans.cluster_centers_
    # print( pred_clusters)

    # clusters results analysis - the density and separation of the samples in clusters
    # using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b)
    # (b - a) / max(a, b) --> range between [-1,1] (best score = 1)
    silhouette_avg = silhouette_score(encoded_data, encoded_pred)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)


# --> best silhouette score resulted for 8 clusters
best_clusters_num = 8
kmeans = KMeans(n_clusters=best_clusters_num, random_state=0).fit(encoded_data)


encoded_test_data = encoder.predict(x_test)  # bottleneck representation - clustering to 10 classes
encoded_test_pred = kmeans.predict(encoded_test_data)
silhouette_avg = silhouette_score(encoded_test_data, encoded_test_pred)
print("For Test data classified into n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

# plot test data results
classes = ['0', '1', '2', '3', '4', '5', '6', '7']
num_classes = len(classes)
samples_per_class = 20
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(encoded_test_pred == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(x_test[idx].reshape(28,28).astype('float64'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# plot Train predictions sample

classes = ['0', '1', '2', '3', '4', '5', '6', '7']
num_classes = len(classes)
samples_per_class = 20
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(encoded_pred == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(x_train[idx].reshape(28,28).astype('float64'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
