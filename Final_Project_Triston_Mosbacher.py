#Importing data
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home=os.getcwd())

images = mnist.data
targets = mnist.target

X_data = images #70,000 obs, 784 features
y = targets

#Split into training and test sets (85%/15% split, 59,500/10,500)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size = 0.15, 
                                                    random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
featSums = np.sum(X_train, axis = 0)
sum(featSums == 0) * np.shape(featSums)[0]**-1 #68 features with zero (8.7%)
X_tr_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score

testAccs = {}
testConfs = {}
trainAccs = {}
trainConfs = {}
for n in range(1, 12) :
    lda = LDA(n_components = n)
    lda = lda.fit(X_tr_scaled, y_train)
    
    #Training set: Accuracy and confusion matrix
    train_labs = lda.predict(X_tr_scaled)
    trainConf = confusion_matrix(y_train, train_labs)
    train_acc = accuracy_score(y_train, train_labs)
    trainAccs[n] = train_acc
    trainConfs[n] = trainConf
    
    #Test set: Accuracy and confusion matrix
    test_labs = lda.predict(X_test_scaled)
    testConf = confusion_matrix(y_test, test_labs)
    test_acc = accuracy_score(y_test, test_labs)
    testAccs[n] = test_acc
    testConfs[n] = testConf

#For 2-dim visualization of LDA:
lda = LDA(n_components = 2)
lda = lda.fit(X_tr_scaled, y_train)
X_lda = lda.transform(X_test_scaled)

target_names = range(10)
plt.figure()
colors = ['blue', 'red', 'green', 'black', 'turquoise', 'darkorange',
          'pink', 'brown', 'grey', 'navy']
lw = 2

for color, i, target_name in zip(colors, range(10), target_names):
    plt.scatter(X_lda[y_test == i, 0], X_lda[y_test == i, 1], color=color, alpha=.5, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of MNIST test set')
plt.xlabel('LD1')
plt.ylabel('LD2')

plt.show()

#PCA, show perc. variability explained:
from sklearn.decomposition import PCA
varExpl = {}
X_pca = {}
for n in range(1, 10) :
    pca = PCA(n_components = n)
    X_pca[n] = pca.fit_transform(X_tr_scaled)
    
    #Variance explained
    varExpl[n] = pca.explained_variance_ratio_ * 100
    
    
#For 2-dim visualization of PCA:
pca = PCA(n_components = 2)
pca = pca.fit(X_tr_scaled, y_train)
X_pca = pca.transform(X_test_scaled)

target_names = range(10)
plt.figure()
colors = ['blue', 'red', 'green', 'black', 'turquoise', 'darkorange',
          'pink', 'brown', 'grey', 'navy']
lw = 2

for color, i, target_name in zip(colors, range(10), target_names):
    plt.scatter(X_pca[y_test == i, 0], X_pca[y_test == i, 1], color=color, alpha=.5, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of MNIST test set')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.show()

#PCA scree plots:    
pca = PCA(n_components = 100)
X_pca100 = pca.fit_transform(X_tr_scaled)
varExpl = pca.explained_variance_ratio_ * 100
np.cumsum(varExpl)
np.sum(varExpl)

plt.figure()
plt.plot(np.array(range(1, 101)), varExpl, 'ro')
plt.title('Scree Plot for PCA')
plt.xlabel('PC')
plt.ylabel('% Variability')
plt.show()

plt.figure()
plt.plot(np.array(range(1, 101)), np.cumsum(varExpl), 'ro')
plt.title('Scree Plot for PCA')
plt.xlabel('PC')
plt.ylabel('Cumulative % Variability')
plt.show()