from batches import BatchGenerator
from dataset import Subset
from datasets.pets import PetsDataset
import ops as ops
import numpy as np
from models.simple import KNN
from models.simple import SGD
from test import Accuracy

# 1. Load the training, validation, and test sets as individual PetsDatasets.

print("Lade Bilder")
train_ds = PetsDataset('C:\\Users\\fabia\\OneDrive\\Dokumente\\Uni\\Deep Learning for Visual Computing\\Assignment1\\Datensatz\\cifar-10-python', Subset.TRAINING)
valid_ds = PetsDataset('C:\\Users\\fabia\\OneDrive\\Dokumente\\Uni\\Deep Learning for Visual Computing\\Assignment1\\Datensatz\\cifar-10-python', Subset.VALIDATION)
test_ds = PetsDataset('C:\\Users\\fabia\\OneDrive\\Dokumente\\Uni\\Deep Learning for Visual Computing\\Assignment1\\Datensatz\\cifar-10-python', Subset.TEST)
print("Bilder geladen")

# 2. Create a BatchGenerator for each one. Traditional classifiers don't usually train in batches so you can set the
# minibatch size equal to the number of dataset samples to get a single large batch - unless you choose a classifier
# that does require multiple batches.
op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])


train = next(iter(BatchGenerator(train_ds, len(train_ds), True, op)))
valid = next(iter(BatchGenerator(valid_ds, len(valid_ds), True, op)))
test = next(iter(BatchGenerator(test_ds, len(test_ds), True, op)))

# 3. Implement random or grid search (your choice) to tune one ore more hyperparameter values (such as k for KNN
# classifiers). Test at least 10 values. This is not a lot but depending on your choice of classifier and parameters
# can take a long time.

# 4. For each parameter to test, "train" a SimpleClassifier and then calculate the accuracy on the validation set.


train_acc_KNN = []
valid_acc_KNN = []
ks = np.arange(1,21)
prozent = 0
"""
print("Starte KNN")
for i in ks:
    prozent = prozent + 5

    model = KNN(3072, train_ds.num_classes(), i)
    acc = Accuracy()

    model.train(train.data, train.label)

    acc.update(model.predict(train.data), train.label)
    train_acc_KNN.append(acc.accuracy())

    acc.update(model.predict(valid.data), valid.label)
    valid_acc_KNN.append(acc.accuracy())

    print("k war:", i, "||", prozent, "% abgeschlossen")

best_k = ks[np.argmax(valid_acc_KNN)]
print("bestes k:", best_k)

model = KNN(3072, train_ds.num_classes(), best_k)  # statt i best?
model.train(train.data, train.label)
acc.update(model.predict(test.data), test.label)

test_acc_KNN = acc.accuracy()

print("Beende KNN")

"""
print("Starte SGD")
size = 50
train_acc_SGD = []
valid_acc_SGD = []


alpha = np.random.uniform(low=0.00001, high=0.005, size=(size,))  # random zwischen 0.0001 und 0.001
np.sort(alpha)
max_iter = 1000  # default =1000
learnin_Rate = []  # hängt von alpha ab
tolerance = True
prozent = 0

for i in alpha:
    prozent = prozent + (100/size)
    model = SGD(3072, train_ds.num_classes(), i, max_iter)
    acc = Accuracy()

    model.train(train.data, train.label)

    acc.update(model.predict(train.data), train.label)
    train_acc_SGD.append(acc.accuracy())

    acc.update(model.predict(valid.data), valid.label)
    valid_acc_SGD.append(acc.accuracy())
    print("alpha:", i, "||", prozent, "% abgeschlossen")


best_a = alpha[np.argmax(valid_acc_SGD)]
print("bestes alpha:", best_a)
print("i:", i)
model = SGD(3072, train_ds.num_classes(), best_a, max_iter)
model.train(train.data, train.label)
acc.update(model.predict(test.data), test.label)

test_acc_SGD = acc.accuracy()
print("train_acc_SGD:", train_acc_SGD)
print("vali_acc_SGD:", valid_acc_SGD)
print("test_acc_SGD:", test_acc_SGD)
print("Beende SGD")

# 5. Report the best parameters found and the corresponding validation accuracy.
# 6. Compute and report the accuracy on the test set with these parameters.

# PLOTTING
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#df_KNN = pd.DataFrame({'k': ks, 'Training': train_acc_KNN, 'Validation': valid_acc_KNN})
#df_KNN = df_KNN.set_index('k')

df_SGD = pd.DataFrame({'a': alpha, 'Training': train_acc_SGD, 'Validation': valid_acc_SGD})
df_SGD = df_SGD.set_index('a')

#p1 = sns.lineplot(data=df_KNN, linewidth=2.5, dashes=False)
p2 = sns.lineplot(data=df_SGD, linewidth=2.5, dashes=False)
#p1.set(xlabel = 'k', ylabel = 'Accuracy')
p2.set(xlabel = 'alpha', ylabel = 'Accuracy')
#p1.axhline(test_acc_KNN, color='g', linestyle='--')
p2.axhline(test_acc_SGD, color='g', linestyle='--')
#fig, axs = plt.subplots(2)
#fig.suptitle("Testing accuracy")
#axs[0].plot(p1)
#axs[1].plot(p2)
plt.annotate('Testing accuracy', xy=(2.5, 0.63), color='g')
plt.show()