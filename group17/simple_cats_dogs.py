from dlvc.batches import BatchGenerator
from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
import dlvc.ops as ops
import numpy as np
from dlvc.models.simple import KNN
from dlvc.models.simple import SGD
from dlvc.test import Accuracy

# 1. Load the training, validation, and test sets as individual PetsDatasets.

fp = 'C:\\Users\\bschwendinger\\Documents\\github\\dlvc\\cifar-10-batches-py'
# fp = 'C:\\Users\\fabia\\OneDrive\\Dokumente\\Uni\\Deep Learning for Visual Computing\\Assignment1\\Datensatz\\cifar-10-python'

print("Lade Bilder")
train_ds = PetsDataset(fp, Subset.TRAINING)
valid_ds = PetsDataset(fp, Subset.VALIDATION)
test_ds = PetsDataset(fp, Subset.TEST)
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

# set seed for reproducibility
np.random.seed(373)

train = next(iter(BatchGenerator(train_ds, len(train_ds), True, op)))
valid = next(iter(BatchGenerator(valid_ds, len(valid_ds), True, op)))
test = next(iter(BatchGenerator(test_ds, len(test_ds), True, op)))

# 3. Implement random or grid search (your choice) to tune one ore more hyperparameter values (such as k for KNN
# classifiers). Test at least 10 values. This is not a lot but depending on your choice of classifier and parameters
# can take a long time.

# 4. For each parameter to test, "train" a SimpleClassifier and then calculate the accuracy on the validation set.
acc = Accuracy()

train_acc_KNN = []
valid_acc_KNN = []
ks = np.arange(1,21)
prozent = 0

print("Starte KNN")
for i in ks:
    prozent = prozent + 100 / len(ks)

    model = KNN(3072, train_ds.num_classes(), i)
    model.train(train.data, train.label)
    acc.reset()
    acc.update(model.predict(train.data), train.label)
    train_acc_KNN.append(acc.accuracy())

    acc.reset()
    acc.update(model.predict(valid.data), valid.label)
    valid_acc_KNN.append(acc.accuracy())

    print("k war: {} || {}% abgeschlossen".format(i, prozent))

best_k = ks[np.argmax(valid_acc_KNN)]
print("bestes k:", best_k)

model = KNN(3072, train_ds.num_classes(), best_k)  # statt i best?
model.train(train.data, train.label)
acc.reset()
acc.update(model.predict(test.data), test.label)
test_acc_KNN = acc.accuracy()

print("Beende KNN")


print("Starte SGD")
size = 20
train_acc_SGD = []
valid_acc_SGD = []

max_iter = 1000  # default =1000
alpha = np.logspace(start=-5, stop=1.5, num=1000, base=10.0)
tolerance = True
prozent = 0

used_alpha = []
used_loss = []

# random search for two losses
for loss in ['modified_huber', 'log']:
    for k in range(size):
        a = np.random.choice(alpha)
        used_alpha.append(a)
        used_loss.append(loss)
        prozent = prozent + (100/(2*size))
        model = SGD(3072, train_ds.num_classes(), loss, a, max_iter)

        model.train(train.data, train.label)

        acc.reset()
        acc.update(model.predict(train.data), train.label)
        train_acc_SGD.append(acc.accuracy())

        acc.reset()
        acc.update(model.predict(valid.data), valid.label)
        valid_acc_SGD.append(acc.accuracy())
        print("loss:{} || alpha: {} || {} % abgeschlossen".format(loss, a, prozent))


best_alpha = used_alpha[np.argmax(valid_acc_SGD)]
best_loss = used_loss[np.argmax(valid_acc_SGD)]

print("bestes alpha:{}".format(best_alpha))
print("bester loss:{}".format(best_loss))

model = SGD(3072, train_ds.num_classes(), best_loss, best_alpha, max_iter)
model.train(train.data, train.label)
acc.reset()
acc.update(model.predict(test.data), test.label)

test_acc_SGD = acc.accuracy()
print("train_acc_SGD:", train_acc_SGD)
print("valid_acc_SGD:", valid_acc_SGD)
print("test_acc_SGD:", test_acc_SGD)
print("Beende SGD")

# 5. Report the best parameters found and the corresponding validation accuracy.
# 6. Compute and report the accuracy on the test set with these parameters.

# PLOTTING
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_KNN = pd.DataFrame({'k': ks, 'Training': train_acc_KNN, 'Validation': valid_acc_KNN})
df_KNN = df_KNN.set_index('k')

y_min = 0.4
y_max = 1

p1 = sns.lineplot(data=df_KNN, linewidth=2.5, dashes=False)
p1.set(xlabel = 'k', ylabel = 'Accuracy', title='KNN')
p1.axhline(test_acc_KNN, color='g', linestyle='--')
plt.annotate('Testing accuracy', xy=(2.5, 0.63), color='g')
plt.ylim(y_min, y_max)
plt.savefig('KNN.png')
plt.close()


df_SGD = pd.DataFrame({'alpha': used_alpha, 'loss': used_loss, 'Training': train_acc_SGD, 'Validation': valid_acc_SGD})
#df_SGD = df_SGD.set_index('alpha')

p2 = sns.scatterplot(x='alpha', y='Training', data=df_SGD.loc[df_SGD['loss'] == 'log'])
p2.set(xlabel = 'alpha', ylabel = 'Accuracy', xscale="log", title='SGD log loss, validation set')
p2.axhline(0.6061, color='g', linestyle='--')
plt.annotate('Testing accuracy', xy=(1e-3, 0.57), color='g')
plt.ylim(y_min, y_max)
#plt.show()
plt.savefig('SGD_log.png')
plt.close()


p3 = sns.scatterplot(x='alpha', y='Training', data=df_SGD.loc[df_SGD['loss'] != 'log'])
p3.set(xlabel = 'alpha', ylabel = 'Accuracy', xscale='log', title='SGD modified_huber loss, validation set')
p3.axhline(0.6031, color='g', linestyle='--')
plt.annotate('Testing accuracy', xy=(1e-4, 0.57), color='g')
plt.ylim(y_min, y_max)
#plt.show()
plt.savefig('SGD_modified_huber.png')
plt.close()
