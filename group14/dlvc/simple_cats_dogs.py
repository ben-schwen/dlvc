from dlvc.batches import BatchGenerator
from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
import dlvc.ops as ops
import numpy as np
from dlvc.models.simple import KNN
from dlvc.test import Accuracy

# 1. Load the training, validation, and test sets as individual PetsDatasets.


train_ds = PetsDataset('C:\\Users\\bschwendinger\\Documents\\github\\dlvc\\cifar-10-batches-py', Subset.TRAINING)
valid_ds = PetsDataset('C:\\Users\\bschwendinger\\Documents\\github\\dlvc\\cifar-10-batches-py', Subset.VALIDATION)
test_ds = PetsDataset('C:\\Users\\bschwendinger\\Documents\\github\\dlvc\\cifar-10-batches-py', Subset.TEST)

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

train_acc = []
valid_acc = []
ks = np.arange(1,21)

for i in ks:
    model = KNN(3072, train_ds.num_classes(), i)
    acc = Accuracy()

    model.train(train.data, train.label)

    acc.update(model.predict(train.data), train.label)
    train_acc.append(acc.accuracy())

    acc.update(model.predict(valid.data), valid.label)
    valid_acc.append(acc.accuracy())

best_k = ks[np.argmax(valid_acc)]

model = KNN(3072, train_ds.num_classes(), i)
model.train(train.data, train.label)
acc.update(model.predict(test.data), test.label)

test_acc = acc.accuracy()

# 5. Report the best parameters found and the corresponding validation accuracy.
# 6. Compute and report the accuracy on the test set with these parameters.


# PLOTTING
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({'k': ks, 'Training': train_acc, 'Validation': valid_acc})
df = df.set_index('k')

p1 = sns.lineplot(data=df, linewidth=2.5, dashes=False)
p1.set(xlabel = 'k', ylabel = 'Accuracy')
p1.axhline(test_acc, color='g', linestyle='--')
plt.annotate('Testing accuracy', xy=(2.5, 0.63), color='g')
plt.show()