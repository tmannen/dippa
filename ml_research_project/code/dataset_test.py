%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
from dataloading import CARLADataset

fig = plt.figure()

dataset = CARLADataset("../data")

for i in range(len(dataset)):
    image, steer = dataset[i]

    print(i, image.shape, steer)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(image)

    if i == 3:
        plt.show()
        break