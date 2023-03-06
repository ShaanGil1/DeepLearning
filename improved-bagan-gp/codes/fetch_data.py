# %% --------------------------------------- Load Packages -------------------------------------------------------------
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image

# %% --------------------------------------- Data Prep -----------------------------------------------------------------
# Read data
DIR = '/content/skewed/data/'

train_dir = DIR + 'Training/'
test_dir = DIR + 'Testing/'

labels = sorted([label for label in os.listdir(train_dir)])
print(labels)
le = LabelEncoder()
le.fit(labels)

xlsts = []
ylsts = []

for data_dir in [train_dir, test_dir]:
    xdata = []
    ydata = []
    for l in labels:
        image_path = data_dir + l + '/'
        for img_dir in os.listdir(image_path):
            #img should have shape of (128,128,1)
            img = np.expand_dims(np.array(Image.open(image_path + img_dir)), -1)
            xdata.append(img)
            ydata.append(le.transform([l])[0])
    xlsts.append(np.array(xdata))
    ylsts.append(np.array(ydata))

x_train, x_val, y_train, y_val = xlsts[0], xlsts[1], ylsts[0], ylsts[1]

#shuffle data
SEED = 42
x_train, y_train = shuffle(x_train, y_train, random_state=SEED)
x_val, y_val = shuffle(x_val, y_val, random_state=SEED)

print(x_train.shape, x_val.shape)
'''
train = [f for f in os.listdir(DIR)]
train_sorted = sorted(train, key=lambda x: int(x[5:-4]))
imgs = []
texts = []
resize_to = 4
for f in train_sorted:
    if f[-3:] == 'png':
        imgs.append(cv2.resize(cv2.imread(DIR + f), (resize_to, resize_to)))
    else:
        texts.append(open(DIR + f).read())

imgs = np.array(imgs)
texts = np.array(texts)

le = LabelEncoder()
le.fit(["red blood cell", "ring", "schizont", "trophozoite"])
labels = le.transform(texts)

# Splitting
SEED = 42
x_train, x_val, y_train, y_val = train_test_split(imgs, labels,
                                                  random_state=SEED,
                                                  test_size=0.2,
                                                  stratify=labels)
'''




# %% --------------------------------------- Save as .npy --------------------------------------------------------------
# Save
save_dir = '/content/drive/MyDrive/DL Project/improved-bagan-gp/codes/'
np.save(save_dir + "x_train.npy", x_train); np.save(save_dir + "y_train.npy", y_train)
np.save(save_dir + "x_val.npy", x_val); np.save(save_dir + "y_val.npy", y_val)