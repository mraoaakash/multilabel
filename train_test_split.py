import os 
import shutil
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

df_train = pd.DataFrame(columns=['filename', 'Tumor','Stroma','Fat Cells','TILs','White Space' ])
df_test = pd.DataFrame(columns=['filename',  'Tumor','Stroma','Fat Cells','TILs','White Space' ])
num_img = len(os.listdir('/Users/mraoaakash/Documents/research/research-tnbc/Archive/multilabel_effnet/datasets/images'))
print(num_img)
random.seed(42)
random_train = random.sample(range(0, num_img-1), num_img//8)
# print(random_train)
images = os.listdir('/Users/mraoaakash/Documents/research/research-tnbc/Archive/multilabel_effnet/datasets/images')
print(len(images))
for i in random_train:
    image_name = images[i]
    label = [*image_name.split('_')[1].split('.')[0]]
    if not len(label) == 5:
        continue
    df_test = df_test.append({'filename': image_name, 'Tumor': label[0], 'Stroma': label[1], 'Fat Cells': label[2], 'TILs': label[3], 'White Space': label[4]}, ignore_index=True)
    print(label)
    images.remove(image_name)

for i in images:
    image_name = i
    label = [*image_name.split('_')[1].split('.')[0]]
    if not len(label) ==5:
        continue
    df_train = df_train.append({'filename': image_name, 'Tumor': label[0], 'Stroma': label[1], 'Fat Cells': label[2], 'TILs': label[3], 'White Space': label[4]}, ignore_index=True)
    print(label)


print(df_train.head())
print(len(df_train))
print(df_test.head())
print(len(df_test))

df_train.to_csv('/Users/mraoaakash/Documents/research/research-tnbc/Archive/multilabel_effnet/datasets/labels/labels_train.csv', index=False, header=False)
df_test.to_csv('/Users/mraoaakash/Documents/research/research-tnbc/Archive/multilabel_effnet/datasets/labels/labels_test.csv', index=False, header=False)