import os
import pandas as pd 
import shutil 
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

df_2014 = pd.read_csv('data/CROHME/2014/caption.txt', sep='\t', header=None, names=['filenames', 'captions'])
df_2016 = pd.read_csv('data/CROHME/2016/caption.txt', sep='\t', header=None, names=['filenames', 'captions'])
df_2019 = pd.read_csv('data/CROHME/2019/caption.txt', sep='\t', header=None, names=['filenames', 'captions'])
df_train = pd.read_csv('data/CROHME/train/caption.txt', sep='\t', header=None, names=['filenames', 'captions'])

data = pd.concat(
    [
        df_2014,
        df_2016,
        df_2019,
        df_train
    ]
)

# First, split off 10% of the data to train and test sets
train, test = train_test_split(data, test_size=0.1, random_state=42)

# Second, split off 10% of the training data to train and validation sets 
train, val = train_test_split(train, test_size=0.1, random_state=42)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Validation shape:", val.shape)

train_filenames = train['filenames'].tolist()
train_captions = train['captions'].tolist()

test_filenames = test['filenames'].tolist()
test_captions = test['captions'].tolist()

val_filenames = val['filenames'].tolist()
val_captions = val['captions'].tolist()

# Extract captions.txt for each split  
with open('data/CROHME_splitted/train/caption.txt', 'w', encoding='utf-8') as f:
    for filename, caption in zip(train_filenames, train_captions):
        f.write(f"{filename}\t{caption}\n")

with open('data/CROHME_splitted/test/caption.txt', 'w', encoding='utf-8') as f:
    for filename, caption in zip(test_filenames, test_captions):
        f.write(f"{filename}\t{caption}\n")

with open('data/CROHME_splitted/val/caption.txt', 'w', encoding='utf-8') as f:
    for filename, caption in zip(val_filenames, val_captions):
        f.write(f"{filename}\t{caption}\n")


IMAGES_DIR = 'data/images'
TRAIN_DIR = 'data/CROHME_splitted/train/images'
TEST_DIR = 'data/CROHME_splitted/test/images'
VAL_DIR = 'data/CROHME_splitted/val/images'

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)        
os.makedirs(VAL_DIR, exist_ok=True)

for train_filename in tqdm(train_filenames, desc="Copying train images"):
    src = os.path.join(IMAGES_DIR, train_filename) + '.bmp'  # Ensure the file extension is correct
    dst = os.path.join(TRAIN_DIR, train_filename) + '.bmp'
    shutil.copy(src, dst)
    
for test_filename in tqdm(test_filenames, desc="Copying test images"):
    src = os.path.join(IMAGES_DIR, test_filename) + '.bmp'
    dst = os.path.join(TEST_DIR, test_filename) + '.bmp'
    shutil.copy(src, dst)

for val_filename in tqdm(val_filenames, desc="Copying validation images"):
    src = os.path.join(IMAGES_DIR, val_filename) + '.bmp'
    dst = os.path.join(VAL_DIR, val_filename) + '.bmp'
    shutil.copy(src, dst)