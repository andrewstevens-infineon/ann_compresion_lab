# %%
#
from email.headerregistry import HeaderRegistry
import tensorflow as tf

from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn import preprocessing
import re

here = os.path.abspath(os.path.dirname(__file__))

class userDataset(object):
    def __init__(self):
        self.batch_size = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.data_max = []
        collect_data_path = here

        self.train_location_combined = os.path.join(collect_data_path, "combined_train.npz")
        self.test_location_combined = os.path.join(collect_data_path, "combined_test.npz")

    def preprocess_data(self, data):
        data = np.abs(data)
        mic_0_max = np.max(np.abs(data[:, 0]))
        mic_1_max = np.max(np.abs(data[:, 1]))
        mic_2_max = np.max(np.abs(data[:, 2]))
        mic_3_max = np.max(np.abs(data[:, 3]))

        data[:, 0] = data[:, 0] / (mic_0_max + 1)
        data[:, 1] = data[:, 1] / (mic_1_max + 1)
        data[:, 2] = data[:, 2] / (mic_2_max + 1)
        data[:, 3] = data[:, 3] / (mic_3_max + 1)
        scaled_data = np.int8(data * 120)
        scaled_data = scaled_data.astype(np.float32)
        return scaled_data

    def prep_data(self, locations, train=False):
        x_raw = []
        x = []
        y_raw = []
        y = []
        count = 0
        if len(locations)==1 and os.path.splitext(locations[0])[1] == ".npz":
            saved = np.load(locations[0])
            x_raw = saved['x']
            y_raw = saved['y']
            count = x_raw.shape[0]
        else:      
            for location in locations:
                for file in os.listdir(location):
                    count += 1
                    label_x_txt = file.split("_")[0]
                    label_y_txt = file.split("_")[1].split("_")[0]
                    label_x = int(label_x_txt)
                    label_y = int(label_y_txt)

                    # keep labels in cm
                    label_x /= 10
                    label_y /= 10
                    data = np.genfromtxt(os.path.join(location, file), delimiter=",")
                    x_raw.append(np.array(data))
                    y_raw.append(np.array([label_x, label_y]))
  
            x_raw=np.array(x_raw)
            y_raw=np.array(y_raw)
            save_location = self.train_location_combined  if train else  self.test_location_combined
            np.savez(save_location,x=x_raw, y=y_raw)

        for i in range(x_raw.shape[0]):
            data = x_raw[i]
            label = y_raw[i]
            processed_data = self.preprocess_data(data)
            processed_data = processed_data.T
            processed_data = processed_data.reshape(4, 1000, 1)
            x.append(processed_data)
            y.append(label)
            if train:
                random_data = np.random.rand(data.shape[0], data.shape[1])
                augmented_data = data + random_data
                augmented_processed_data = self.preprocess_data(augmented_data)
                augmented_processed_data = augmented_processed_data.T
                augmented_processed_data = augmented_processed_data.reshape(
                    4, 1000, 1
                )
                x.append(augmented_processed_data)
                y.append(label)

        print(f"processed {x_raw.shape[0]} samples")

        return np.array(x), np.array(y)

    def get_datasets(self, set="Train"):
        # Load datasets from locals just first time
        if self.X_train is None:
            self.__load_data()
        # Return desired set
        if set == "Train":
            return self.X_train, self.y_train
        elif set == "Test":
            return self.X_test, self.y_test

    def __load_data(self):

        train_locations = [self.train_location_combined]
        test_locations = [self.test_location_combined]
        
        self.X_train, self.y_train = self.prep_data(train_locations, train=True)
        self.X_test, self.y_test = self.prep_data(test_locations)


# %%
if __name__ == "__main__":
    c = userDataset()
    x, y = c.get_datasets("Test")
    print(x.shape)
    # print(c.X[0, -1, :])
    # print(np.max(np.abs(x)))

#%%
