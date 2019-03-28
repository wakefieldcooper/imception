import matplotlib.pyplot as plt
import os
import json


with open('config.json') as f:
    conf = json.load(f)


def train_samples():
    path_real, dirs_real, files_real = next(os.walk(conf['train_path'] + '/real'))
    path_fake, dirs_fake, files_fake = next(os.walk(conf['train_path'] + '/fake'))

    file_count_real = len(files_real)
    file_count_fake = len(files_fake)
    print(file_count_fake+file_count_real)
    return file_count_fake+file_count_real, file_count_real, file_count_fake

def validation_samples():
    path_real, dirs_real, files_real = next(os.walk(conf['validation_path'] + '/real'))
    path_fake, dirs_fake, files_fake = next(os.walk(conf['validation_path'] + '/fake'))

    file_count_real = len(files_real)
    file_count_fake = len(files_fake)
    print(file_count_fake+file_count_real)
    return file_count_fake+file_count_real, file_count_real, file_count_fake
def vis_dataset():

    fig = plt.figure()
    x = ['real', 'fake']
    y = [file_calculations()[1], file_calculations()[2]]

    plt.bar(x, y)
    plt.title('Balance of training dataset')
    plt.xlabel('Labels')
    plt.ylabel('Number of images')
    plt.show()
    fig.savefig(conf['directory'] + '/training_visualisation.jpg')

# vis_dataset()
