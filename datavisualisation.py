import matplotlib.pyplot as plt
import os
import json


with open('config.json') as f:
    conf = json.load(f)
def vis_dataset():
    path_real, dirs_real, files_real = next(os.walk("C:/Users/enqui/AppData/Local/Programs/Python/Python36/Thesis/repo/imception/new curated smaller/training/real"))
    path_fake, dirs_fake, files_fake = next(os.walk("C:/Users/enqui/AppData/Local/Programs/Python/Python36/Thesis/repo/imception/new curated smaller/training/fake"))

    file_count_real = len(files_real)
    file_count_fake = len(files_fake)
    fig = plt.figure()
    x = ['real', 'fake']
    y = [file_count_real, file_count_fake]

    plt.bar(x, y)
    plt.title('Balance of training dataset')
    plt.xlabel('Labels')
    plt.ylabel('Number of images')
    plt.show()
    fig.savefig(conf['directory'] + '/training_visualisation.jpg')