import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 2 # number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = '.../datasets/LU-data/train'
# validation images and XML files directory
VALID_DIR = '.../datasets/LU-data/valid'
# classes: 0 index is reserved for background
CLASSES = [
    'background', 'warning', 'prohibitory', 'instruction', 'mandatory', 'location_direction'
]
NUM_CLASSES = 6

# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '/outputs'
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs

def main():
    pass


if __name__ == '__main__':
    main()
