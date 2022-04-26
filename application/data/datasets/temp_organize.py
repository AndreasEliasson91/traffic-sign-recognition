import os
import uuid

import cv2

from application.data.data_utils import LETTERS


def main():
    # labels = [
    #     'warning_signs',
    #     'priority_signs',
    #     'prohibitory_signs',
    #     'mandatory_signs',
    #     'instruction_signs',
    #     'location_directions_signs',
    #     'location_public_signs',
    #     'location_service_signs',
    #     'location_POI_signs',
    #     'information_signs',
    #     'policemen_signs',
    #     'symbols',
    #     'additional_boards',
    #     'other',
    #     'uncategorized'
    # ]
    #
    # image_path = os.path.join('training_data', 'wikipedia')
    # if not os.path.exists(image_path):
    #     if os.name == 'posix':  # Mac, Linux
    #         os.system(f'mkdir -p {image_path}')
    #     elif os.name == 'nt':  # Windows
    #         os.system(f'mkdir {image_path}')
    #
    # for label in labels:
    #     path = os.path.join(image_path, label)
    #     if not os.path.exists(path):
    #         os.system(f'mkdir {path}')
    # #
    # # for i, label in enumerate(labels):
    # #     for file in os.listdir('./training_data/wikipedia_signs/'):
    # #         if LETTERS[i] in file:
    # #             os.rename(f'./training_data/wikipedia_signs/{file}', os.path.join(image_path, label, f'{label}-{uuid.uuid1()}.jpg'))
    #
    # with open('./training_data/wikipedia/wikipedia_annotations.csv', 'w') as out_file:
    #     out_file.write('img_name,category,classification,sign,description\n')
    #     for i in range(len(labels)):
    #         for file in os.listdir(f'./training_data/wikipedia/{labels[i]}'):
    #             if os.path.isfile(os.path.join(f'./training_data/wikipedia/{labels[i]}', file)):
    #                 out_file.write(f'{file},{labels[i]},{LETTERS[i]},,\n')

    img = cv2.imread('./training_data/wikipedia/warning_signs-e1bc994e-c228-11ec-9628-18cc1895e0b0.jpg')
    cv2.imshow('Image', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
