import os
from collections import OrderedDict
import json
from dhtnet.paths import DHTNet_raw_data

def gen_json():
    base_folder = os.path.join(DHTNet_raw_data, 'Task099_LiTS')

    tr_image_folder = os.path.join(base_folder, "imagesTr")
    ts_image_folder = os.path.join(base_folder, 'imagesTs')
    tr_label_folder = os.path.join(base_folder, 'labelsTr')
    ts_label_folder = os.path.join(base_folder, 'labelsTs')
    tr_images = os.listdir(tr_image_folder)
    tr_labels = os.listdir(tr_label_folder)
    ts_images = os.listdir(ts_image_folder)
    # ts_labels = os.listdir(ts_label_folder)

    json_dict = OrderedDict()
    json_dict['name'] = 'LiTS'
    json_dict['description'] = 'LiTS'
    json_dict['reference'] = 'LiTS'
    json_dict['release'] = '0.0'
    json_dict['tensorImageSize'] = '4D'
    json_dict['modality'] = {
        "0": "CT"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "liver",
        "2": "tumor",
    }
    json_dict['numTraining'] = len(tr_images)
    json_dict['numTest'] = len(ts_images)
    json_dict['training'] = list()
    json_dict['test'] = list()
    for image in tr_images:
        case_id = os.path.split(image)[-1][:-12]
        case_virtul_file_image = "./imagesTr/{}.nii.gz".format(case_id)
        case_virtul_file_label = "./labelsTr/{}.nii.gz".format(case_id)
        json_dict['training'].append({"image": case_virtul_file_image, "label": case_virtul_file_label})
    for image in ts_images:
        case_id = os.path.split(image)[-1][:-12]
        case_virtul_file = "./imagesTs/{}.nii.gz".format(case_id)
        json_dict['test'].append(case_virtul_file)

    print(json_dict)

    json_file = os.path.join(base_folder, "dataset.json")
    with open(json_file, "w+") as fp:
        json.dump(json_dict, fp)


if __name__ == "__main__":
    gen_json()