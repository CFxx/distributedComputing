import json
import pandas as pd

def load_coco_data(dir, meta_filename, merge_on, drop_columns=[]):

    with open(dir + meta_filename, 'r') as file:
        meta_data = json.load(file)

    images = pd.DataFrame(meta_data['images'])
    annotations = pd.DataFrame(meta_data['annotations'])
    for col in drop_columns:
        annotations.drop(columns=col) # image_id
    train = images.merge(annotations, on=merge_on) # id

    return train