import json
import pandas as pd
from torch.utils.data import DataLoader
from .classes.sampleGetter import SampleGetter

def load_coco_data(dir, meta_filename, x_key='images', y_key='annotations', class_id='category_id', merge_on='id', drop_columns=['image_id']):

    with open(dir + meta_filename, 'r') as file:
        meta_data = json.load(file)

    images = pd.DataFrame(meta_data[x_key])
    annotations = pd.DataFrame(meta_data[y_key])
    for col in drop_columns:
        annotations.drop(columns=col)
    train = images.merge(annotations, on=merge_on)
    nb_classes = len(train[class_id].value_counts())

    return train, nb_classes


def create_data_loader(dir, df, x_key, y_key, tr, batch=16, limit=None):

    nb_classes = len(df[y_key].value_counts())
    if limit is None:
        data = df
    else:
        data = df[0:limit]
    
    x_train, y_train = data[x_key].values, data[y_key].values
    return DataLoader(SampleGetter(dir, x_train, y_train, tr), batch_size=batch, shuffle=True), nb_classes