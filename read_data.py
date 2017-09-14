"""
File is stored in bson format. It is supported by pymongo package.
From https://www.kaggle.com/inversion/processing-bson-files.
"""
import csv
import bson
import matplotlib.pylab as plt
from skimage.data import imread
import io
import pandas as pd
import numpy as np


DATA_FILE_NAME = '/Users/Sophie/Documents/cdiscount/train_example.bson'
CATEGORY_NAMES_FILE_NAME = '/Users/Sophie/Documents/cdiscount/category_names.csv'


class Category(object):
    def __init__(self):
        self.mapping = dict()
        # Read the content of the csv file that defines the mapping from category id to name.
        with open(CATEGORY_NAMES_FILE_NAME, newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')
            for row in reader:
                category_id = int(row['category_id'])
                del row['category_id']
                self.mapping[category_id] = row

    def category_to_name(self, category_id):
        """
        :param category_id: A Python int.
        :return: A Python str that represents the name of the category_id.
        """
        if category_id in self.mapping:
            return self.mapping[category_id]
        else:
            return None


if __name__ == '__main__':
    # Parse the mappings from category_id to category names in three levels.
    category = Category()
    print('{}: {}'.format(1000012776, category.category_to_name(1000012776)))

    data = bson.decode_file_iter(open(DATA_FILE_NAME, 'rb'))

    prod_to_category = dict()

    for c, d in enumerate(data):
        product_id = d['_id']
        category_id = d['category_id']  # This won't be in Test data
        prod_to_category[product_id] = category_id
        for e, imgs in enumerate(d['imgs']):
            picture = imread(io.BytesIO(imgs['picture']))
            # do something with the picture, etc
            """
            # Show a picture
            print(category_id)
            plt.imshow(picture)

            plt.show()
            """

    prod_to_category_pd = pd.DataFrame.from_dict(prod_to_category, orient='index')
    prod_to_category_pd.index.name = '_id'
    prod_to_category_pd.rename(columns={0: 'category_id'}, inplace=True)
    # Show pandas data header
    prod_to_category_pd.head()
