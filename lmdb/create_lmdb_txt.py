# Create a txt file of images and business ids.
import csv
from collections import defaultdict

yelp_picture_root = '/home/ubuntu/Yelp/data/'
yelp_csv_root = '/home/ubuntu/YelpVisionProject/'

def create_list_txt(csv_file, output_file):
     with open(csv_file) as list_csv:
        with open(output_file, 'w') as list_txt:
            reader = csv.DictReader(list_csv)
            writer = csv.writer(list_txt, delimiter =' ')
            for row in reader:
                writer.writerow([row['photo_id']+'.jpg', row['business_id']])

train_csv = yelp_csv_root + 'train_photo_to_biz_ids2.csv'
train_txt = yelp_picture_root + 'train.txt'
#create_list_txt(train_csv, train_txt)

val_csv = yelp_csv_root + 'validation_photo_to_biz_ids2.csv'
val_txt = yelp_picture_root + 'val.txt'
#create_list_txt(val_csv, val_txt)

import json

# writing
#json.dump(yourdict, open(filename, 'w'))

# reading
#yourdict = json.load(open(filename))

def create_test_list_txt(csv_file, output_file, split='test'):
    mapping = defaultdict(int)
    index = 1
    with open(csv_file) as list_csv:
        with open(output_file, 'w') as list_txt:
            reader = csv.DictReader(list_csv)
            writer = csv.writer(list_txt, delimiter =' ')
            for row in reader:
                b_id = mapping[row['business_id']]
                if b_id == 0:
                    b_id = index
                    mapping[row['business_id']] = index
                    index += 1
                writer.writerow([row['photo_id']+'.jpg', b_id])
            json.dump(mapping, open(yelp_picture_root + split + '_mapping.json', 'w'))

test_csv = yelp_picture_root + 'test_photo_to_biz.csv'
test_txt = yelp_picture_root + 'test.txt'
create_test_list_txt(test_csv, test_txt)

# For poster.
poster_csv = yelp_picture_root + 'poster_photo_to_biz_id.csv'
poster_txt = yelp_picture_root + 'poster.txt'
create_test_list_txt(poster_csv, poster_txt, split='poster')

