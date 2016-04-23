import numpy as np
import csv
import random
from sets import Set

vali_busId_index = Set([]);
total = 2000 * 0.2;

while len(vali_busId_index) < total :
    vali_busId_index.add(random.randint(1,2000));

vali_busId = Set([]);

f_vali = open('validation2.csv','w');
f_train = open('train2.csv','w');

f_vali.write('business_id,labels\n')
f_train.write('business_id,labels\n')

lineNum = 1;
with open('train.csv') as labelfile:
    reader = csv.DictReader(labelfile)
    for row in reader:
        if lineNum in vali_busId_index :
            vali_busId.add(row['business_id']);
            f_vali.write(row['business_id'] + "," + row['labels'] + '\n');
        else :
            f_train.write(row['business_id'] + "," + row['labels'] + '\n');
        lineNum += 1;

f_vali.close();
f_train.close();

f_vali_mapping = open('validation_photo_to_biz_ids2.csv','w');
f_train_mapping = open('train_photo_to_biz_ids2.csv','w');

f_vali_mapping.write('photo_id,business_id\n')
f_train_mapping.write('photo_id,business_id\n')

with open('train_photo_to_biz_ids.csv') as labelfile:
    reader = csv.DictReader(labelfile)
    for row in reader:
        if row['business_id'] in vali_busId :
            f_vali_mapping.write(row['photo_id'] + "," + row['business_id'] + '\n');
        else :
            f_train_mapping.write(row['photo_id'] + "," + row['business_id'] + '\n');

f_vali_mapping.close();
f_train_mapping.close();


# with open('train_photo_to_biz_ids.csv') as mappingfile:
#     reader = csv.DictReader(mappingfile)
#     for row in reader:
#

#
# f = open('predictions.csv','w')
# f.write('Id,Category\n')
# for (i, e) in enumerate(predictions):
#     f.write('%d,%d\n' % (i+1, e))
# f.close()
