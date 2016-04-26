# 0: good_for_lunch
# 1: good_for_dinner
# 2: takes_reservations
# 3: outdoor_seating
# 4: restaurant_is_expensive
# 5: has_alcohol
# 6: has_table_service
# 7: ambience_is_classy
# 8: good_for_kids
#logic : use fixed threshold for 0, 1, 2, 4, 5, 6, 7, 8
# as long as one picture predicts that this place is 3 / 8 then 3 / 8 should be assigned to this business

import numpy as np

def convert (image_values):
    threshold = [0.3, 0.3, 0.3, 0.1, 0.5, 0.1, 0.3, 0.4, 0.1]
    mean_value = np.mean(image_values, axis=0)
    max_value = np.max(image_values, axis=0)

    business_labels = np.zeros(9);

    for i in range(9):
	if mean_value[i] > threshold[i]:
	    business_labels[i] = 1

    return business_labels

if __name__ == '__main__':
    img_labels = [
              	  [1, 1, 1, 0, 1, 0, 0, 0, 1],
                  [0 ,0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 1, 0, 0, 0]
    ]
    print convert(img_labels)
