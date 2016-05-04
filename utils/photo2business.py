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

class p2b:
    def __init__(self, thresholds = [0.3, 0.3, 0.3, 0.1, 0.5, 0.1, 0.3, 0.4, 0.1]):
        self.thresholds = thresholds

    def convert_mean(self, business_mean):
        business_labels = np.zeros(9);

        for i in range(9):
    	    if business_mean[i] > self.thresholds[i]:
	            business_labels[i] = 1

        return business_labels


    def convert(self, image_values):
        mean_value = np.mean(image_values, axis=0)
        return self.convert_mean(mean_value)


if __name__ == '__main__':
    img_labels = [
              	  [1, 1, 1, 0, 1, 0, 0, 0, 1],
                  [0 ,0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 1, 0, 0, 0]
    ]
    converter = p2b()
    print converter.convert(img_labels)
