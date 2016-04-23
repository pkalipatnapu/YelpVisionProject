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

testInput = np.array([np.array([0, 1]),np.array([5]),np.array([4,5]),np.array([7,8]),np.array([1,2,3])]);
threshold = [0.3, 0.3, 0.3, -1, 0.5, 0.1, 0.3, 0.4, -1];
count = testInput.size;
predict = np.zeros(9);

for i in range(0, count):
	for j in range(0, testInput[i].size):
		predict[testInput[i][j]] += 1;

testOutput = np.array([]);

for i in range(0,9):
	if predict[i]/count > threshold[i]:
		testOutput = np.insert(testOutput,testOutput.size, i);

print testOutput

