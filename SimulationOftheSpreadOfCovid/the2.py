#
# WRITE YOUR CODE HERE AND SEND ONLY THIS FILE TO US.
#
# DO NOT DEFINE get_data() here. Check before submitting

import math
import random
from evaluator import *    # get_data() will come from this import
data = get_data()
m = data[0]
n = data[1]
dist = data[2]
c_k = data[3]
c_l = data[4]
mu = data[5]
individuals = data[6]
def new_move():
	# this function takes our universal data then, it determines individuals motion with respect to their probabilities.
	# Finally, it updates individuals infection status with the helper functions.
	global data
	global m
	global n
	global dist
	global mu
	global individuals
	green = (1 / 2) * mu
	yellow = (1 / 8) * mu
	blue = (1 / 2) * (1 - mu - (mu ** 2))
	purple = (2 / 5) * (mu ** 2)
	gray = (1 / 5) * (mu ** 2)
	new_individuals = []
	for individual in individuals:
		position = list(individual[0])
		last_move_updated = individual[1]
		# Firstly, we will check which kind of movement the individual has done last time.
		# According to its last move, we will calculate the probabilities.
		# After "random.choices" selects any motion with respect to their probabilities,
		# We update its new position and its last move value after checking some limitations.
		if individual[1] == 0:
			color_list0 = ["green", "yellow_forward_right", "blue_right", "purple_backward_right", "gray", "purple_backward_left", "blue_left", "yellow_forward_left"]
			chosen = random.choices(color_list0, weights=(green, yellow, blue, purple, gray, purple, blue, yellow))
			if chosen[0] == "blue_left":
				position[0] += 1
				last_move_updated = 6
			elif chosen[0] == "blue_right":
				position[0] += -1
				last_move_updated = 2
			elif chosen[0] == "yellow_forward_left":
				position[1] += 1
				position[0] += 1
				last_move_updated = 7
			elif chosen[0] == "yellow_forward_right":
				position[1] += 1
				position[0] += -1
				last_move_updated = 1
			elif chosen[0] == "purple_backward_left":
				position[0] += 1
				position[1] += -1
				last_move_updated = 5
			elif chosen[0] == "purple_backward_right":
				position[0] += -1
				position[1] += -1
				last_move_updated = 3
			elif chosen[0] == "green":
				position[1] += 1
				last_move_updated = 0
			elif chosen[0] == "gray":
				position[1] += -1
				last_move_updated = 4
		elif individual[1] == 2:
			color_list2 = ["green", "y_f_r", "b_r", "p_b_r", "gray", "p_b_l", "b_l", "y_f_l"]
			chosen = random.choices(color_list2, weights=(green, yellow, blue, purple, gray, purple, blue, yellow))
			if chosen[0] == "b_r":
				position[1] += -1
				last_move_updated = 4
			elif chosen[0] == "b_l":
				position[1] += 1
				last_move_updated = 0
			elif chosen[0] == "y_f_r":
				position[1] += -1
				position[0] += -1
				last_move_updated = 3
			elif chosen[0] == "y_f_l":
				position[1] += 1
				position[0] += -1
				last_move_updated = 1
			elif chosen[0] == "p_b_r":
				position[0] += 1
				position[1] += -1
				last_move_updated = 5
			elif chosen[0] == "p_b_l":
				position[0] += 1
				position[1] += 1
				last_move_updated = 7
			elif chosen[0] == "green":
				position[0] += -1
				last_move_updated = 2
			elif chosen[0] == "gray":
				position[0] += 1
				last_move_updated = 6
		elif individual[1] == 4:
			color_list4 = ["green", "y_f_r", "b_r", "p_b_r", "gray", "p_b_l", "b_l", "y_f_l"]
			chosen = random.choices(color_list4, weights=(green, yellow, blue, purple, gray, purple, blue, yellow))
			if chosen[0] == "green":
				position[1] += -1
				last_move_updated = 4
			elif chosen[0] == "y_f_r":
				position[1] += -1
				position[0] += 1
				last_move_updated = 5
			elif chosen[0] == "y_f_l":
				position[1] += -1
				position[0] += -1
				last_move_updated = 3
			elif chosen[0] == "b_r":
				position[0] += 1
				last_move_updated = 6
			elif chosen[0] == "b_l":
				position[0] += -1
				last_move_updated = 2
			elif chosen[0] == "p_b_r":
				position[0] += 1
				position[1] += 1
				last_move_updated = 7
			elif chosen[0] == "p_b_l":
				position[0] += -1
				position[1] += 1
				last_move_updated = 1
			elif chosen[0] == "gray":
				position[1] += 1
				last_move_updated = 0
		elif individual[1] == 6:
			color_list6 = ["green", "y_f_r", "b_r", "p_b_r", "gray", "p_b_l", "b_l", "y_f_l"]
			chosen = random.choices(color_list6, weights=(green, yellow, blue, purple, gray, purple, blue, yellow))
			if chosen[0] == "b_l":
				position[1] += -1
				last_move_updated = 4
			elif chosen[0] == "b_r":
				position[1] += 1
				last_move_updated = 0
			elif chosen[0] == "p_b_l":
				position[0] += -1
				position[1] += -1
				last_move_updated = 3
			elif chosen[0] == "p_b_r":
				position[0] += -1
				position[1] += 1
				last_move_updated = 1
			elif chosen[0] == "y_f_r":
				position[0] += 1
				position[1] += 1
				last_move_updated = 7
			elif chosen[0] == "y_f_l":
				position[0] += 1
				position[1] += -1
				last_move_updated = 5
			elif chosen[0] == "green":
				position[0] += 1
				last_move_updated = 6
			elif chosen[0] == "gray":
				position[0] += -1
				last_move_updated = 2
		elif individual[1] == 1:
			color_list1 = ["green", "y_f_r", "b_r", "p_b_r", "gray", "p_b_l", "b_l", "y_f_l"]
			chosen = random.choices(color_list1, weights=(green, yellow, blue, purple, gray, purple, blue, yellow))
			if chosen[0] == "green":
				position[1] += 1
				position[0] += -1
				last_move_updated = 1
			elif chosen[0] == "y_f_r":
				position[0] += -1
				last_move_updated = 2
			elif chosen[0] == "y_f_l":
				position[1] += 1
				last_move_updated = 0
			elif chosen[0] == "p_b_r":
				position[1] += -1
				last_move_updated = 4
			elif chosen[0] == "p_b_l":
				position[0] += 1
				last_move_updated = 6
			elif chosen[0] == "b_r":
				position[0] += -1
				position[1] += -1
				last_move_updated = 3
			elif chosen[0] == "b_l":
				position[0] += 1
				position[1] += 1
				last_move_updated = 7
			elif chosen[0] == "gray":
				position[0] += 1
				position[1] += -1
				last_move_updated = 5
		elif individual[1] == 3:
			color_list3 = ["green", "y_f_r", "b_r", "p_b_r", "gray", "p_b_l", "b_l", "y_f_l"]
			chosen = random.choices(color_list3, weights=(green, yellow, blue, purple, gray, purple, blue, yellow))
			if chosen[0] == "green":
				position[1] += -1
				position[0] += -1
				last_move_updated = 3
			elif chosen[0] == "y_f_r":
				position[1] += -1
				last_move_updated = 4
			elif chosen[0] == "y_f_l":
				position[0] += -1
				last_move_updated = 2
			elif chosen[0] == "p_b_l":
				position[1] += 1
				last_move_updated = 0
			elif chosen[0] == "p_b_r":
				position[0] += 1
				last_move_updated = 6
			elif chosen[0] == "b_r":
				position[0] += 1
				position[1] += -1
				last_move_updated = 5
			elif chosen[0] == "b_l":
				position[0] += -1
				position[1] += 1
				last_move_updated = 1
			elif chosen[0] == "gray":
				position[0] += 1
				position[1] += 1
				last_move_updated = 7
		elif individual[1] == 5:
			color_list5 = ["green", "y_f_r", "b_r", "p_b_r", "gray", "p_b_l", "b_l", "y_f_l"]
			chosen = random.choices(color_list5, weights=(green, yellow, blue, purple, gray, purple, blue, yellow))
			if chosen[0] == "green":
				position[1] += -1
				position[0] += 1
				last_move_updated = 5
			elif chosen[0] == "y_f_l":
				position[1] += -1
				last_move_updated = 4
			elif chosen[0] == "y_f_r":
				position[0] += 1
				last_move_updated = 6
			elif chosen[0] == "p_b_l":
				position[0] += -1
				last_move_updated = 2
			elif chosen[0] == "p_b_r":
				position[1] += 1
				last_move_updated = 0
			elif chosen[0] == "b_l":
				position[0] += -1
				position[1] += -1
				last_move_updated = 3
			elif chosen[0] == "b_r":
				position[0] += 1
				position[1] += 1
				last_move_updated = 7
			elif chosen[0] == "gray":
				position[0] += -1
				position[1] += 1
				last_move_updated = 1
		elif individual[1] == 7:
			color_list7 = ["green", "y_f", "b_r", "p_b_r", "gray", "p_b_l", "b_l", "y_f_l"]
			chosen = random.choices(color_list7, weights=[green, yellow, blue, purple, gray, purple, blue, yellow])
			if chosen[0] == "green":
				position[1] += 1
				position[0] += 1
				last_move_updated = 7
			elif chosen[0] == "y_f_l":
				position[0] += 1
				last_move_updated = 6
			elif chosen[0] == "y_f":
				position[1] += 1
				last_move_updated = 0
			elif chosen[0] == "p_b_l":
				position[1] += -1
				last_move_updated = 4
			elif chosen[0] == "p_b_r":
				position[0] += -1
				last_move_updated = 2
			elif chosen[0] == "b_r":
				position[0] += -1
				position[1] += 1
				last_move_updated = 1
			elif chosen[0] == "b_l":
				position[0] += 1
				position[1] += -1
				last_move_updated = 5
			elif chosen[0] == "gray":
				position[0] += -1
				position[1] += -1
				last_move_updated = 3
		# To prevent them from going out of the universe
		# We need to restrict their motion area with the magnitudes of the given universe.
		if 0 <= position[0] < n and 0 <= position[1] < m:
			# checker will help us to avoid conflictions (preoccupied positions).
			checker = any((position[0],position[1]) in e for e in new_individuals) # I use list comprension here because universal state is nested.
			#if the position is not the same as any other position, we will update its position.
			if not checker:
				new_individual = [(position[0], position[1]), last_move_updated, individual[2], individual[3]]
			#Otherwise, it will skip this tour.
			else:
				new_individual = [individual[0], individual[1], individual[2], individual[3]]
		# If it wants to go out, we will skip this tour and make it wait too.
		else:
			new_individual = [individual[0], individual[1], individual[2], individual[3]]
		new_individuals.append(new_individual)
	# we need to deepcopy the list to avoid individual who just gets infected to transmit the virus at the same time zone.
	copy_list = deepcopy(new_individuals)
	# update_infection_status helps us to update infection status of individuals.
	updated = update_infection_status(new_individuals,copy_list)
	# We also need to update "individuals", which comes from get_data(), in order to reuse updated data again and again.
	individuals = updated
	return updated
def combinations(mylist):
	# this function helps us to find the combinations of the given number of people which will provide us with the chance to calculate probabilities of infection.
	if len(mylist) == 0:
		return [[]]
	result = []
	for i in combinations(mylist[1:]):
		result += [i, i+[mylist[0]]]
	return result
def update_infection_status(updated_individuals,copy_list):
	combs = combinations(range(len(copy_list)))
	dual_combs = []
	# We need dual combinations to calculate infection probabilities.
	for comb in combs:
		if len(comb) == 2:
			dual_combs.append(comb)
	# To calculate the distance between two individuals, we can use eukleides theorem
	for i, j in dual_combs:
		# we need to convert tuple into list because tuples are immutable.
		list_dist_i = list(copy_list[i][0])
		list_dist_j = list(copy_list[j][0])
		c_d = ((((list_dist_i[0] - list_dist_j[0]) ** 2) + ((list_dist_i[1] - list_dist_j[1]) ** 2)) ** (1 / 2)) #calculated_distance
		# we need to define copy_list because once an individual gets infected, he/she should not transmit the virus to the others at that step.
		if (updated_individuals[i][3] == "infected" and updated_individuals[j][3] == "notinfected"):
			if control_mask_status_and_distance(updated_individuals[i][2],updated_individuals[j][2],c_d)[0] == "get_infected":
				copy_list[j][3] = "infected"
		if (updated_individuals[i][3] == "notinfected" and updated_individuals[j][3] == "infected"):
			if control_mask_status_and_distance(updated_individuals[i][2],updated_individuals[j][2],c_d)[0] == "get_infected":
				copy_list[i][3] = "infected"
	return copy_list
def control_mask_status_and_distance(mask_status1,mask_status2,c_d):
	global dist
	global c_l
	global c_k
	# This function will check whether individuals are as close to each other as threshold distance
	# and if so, it will check mask status, accordingly will calculate the infection probabilities.
	infection_status_list = ["get_infected", "not_infected"]
	if c_d <= dist:
		if mask_status1 == "notmasked" and mask_status2== "notmasked":
			p = min(1, c_k / (c_d ** 2))
			p_chosen = random.choices(infection_status_list, weights=(p, 1-p))
			return p_chosen
		elif mask_status1 == "masked" and mask_status2 == "notmasked":
			raw_p = min(1, c_k / (c_d ** 2))
			p = raw_p / c_l
			p_chosen = random.choices(infection_status_list, weights=(p, 1 - p))
			return p_chosen
		elif mask_status1 == "notmasked" and mask_status2 == "masked":
			raw_p = min(1, c_k / (c_d ** 2))
			p = raw_p / c_l
			p_chosen = random.choices(infection_status_list, weights=(p, 1 - p))
			return p_chosen
		elif mask_status1 == "masked" and mask_status2 == "masked":
			raw_p = min(1, c_k / (c_d ** 2))
			p = raw_p / (c_l ** 2)
			p_chosen = random.choices(infection_status_list, weights=(p, 1 - p))
			return p_chosen
	else:
		return [infection_status_list[1]]
def deepcopy(mylist):
	# we can deepcopy recursively by checking elements types in each step and accordingly append it to the result.
	result = []
	for elem in mylist:
		if isinstance(elem,list):
			result.append(deepcopy(elem))
		else:
			result.append(elem)
	return result
