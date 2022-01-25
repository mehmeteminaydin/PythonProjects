# 
# MODIFY get_data() AS YOU LIKE.
# DO NOT SEND THIS FILE TO US

import random
random.seed(111)  #remove hash-sign to get randomization seed we will be using at evaluation
#                    (if you fix the seed you get always the same probabilty choice sequence)




def get_data():
	"""Get the initial state of the individuals & the environment"""
	return [50, 100, 5, 80, 30, 0.55, [
[(40, 7), 6, 'notmasked', 'notinfected'] ,
[(49, 9), 0, 'masked', 'notinfected'] ,
[(31, 12), 6, 'masked', 'infected'] ,
[(38, 12), 2, 'notmasked', 'infected'] ,
[(62, 12), 2, 'masked', 'infected'] ,
[(47, 13), 1, 'notmasked', 'notinfected'] ,
[(33, 17), 0, 'notmasked', 'notinfected'] ,
[(55, 17), 2, 'masked', 'notinfected'] ,
[(38, 20), 4, 'notmasked', 'notinfected'] ,
[(45, 20), 4, 'notmasked', 'infected'] ,
[(62, 20), 5, 'notmasked', 'notinfected'] ,
[(29, 21), 7, 'masked', 'notinfected'] ,
[(45, 25), 4, 'masked', 'notinfected'] ,
[(52, 25), 0, 'notmasked', 'notinfected'] ,
[(31, 26), 2, 'notmasked', 'notinfected'] ,
[(61, 26), 7, 'notmasked', 'infected'] ,
[(38, 29), 3, 'notmasked', 'infected'] ,
[(51, 31), 4, 'masked', 'infected'] ,
]]