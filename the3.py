# I prefer to use list data type in my tree structure. This function helps me.
def tuple_to_list(part_list):
    for i in range(len(part_list)):
        for j in range(len(part_list[i])):
            if type(part_list[i][j]) is tuple:
                part_list[i][j] = list(part_list[i][j])
    return part_list


leaves = []


def helper_converter(part_list):
    # This function has the main mission for converting part_list to my tree structure.
    # The idea is simple. Firstly, I seperate leaves and remove them from part_list.
    # Then, I check whether any leaves exist in part_list. If yes, I implement this leaf into that place.
    # Until one element left in part_list, I should continue recursively.
    global leaves
    i = 0
    # I seperate leaves and the left ones for the first time.
    while i < len(part_list):
        if type(part_list[i][1]) is float:
            leaves.append(part_list[i])
            part_list.pop(i)
        elif all(len(part_list[i][a]) >= 3 for a in range(1, len(part_list[i]))):
            leaves.append(part_list[i])
            part_list.pop(i)
        else:
            i += 1
    #for the second time
    for i in range(len(part_list)):
        for j in range(len(part_list[i])):
            for leaf in leaves:
                if type(part_list[i][j]) is list and part_list[i][j][1] == leaf[0] and len(leaf) <= 2:
                    part_list[i][j] = part_list[i][j] + [leaf[1]]
                    leaves.remove(leaf)
                elif type(part_list[i][j]) is list and part_list[i][j][1] == leaf[0] and len(leaf) >= 3:
                    part_list[i][j] = part_list[i][j] + leaf[1:]
                    leaves.remove(leaf)
    # this is the recursive part. In my algorithm, until only one element left, I need to call this function again and again.
    if 1 < len(part_list):
        return helper_converter(part_list)
    else:
        part_list[0].insert(0, 1)
        return part_list[0]


"""
    My tree structure is =>  [root_quantity, root_name, [children1_quantity, children1_name, [leaf1_quantity, leaf1_name, leaf1_price]]]
    (root_quantity is just for the sake of my calculation function)
"""


def converter(part_list):
    # this function just helps me to connect two functions.
    # I will only call this function when needed.
    tuple_part_list = tuple_to_list(part_list)
    return helper_converter(tuple_part_list)


def calculate_price(part_list):
    # this function converts part_list to the tree and put it into the helper function as parameter.
    t = converter(part_list)
    return helper_calculater(t)


def helper_calculater(t):
    # I came up with that I should multiply leaves quantites and their prices, then add them each other recursively.
    var = 0
    if is_leaf(t):
        return t[0] * t[2]
    else:
        for child in t[2:]:
            var += helper_calculater(child)
        return var * t[0]


def is_leaf(t):
    # In my tree structure, [root_quantity, root_name, [children1_quantity, children1_name, [leaf1_quantity, leaf1_name, leaf1_price]]]
    # I need to check the second index's element, if it's type is float then it is a leaf.
    if type(t[2]) is float:
        return True


def required_parts(part_list):
    t = converter(part_list)
    return helper_required_parts(t)


leavess = []
num = 1


def helper_required_parts(t):
    global leavess
    global num
    if is_leaf(t):
        return t[0]
    else:
        for children in t[2:]:
            if is_leaf(children):
                leavess.append((t[0] * num * children[0], children[1]))

            else:
                num = num * t[0]
                helper_required_parts(children)
                num = 1
        return leavess


def stock_check(part_list, stock_list):
    checked = []
    required_part_list = required_parts(part_list)
    if stock_list == []:
        for required in required_part_list:
            new_list = list(required)
            new_tuple = (new_list[1], new_list[0])
            checked.append(new_tuple)
    else:
        for required_part in required_part_list:
            num = 0
            for stock_part in stock_list:
                if required_part[1] == stock_part[1]:
                    if required_part[0] > stock_part[0]:
                        checked.append((required_part[1], required_part[0]-stock_part[0]))
                        break
                else:
                    num += 1
                    if num == len(stock_list):
                        checked.append((required_part[1], required_part[0]))
    return checked











