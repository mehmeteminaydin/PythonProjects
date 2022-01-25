# n-ary tree functions module

def datum(t):
    return t[0]

def children(t):
    return t[1:]

def is_leaf(t):
    return len(children(t)) == 0

def is_empty(t):
    return t == []

def print_tree(tree, level=0):
    if not tree:
        return
    print(" " * 4 * level + str(datum(tree)))
    for child in children(tree):
        print_tree(child, level+1)
