# treefunctions.py:
def datum(tree):
    return tree[0]
 
def left_child(tree):
    return tree[1]
 
def right_child(tree):
    return tree[2]
 
def children(tree):
    return tree[1:]
 
def maketree(node, children):
    return [node] + children
 
def isleaf(tree):   # for binary tree
    return left_child(tree) == [] and right_child(tree) == []

def is_empty(tree):
    return False if tree else True

def print_tree(tree, level=0):  # for binary tree
    if is_empty(tree):
        return
    print_tree(left_child(tree), level+1)
    print(" " * 4 * level + ("->" if level else "") + str(datum(tree)))
    print_tree(right_child(tree), level+1)
