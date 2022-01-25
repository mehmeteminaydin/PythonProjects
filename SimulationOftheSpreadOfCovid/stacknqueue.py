### Stack interface

def CreateStack():
    """ Creates an empty Stack """
    return []

def Push(item, Stack):
    """ Add item to the top of the Stack """
    Stack.append(item)

def Pop(Stack):
    """ Remove and return the item at the top of the Stack """
    return Stack.pop()

def Top(Stack):
    """ 
    Return the value of the item at the top of the Stack
    without removing it.
    """
    return Stack[-1]

def IsEmptyStack(Stack):
    """ Check whether the Stack is empty """
    return Stack == []

### Queue interface

def CreateQueue():
    """ Creates an empty Queue """
    return []

def Enqueue(item, Queue):
    """ Add item to the end (back) of the Queue """
    Queue.append(item)

def Dequeue(Queue):
    """ Remove and return the item at the front of the Queue """
    return Queue.pop(0)

def Front(Queue):
    """ Return the value of the current front item without removing it """
    return Queue[0]

def IsEmptyQueue(Queue):
    """ Check whether the Queue is empty """
    return Queue == []

