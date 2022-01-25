from stacknqueue import *
def permutations_of_all_size(lst):
    output = []
    queue = CreateQueue()
    for i in range(len(lst)):
        output.append([lst[i]])
        initializer1, remainer1 = ([lst[i]] , lst[:i]+lst[i+1:])
        Enqueue((initializer1, remainer1),queue)
    while queue:
        initializer, remainer = Dequeue(queue)
        if len(remainer) == 1:
            newPerm = initializer + remainer
            output.append(newPerm)
        if len(remainer) > 1:
            for j in range(len(remainer)):
                newPerm = initializer + [remainer[j]]
                output.append(newPerm)
                Enqueue((newPerm,remainer[:j]+remainer[j+1:]),queue)
    return output
def str_to_list(str):
    stack = CreateStack()
    for i in str:
        if i == "[":
            Push([],stack)
        if i != "," and i != "]" and i != "[":
            Push(int(i),stack[-1])
        if i == "]":
            if len(stack) > 1:
                pop = Pop(stack)
                Push(pop,stack[-1])
            elif len(stack) == 1:
                return Pop(stack)


