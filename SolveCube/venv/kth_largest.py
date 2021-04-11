input1 = [2, 3, 1, 5, 6, 2, 1]
input2 = 4


def max_k(input1, input2):
    # Removing duplicate elements from input
    res = []
    for i in input1:
        if i not in res:
            res.append(i)

    if len(res) < input2:
        # Returning (-1) if k > the number of elements in the array
        return int(-1)
    else:
        # Sort the given array in descending order
        res.sort(reverse=True)

        # Return k'th element in the sorted array
        return res[input2 - 1]


print(max_k(input1, input2))