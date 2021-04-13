



def pattern(num):

    # Defining outbound limit
    limit = 100

    # Pattern Size
    size = 2 * num - 1
    front = 0
    back = size - 1
    a = [[0 for i in range(limit)] for i in range(limit)]
    while (num != 0):
        for i in range(front, back + 1):
            for j in range(front, back + 1):
                if (i == front or i == back or
                        j == front or j == back):
                    a[i][j] = num
        front += 1
        back -= 1
        num -= 1

    result = ''
    for i in range(size):
        for j in range(size):
            result = str(result + str(a[i][j]))
        result += ' '
    print(result);

pattern(4)