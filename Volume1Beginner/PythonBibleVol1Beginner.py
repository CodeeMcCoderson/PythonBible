# Typecasting, one data type to another
def typeCasting():
    value = '10'
    value2 = '20'
    value3 = value + value2
    print(value3)

    number = int(value)
    number2 = int(value2)
    number3 = number + number2
    print(number3)

# Arithmetic Operators
def arithmeticOperators():
    ADD = 10 + 10
    print(ADD)
    SUB = 20 - 10
    print(SUB)
    MUL = 10 * 10
    print(MUL)
    DIV = 20 / 10
    print(DIV)
    MODULUS = 23 % 10
    print(MODULUS)
    EXP = 5 ** 3
    print(EXP)
    FLOORDIV = 27 // 6
    print(FLOORDIV)

# Assignment Operators
def assignmentOperators():
    Variable = 'Hello'
    x = 0
    print(Variable)
    x += 30
    print(x)
    x -= 10
    print(x)
    x *= 2
    print(x)
    x /= 2
    print(x)
    x %= 3
    print(x)
    x **= 2
    print(x)
    x //= 2
    print(x)

# Comparison Operators NOT DONE
#def comparisonOperators():
    x1 = 2
    if x1 == 0:
        print('x1 equals 0')
    elif x1 == 2:
        print('x1 equals 2')
    else:
        print('Who knows what x1 equals')

    x1 = 2
    if x1 != 0:
        print('x1 does not equal 0')
    elif x1 != 2:
        print('x1 does not equal 2')
    else:
        print('x1 may or may not equal 2')
    #>
    #<
    #>=
    #<=

# Booleans NOT DONE
#def booleans():
    #True
    #False

# Logical Operators NOT DONE
#def logicalOperators():
    #1 or 2
    #1 and 2
    #not True

# Input Function
def inputFunction():
    y = input('Enter a number')
    y2 = input('Enter another number')
    sum = y + y2
    print('The sum of those two numbers is ' + sum )

# IF, ELIF, ELSE
def ifElifElse():
    number = 10
    if number < 10:
        print('The number is less than 10')
    elif number > 10:
        print('The number is less than 10')
    else:
        print('Your number is 10')

# While loop
def whileLoop():
    number2 = 0
    while number2 < 10:
        number2 += 1
        print(number2)

# For loop (Can use Break, Continue, Pass statements as well)
def forLoop():
    numbers = [10, 20, 30, 40]
    for x in numbers:
        print(x)

# Opening and closing files
# file = open('myfile.txt', 'r')
# The r can be r+, rb, rb+, w, w+, wb, wb+, a, a+, ab, ab+

#Lists, tuples, dictionaries, dictionary arrays

#String functions (.join, .replace, .split)






print('Begin')
#typeCasting()
#arithmeticOperators()
#assignmentOperators()
#comparisonOperators() #NOT DONE
#booleans() #NOT DONE
#logicalOperators() #NOT DONE
#inputFunction()
#ifElifElse()
#whileLoop()
#forLoop()
print('End')
