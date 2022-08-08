word = lambda x: x.split(' ')
char = lambda x: [y for y in x]

# s = '我是大肥猪'
# print(char(s))

# s = '我 是 大 肥猪'
# print(word(s))


def def_char(x):#x = '我是大肥猪'
    return [y for y in x]

def def_word(x):#x = '我 是 大 肥猪'
    return x.split(' ')
#def_word = ( lambda x: x.split(' ') )

print(def_word(x = '我 是 大 肥猪'))