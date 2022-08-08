
class_list = [
    x.strip() for x in open(
        'class.txt', encoding='utf-8').readlines()
]
print(class_list)
print(len(class_list))