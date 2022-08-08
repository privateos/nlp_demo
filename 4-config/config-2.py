
with open('class.txt', encoding='utf-8') as f:
    class_list = []
    for line in f.readlines():
        class_list.append(line.strip())

print(class_list)
print(len(class_list))