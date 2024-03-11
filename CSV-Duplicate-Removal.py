#import data set first from a text file
import csv
with open('filename.csv','r') as fd:
    reader = csv.reader(fd)
    data = list(csv.reader(fd))


#define a function for removing duplicates in a list
def remove_duplicates(a):
    i = 0
    while i < len(a):
        j = i + 1
        if a[i] == a[j]:
            del a[j]
        else:
            j += 1
    i += 1

remove_duplicates(data)
print(data)