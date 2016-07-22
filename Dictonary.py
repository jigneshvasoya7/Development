# Create and updeate dictionary
dict = {'jignesh':1,'Raviraj':2}
# dict = {}

strKey = 'jignesh'
uid = dict.get(strKey)
if uid != None:
    print "Key found: ",uid
    print dict.keys()
    uname = dict.keys()[dict.values().index(uid)]
    print uname
    print type(uname)
    # print "Key Value: ".format(dict.keys()[dict.values().index(uid)])
else:
    dictTemp = {strKey:3}
    dict.update(dictTemp)
    print "Key not found: ",dict.get(strKey)



strKey = 'Jignesh'
uid = dict.get(strKey)
if uid != None:
    print "Key found: ",uid
    print dict.keys()
    uname = dict.keys()[dict.values().index(uid)]
    print uname
    print type(uname)
    # print "Key Value: ".format(dict.keys()[dict.values().index(uid)])
else:
    # Obtain new uid number from list
    cnt = 1

    dictTemp = {strKey:3}
    dict.update(dictTemp)
    print "Key not found Added: ",dict.get(strKey)
print "Full Dict: ",dict



############# Create and update csv file
import csv
import pandas as pd

# Get user id from list
def getUID(listUids):
    listUids.sort()
    cnt = 1
    for uid in listUids:
        if uid != cnt:
            print "UID Not Found",cnt
            break
        else:
            cnt = cnt + 1
    return cnt

headerNames = ['UID', 'UserName']

def writeUserDetails(filename,headerNames,dictionary):
    with open(filename, 'w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = headerNames)
        writer.writeheader()
        writer.writerows(dictionary)

def ReadUserDetails(filename):
    df = pd.read_csv(filename)
    UIds = df.UID
    UserNames = df.UserName
    return UIds,UserNames

# Create and update csv file


with open('D:/test.csv', 'w') as csvfileW:
    fieldnames = ['UID', 'UserName']
    writer = csv.DictWriter(csvfileW, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'UID': 1, 'UserName': 'jignesh'})
    writer.writerow({'UID': 2, 'UserName': 'raviraj'})

with open('D:/test.csv', 'r') as csvfileR:
    reader = csv.DictReader(csvfileR)
    for row in reader:
        print(row['UID'], row['UserName'])

# CSV File handling with pandas
import pandas as pd
df = pd.read_csv('D:/test.csv')
# print df.keys()
UIds = df.UID
UserNames = df.UserName
print UIds[0]
print UserNames[0]




mydict = {'george':1,'amber':2}
print mydict.keys()[mydict.values().index(1)]
print getUID(mydict.values())

