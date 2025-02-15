import csv
import os

def to_dict_ledger():
    pass



#This code will create a dictionary from the target csv file
mydict = {}
with open('Employee_Satisfaction.csv', mode='r') as inp:
    reader = csv.reader(inp)
    dict_from_csv = {rows[0]:rows[10] for rows in reader}
print(dict_from_csv)




#This will be a code for the transaction ledger
a = "Transaction_Ledger"
a_file = open("Companies/" + a + ".csv", "w")
ledger2 = {}
ledger2["status"] = "active"
ledger2["capacity"] = "5"
a_dict = ledger2
writer = csv.writer(a_file)
for key, value in a_dict.items():
    writer.writerow([key, value])
a_file.close()



# This code will create a main ledger csv file from the dictionary
b = "main_ledger"
b_file = open("Companies/" + b + ".csv", "w")
ledger = {}
ledger = dict_from_csv

b_dict = ledger
writer = csv.writer(b_file)
for key, value in b_dict.items():
    writer.writerow([key, value])
b_file.close()













#This is the code for each Block in the blockchain
class Node:
    global ledger
    global ledger2

    def file_copy(self):


        y = self.data
        z = self.data + "_Transaction_Ledger"

        # The local ledger is the global ledger that is the main ledger for the data
        local_ledger = ledger
        local_transaction_ledger = ledger2

        #base dir
        _dir = r'C:\Users\USER\PycharmProjects\PandasPractise\Companies'
        # create dynamic name, like "D:\Current Download\Attachment82673"
        _dir = os.path.join(_dir, '%s' % y)
        #print(_dir)

        # create 'dynamic' dir, if it does not exist
        if not os.path.exists(_dir):
            os.makedirs(_dir)

        # this next line is VVIP to understand
        #This is the code that makes a dynamic file name inside the companies folder
        _dir2 = '{0}\\'.format(_dir)


        #This code will be making the copy of the final ledger copy from the main medger
        copy_file = open(_dir2 + y + ".csv", "w")
        #print(copy_file)
        a_dict = local_ledger
        writer = csv.writer(copy_file)
        for key, value in a_dict.items():
            writer.writerow([key, value])
        copy_file.close()


        #This code will be making a local copy of the transaction ledger
        copy_file_2 = open(_dir2 + z + ".csv", "w")
        #print(copy_file_2)
        b_dict = local_transaction_ledger
        writer = csv.writer(copy_file_2)
        for key, value in b_dict.items():
            writer.writerow([key, value])
        copy_file_2.close()



#we should be giving the hash here and the password here too
    def __init__(self, data = None, next = None):
        self.data = data
        self.next = next
        self.copy = ledger # this is a dictionary
        print(self.copy)
        #File copy gets called here
        self.file_copy()
        print("Node is created")


    @classmethod
    def update(cls):
        local_ledger = ledger
        return local_ledger











#This is the code for a private channel in the blockchain
class LinkedList:

    global ledger

    def __init__(self):
        self.head = None



    # data will be the name of the company
    def insert_at_beginning(self, data):
        node = Node(data, self.head)
        self.head = node

    def insert_at_end(self, data):
        if self.head is None:
            self.head = Node(data, None)
            return

        itr = self.head

        while itr.next:
            itr = itr.next

        itr.next = Node(data, None)


    def print(self):
        if self.head is None:
            print("linked List is empty")
            return

        itr = self.head
        llstr = ''

        while itr:
            llstr += str(itr.data) + "-->"
            itr = itr.next

        print(llstr)















