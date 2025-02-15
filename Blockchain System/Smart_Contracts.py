
import csv
import os
import Linked_List_Blockchain
from Linked_List_Blockchain import *
# Import writer class from csv module
from csv import writer



#This is the start of the experiemntal code
# List
List = [6, 'William']

# Open our existing CSV file in append mode
# Create a file object for this file
with open('Transaction_Ledger.csv', 'a') as f_object:
    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)

    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(List)

    # Close the file object
    f_object.close()

#This is the end of the experimental code


password_list = []

def create_private_channel(channel_name):
    channel_name = LinkedList()
    channel_name.insert_at_beginning("Genesis")
    #There should be a dictionary here that will log in the transaciton ledger
    return channel_name


# this will not be shown to the user here for now
def contract_block_creation(name, channel_name):
    channel_name.insert_at_end(name)

def show_contract_options():
    print("1.  These are the avaiable contracts --> 001")
    print("2.  create_private_channel --> 002")
    print("3.  push_new_block -->003")
    #print("4.  contract_block_creation -->004")
    print("5.  make_transaction_ledger_log -->005")


def show_blocks(channel_name):
    print(channel_name.print())

def make_transaction_ledger_log():
    pass


def main_ledger_log(x):
    pass



def push_new_block(name):
    private_channel_ = create_private_channel("Channel_1")
    print(private_channel_)
    contract_block_creation(name, private_channel_)

#print("Welcome to the system:")
#print("These are the available contracts:")
#make this uncommented orelse the options will not come!
#show_contract_options()


def system_run(Function_Code):
    if Function_Code == "001":
        show_contract_options()
        Function_Code_02 = input("Enter code for the function to run")
        system_run(Function_Code_02)

    if Function_Code == "002":
        name = input("Enter private channel name")
        create_private_channel(name)
        Function_Code_02 = input("Enter code for the function to run")
        system_run(Function_Code_02)

    if Function_Code == "003":
        name = input("Enter name of new company")
        push_new_block(name)
        Function_Code_02 = input("Enter code for the function to run")
        system_run(Function_Code_02)

    if Function_Code == "005":
        pass


#Function_Code = input("Enter code for the function to run")
#Function_Code = str(Function_Code)
#system_run(Function_Code)


