from MSP import OTP_List, member_validation
from Smart_Contracts import create_private_channel,push_new_block
import smtplib, ssl
from CA_certification import Question, CA_Pointer

#these are the codes for storing the usernames and the passwords
passwords = {}


x = member_validation()
create_private_channel("Private_01")


#code for asking for the registration phase
def register_process():
    id = input("Input your company name")
    password = input("Input your password")
    #OTP =  input("enter your OTP")
    push_new_block(id)


#code for checking the OTP that was entered by the user
def check_OTP(x):
    if x in OTP_List:
        register_process()
    else:
        print("OTP error")

def login(id, password):
    pass




OTP_provided = input("Enter the OTP sent")
check_OTP(OTP_provided)
