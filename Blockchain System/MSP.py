
import random
from CA_certification import *
from whatspp_OTP import *
#from Login_Register import *
import smtplib, ssl

OTP_List = []
OTP_Pointer = 0

for i in range(100):
    x = random.randrange(100, 999, 1)
    # print(x)
    y = random.choices(["e", "f", "g", "f", 'i'])
    # print(y)
    z = str(x) + y[0]
    # print(z)
    OTP_List.append(z)
print(OTP_List)


def OTP_Provider(OTP_Pointer):
    OTP = OTP_List[OTP_Pointer]
    OTP_Pointer = OTP_Pointer + 1
    return OTP



def MSP_communication(a):
    if a in certification_keys:
        OTP_Provider(OTP_Pointer)
    else:
        return False


#Trying to establish a conneciton between the CA and the MSP
def cartification_validity(Company_ID):
    Bool_input = input("What is the validity of the company?")
    if Bool_input == True:
        MSP_communication(a = True)


def email_send(x):
    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls
    sender_email = "emp.site2021@gmail.com"
    password = 'empsity#'

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.ehlo()  # Can be omitted
        server.starttls(context=context)  # Secure the connection
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, 'emp.site2021@gmail.com', 'This is the message with OTP' + str(x))

    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit()


def member_validation():
    #this is the function from the CA_certification
    X = Question()
    if X == True:
        print("The member validation process was run")
        #send_whatsapp_OTP(OTP_Provider(OTP_Pointer)) #this method to send the OTP to user
        #this sends the automatic email
        email_send(OTP_Provider(OTP_Pointer))
        return OTP_Provider(OTP_Pointer)

    else:
        print("The validation was unsuccessful")



