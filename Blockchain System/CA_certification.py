
import random
from collections import deque

certification_keys = []
certificate_ID_val = deque()

for i in range(100):
    x = random.randrange(100,999,1)
    #print(x)
    y = random.choices(["a","b","c","d",'e'])
    #print(y)
    z = str(x) + y[0]
    #print(z)
    certification_keys.append(z)

print("These are the certification keys:")
print(certification_keys)


#this pointer will increment when a certificate key is given by the CA
CA_Pointer = 0



def Question():
    Q = bool(input("Is the data provided valid to process with the certificate ID?"))
    return (Provide_Certificate_ID(Q,CA_Pointer))


def Provide_Certificate_ID(a, CA_Pointer):
    if a == True:

        print("This is the provided certification ID: " + certification_keys[CA_Pointer -1 ])
        certificate_ID_val.append(certification_keys[CA_Pointer])
        CA_Pointer = CA_Pointer + 1
        return True
    else:
        input("What is the status now?")
        return False













