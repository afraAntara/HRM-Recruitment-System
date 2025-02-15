

import random


Hash = []

def Generate_Hash(Hash):
    for i in range(100):
        x = random.randrange(100, 999999, 1)
        # print(x)
        y = random.choices(["e", "f", "g", "f", 'i', 'j', 'k'])
        # print(y)
        z = '#' + str(x) + y[0]
        # print(z)
        Hash.append(z)

#There will be a dictionary here that will map the name of a company with two hash values,
# the hash value of the company and the hash value of the company block before that

Generate_Hash(Hash)
print(Hash)