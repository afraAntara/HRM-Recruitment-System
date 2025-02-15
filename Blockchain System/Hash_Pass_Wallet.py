

from Hash_Generator import Hash
from tinydb import *


print(Hash)

db = TinyDB('db.json')

#counter = 0

#for i in range(len(Hash)):
#    db.insert({counter: Hash[i]})
#    counter = counter + 1


print(db.all())

#so now 100 hash values has been mapped in the db database and these will not change

x = db.all()
Hash_Store = []
for i in range(100):
    print(x[i])

    y = dict(x[i])
    #print(type(y))
    #print(y.items())
    z = str(y).split("'")
    #print(z)
    a = z[3]
    Hash_Store.append(a)


print(Hash_Store)

#this is 100
print(len(db))

password_manager = []


#to store the passwords entered by the user
def store_password(id, password):
    pass

