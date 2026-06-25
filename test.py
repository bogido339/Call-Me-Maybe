from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int


person = Person

person.name = 39
person.age = "*"

print(person.name, person.age)