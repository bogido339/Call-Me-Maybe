from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age
    email: str

user = UserProfile(name="Liam Svensson", age="a", email="liam.svensson@example.com")
print(user)