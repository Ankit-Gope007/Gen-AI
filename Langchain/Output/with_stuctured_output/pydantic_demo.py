from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str
    age : Optional[int] = 18
    email : Optional[EmailStr] = None
    cgpa : Optional[float] = Field( gt=0.0, lt=10.0,default=7.0, description="CGPA must be between 0.0 and 10.0")
    


new_student = {'name':'Ankit'}

student_obj = Student(**new_student)

print(student_obj)
print(type(student_obj))