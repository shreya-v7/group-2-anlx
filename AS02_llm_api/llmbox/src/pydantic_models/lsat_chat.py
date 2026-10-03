import os
import sys
import typing
import pydantic
import typing
import json

from typing import List, Optional
from pydantic import BaseModel

#class CodeGenerator(BaseModel):
    #programmingLanguage: List['python']

#What does application do?
# In response to a user question about dogs, it prints out a dictionary key with a dog breed name
class DogList(BaseModel):
    breedname: str



#class LSATCHAT(BaseModel):
    #question_prompts:
