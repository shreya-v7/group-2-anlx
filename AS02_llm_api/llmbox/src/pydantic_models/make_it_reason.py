import os
import sys
import typing
import pydantic
import typing
import json

from typing import List, Optional
from pydantic import BaseModel

class Reasoner(BaseModel):
    userquestion: str="When will final exams be over and when will we get vacation?"
    questionmeaning: str = "Go to the final exam schedule webiste and step by step describe when the final exam will be and when we will get vacation"
    answeroptions: list['ClassA', 'ClassB', 'ClassC', 'ClassD']
    optionreason: str
    answer: str
