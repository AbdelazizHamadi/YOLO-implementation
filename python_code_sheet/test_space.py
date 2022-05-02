from PyQt5.QtCore.QTextCodec import kwargs
import pandas as pd

class Person():
    __number_of_people = 0

    def __init__(self, name):
        self.name = name
        Person.add_people()

    @classmethod
    def add_people(cls):
        Person.__number_of_people += 1

    @classmethod
    def get_number_of_people(cls):
        return Person.__number_of_people


class Parent():

    def __init__(self, name, **kwargs):
        self.name = name

        self.age, self.accent = self.details(**kwargs)

    def details(self, age, accent):
        return age, accent

        return True#

class shit():

    def __init__(self, max):
        pass
    def __setitem__(self, key, value):
        pass
    def __getitem__(self, item):
        pass




