class Student(object):
    # def __init__(self, name, age):
    #     self.__name = name
    #     self.__age = age
    #     pass

    def print_score(self):
        print(self.__name)

    def __str__(self):
        return 'this is (name:%s)' % self.__name

    def __iter__(self):
        return self

    def __next__(self):
        return self.__name

    def __getitem__(self, item):
        for i in range(item):
            return i

# student = Student('lisi', '18')
# #print(student.age)
# student.score = 88
# print(student.score)
# student.print_score()
# print(student)

# for item in student:
#     print(item)


f = Student()
f[10]

