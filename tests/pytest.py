#!/usr/bin/env python


class test:
	def __init__(self, name):
		self.name = name;

	def printname(self):
		print(self.name);


if __name__ == '__main__':
	t = test("wangxiao");
	t.printname();
