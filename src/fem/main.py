#-*- coding:utf-8 -*-
import json
from gen import Lame, json2inp
from solver import solver
from tester import LameTest

config = {
	"rMin":1,
	"rMax":1.5,
	"resolution":50,
	"E":2*10**11,
	"Nu":0.25,
	"P1":0.0,
	"P2":1.0
}

if __name__ == '__main__':

	task = Lame(config) # массив данных

	inp = json2inp(task)
	with open("task.inp", "w") as write_file:
		write_file.write(inp)

	with open("task.json", "w") as write_file:
		json.dump(task, write_file)

	with open("task.json", "r") as read_file:
		task = json.load(read_file)

	result = solver(task)

	LameTest(result, config)


