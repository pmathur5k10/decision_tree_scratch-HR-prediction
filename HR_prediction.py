import csv
import numpy
import pandas


def class_count(data):

	count={}

	for row in data:
		label=row[-1]

		if label not in count:
			count[label]=0
		
		count[label]+=1

	return count	

def is_numeric(value):

	return isinstance(value,int) or isinstance(value,float)



header=['satisfaction_level','last_evaluation','number_project','average_monthly_hours','time_spent','work_accident','promotion','occupation','salary','left']

class Question:

	def __init__(self,column,value):

		self.column=column
		self.value=value



	def match(self,example):

		val=example[self.column]

		if is_numeric(val):
			return val >= self.value
		else :
			return val == self.value

	def repr(self):

		condition = '=='
		if is_numeric(self.value):
			condition='>='

		return "Is %s %s %s ?" %(header[self.column],condition,str(self.value))


			
def partition(data,question):

	true_rows=[]
	false_rows=[]

	for row in data:
		if question.match(row):
			true_rows.append(row)
		else:
			false_rows.append(row)
	
	return true_rows , false_rows		

def gini(data):

	impurity=1
	count=class_count(data)
	for c in count:
		pi=count[c]/float(len(data))
		prob=pi*(1-pi)
		impurity+=prob

	return impurity	


def info_gain(curr,left,right):

	ratio=float(len(left))/(len(left)+ len(right))
	return curr - ratio*gini(left) - (1-ratio)*gini(right) 


def best_split(data):

	best_gain=0
	best_ques=None
	curr_uncertainity=gini(data)

	n_features=len(data[0])-1

	for col in range(n_features):

		values=set([row[col] for row in data])

		for val in values:

			q=Question(col,val)

			true_rows, false_rows=partition(data,q)
			if len(true_rows)==0 or len(false_rows)==0:
				continue


			gain=info_gain(curr_uncertainity,true_rows,false_rows)

			if gain>=best_gain:
				best_gain,best_ques=gain,q


	return best_gain,best_ques			



class Leaf:

	def __init__(self,data):
		self.prediction=class_count(data)


class Node:

	def __init__(self,true_branch,false_branch,question):

		self.true_branch=true_branch
		self.false_branch=false_branch
		self.question=question


def build_tree(data):

	if len(data)==0:
		return null

	best_gain,best_ques=best_split(data)

	if best_gain==0:
		return Leaf(data)

	
	true_rows,false_rows=partition(data,best_ques)

	true_branch=build_tree(true_rows)
	false_branch=build_tree(false_rows)

	d=Node(true_branch,false_branch,best_ques)	

	return d


def classify(row,node):

	if isinstance(node,Leaf):
		return node.prediction

	
	if node.question.match(row):
		return classify(row,node.true_branch)
	else:
		return classify(row,node.false_branch)

	

	


if __name__ =='__main__' :

	f=open('HR.csv','rb')
	reader=csv.reader(f)

	data=[]

	for row in reader:
		data.append(row)
		

	f.close()

	data=data[1:]
	
	training_data=data[100:]
	testing_data=data[:100]

	myTree=build_tree(training_data)
	for r in testing_data:
		pred=classify(r,myTree)
		max_count=0
		max_class=None
		for c in pred:
			if pred[c]>=max_count:
				max_count=pred[c]
				max_class=c

		print(max_class)		



	


	


	






		