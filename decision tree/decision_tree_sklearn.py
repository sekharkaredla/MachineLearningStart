from sklearn import datasets
from sklearn import tree
import graphviz
import numpy

model = tree.DecisionTreeClassifier(criterion = "gini")

iris = open('IRIS.csv','r')

data = numpy.loadtxt(iris, delimiter=',', dtype = 'float',usecols=range(0,8))

iris.close()

iris = open('IRIS.csv','r')

target = numpy.loadtxt(iris, delimiter=',', dtype = 'str',usecols=[8])

iris.close()


model.fit(data,target)

graphviz.Source(tree.export_graphviz(model, out_file=None)).render("iris_1")

