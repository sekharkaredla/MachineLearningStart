from sklearn import datasets
from sklearn import tree
import graphviz
import numpy

model = tree.DecisionTreeClassifier(criterion = "gini")

iris = open('IRIS.csv','r')

data = numpy.loadtxt(iris, delimiter=',', dtype = 'float',usecols=range(0,8))

iris.close()

iris = open('IRIS.csv','r')

target = numpy.loadtxt(iris, delimiter=',', dtype = 'string',usecols=[8])

iris.close()

print data,'\n',target

# print dataset.data,'\n',dataset.target

model.fit(data,target)

print model

graph_data = tree.export_graphviz(model, out_file=None)

graph = graphviz.Source(graph_data)

graph.render("iris")
