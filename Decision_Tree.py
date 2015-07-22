import numpy as np
import math

def createDataset():
	features_name = ["OutLook","Temp","Humidity","Windy","Play"]
	X = np.array([
				["Rainy","Hot","High","False","No"],
				["Rainy","Hot","High","True","No"],
				["Overcast","Hot","High","False","Yes"],
				["Sunny","Mild","High","False","Yes"],
				["Sunny","Cool","Normal","False","Yes"],
				["Sunny","Cool","Normal","True","No"],
				["Overcast","Cool","Normal","True","Yes"],
				["Rainy","Mild","High","False","No"],
				["Rainy","Cool","Normal","False","Yes"],
				["Sunny","Mild","Normal","False","Yes"],
				["Rainy","Mild","Normal","True","Yes"],
				["Overcast","Mild","High","True","Yes"],
				["Overcast","Hot","Normal","False","Yes"],
				["Sunny","Mild","High","True","No"]
			 ])

	return X,features_name

def createTree(data,features_allows,tree):
	#Calcular Entropia Target antes de Split
	target_entropy = entropy(data)

	#verificar si todo es una sola clase
	# para ello, si la entropia es igual a cero,
	# quiere decir que el dataset esta puro
	if target_entropy == 0:
		tree["Play"] = data[0,-1]
		return None
	#En caso contrario seleccionar el mejor atributo, segun la ganancia de Informacion
	#Information Gain = EntropiaTarget - EntropiaAtributo
	list_information_gain = []
	for feature in features_allows:
		index_feature = features_allows.index(feature)
		calc_entropy = entropy(data,index_feature)
		list_information_gain.append(target_entropy - calc_entropy)

	index_best_attribute = list_information_gain.index(max(list_information_gain))
	print("Best Attribute : %s"%features_allows[index_best_attribute])
	
	best_feature = features_allows[index_best_attribute]
	tree[best_feature] = dict()
	set_best_attribute_values = set(data[:,index_best_attribute])
	print(tree)
	for value in set_best_attribute_values:
		tree[best_feature][value] = dict()
		index_value = data[:,index_best_attribute]==value
		sub_data = data[index_value]
		sub_features_allows = features_allows[:index_best_attribute] + features_allows[index_best_attribute+1:]
		createTree(sub_data,features_allows,tree[best_feature][value])

	return tree


#Entropy = - p * log2p - q * log2q
def entropy(data,index_feature = None):
	total_instace = len(data)

	'''Si no recive name_feature 
	   calcular Entropia del Target
	'''
	if index_feature is None:
		values = set(data[:,-1])
		calc_entropy = 0
		for value in values:
			p = np.sum(data[:,-1] == value) / float(total_instace)
			if p != 0:
				calc_entropy += -p * math.log(p,2) 
		return calc_entropy
	else:
		'''	Obtener conjunto de valores,
			Calcular entropia del atributo en base al target
		
			Por cada valor del atributo:
			Entropy += p(i) * entropy(particion_target)
			p(i) ~ Probabilidad del Valor i del atributo
			entropy(particion_target) ~ Particionar en base al valor del atributo y calcular la entropia del Target
		'''
		values = set(data[:,index_feature])
		probability_sum = 0
		for value in values:
			#print value
			index_value = data[:,index_feature]==value
			sub_data = data[index_value]
			calc_entropy = entropy(sub_data)
			p_frecuency = np.sum(data[:,index_feature]==value)
			probability_sum += p_frecuency/float(total_instace) * calc_entropy
		return probability_sum


def classify(input,features,tree):
	return None


def test_create_tree():

	data,features = createDataset()

	#Todos los features disponibles sin incluir el ultimo, ya que es el Target
	tree =  createTree(data,features[:-1],dict())
	print(tree)
	