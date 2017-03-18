import numpy as np
import random
import game

def add_target(asked_questions, target_objects, weights, questions, question_features, asked_questions_data, answer_target_data):
	new_target_name = input('Ingrese nombre : ').strip()
	target_object_id = check_target(new_target_name, target_objects)

	features = {}
	features['ingredients'] = input('Ingredientes: ').split(',')
	features['flavor'] = input('Sabor: ').split(',')
	print('Tipo: ')
	types = ['entrada', 'sopa', 'segundo', 'postre', 'aperitivo', 'otros']
	for i in range(len(types)):
		print('[%d] %s' %(i, types[i]))

	options = input('Opcion(es): ').split(',')
	features['target_type'] = [ types[int(option)] for option in options] 
	features['country'] = input('Origen o Pais: ').split(',')
	features['other'] = input('Otra caracteristica: ').split(',')
	weights = game.add_features(features, questions, question_features, weights, asked_questions)

	if target_object_id is not None:
		game.learn(asked_questions, target_object_id, weights, asked_questions_data, answer_target_data)
	else:
		weights = game.learn_new_target(asked_questions, new_target_name, target_objects, weights, asked_questions_data, answer_target_data)

	return weights

def check_target(name, target_objects):
	similar = []

	for target_object_id in range(len(target_objects)):
		value = target_objects[target_object_id]
		if name in value or value in name:
			similar.append((target_object_id, value))

	if len(similar) == 0:
		return None	

	print('Nombres similares:')

	for i in range(len(similar)):
		value = similar[i][1]
		print('[%d] %s' %(i, value))

	answer = input('Es alguno de estos?(y/n):').strip()

	if answer == 'n':
		return None

	option = int(input('Opcion:').strip())
	return similar[option][0]


def train(filepath, weights, asked_questions_data, answer_target_data):

	m = []
	with open(filepath, 'r') as file:
		for line in file:
			aux = line.split()
			m.append(aux)

	#print(m)

	for j in range(len(m[0])):
		asked = {}
		for i in range(len(m)):
			asked[i] = int(m[i][j])

		print(asked)

		#game.learn(asked, j, weights, asked_questions_data, answer_target_data)

def main():

	#game.get_data()

	asked_questions_data = game.load_asked_questions_data()
	answer_target_data = game.load_answer_target_data()

	#game.init()
	question_features = game.load_question_features()
	questions = game.load_questions()
	target_objects = game.load_target_objects()
	weights = game.load_weights()

	# for i in range(9):
	# 	train('data.in', weights, asked_questions_data, answer_target_data)

	# with open('test', 'a') as file:
	# 	file.write('Targets: \n')
	# 	for i in range(len(target_objects)):
	# 		file.write('[%d] %s \n' %(i,target_objects[i]) )

	# 	file.write('Questions: \n')
	# 	for i in range(len(questions)):
	# 		file.write('[%d] %s \n' %(i, questions[i]))


	# print(asked_questions_data)
	# print(answer_target_data)
	# for i in range(len(target_objects)):
	# 	print('[%d]%s: %d' %(i, target_objects[i], answer_target_data.count(i) ) )

	while(True):
		print('Targets: ')
		print(target_objects)
		print('Features: ')
		print(question_features)
		print('Questions: ')
		print(questions)
		print('Weights:')
		print(weights)
		asked_questions = {}
		initial_questions = game.load_initial_questions(questions)

		rank_target_objects = {}

		for target_object_id in range(len(target_objects)):
			rank_target_objects[target_object_id] = 0

		question_number = 0
		question_count = 0

		n = len(target_objects)
		r = random.randint(0, n - 1)
		print( 'try: %s', target_objects[r])

		while(True):
			print('asked questions:')
			print(asked_questions)
			tmp_weights = weights.copy()
			question_id = game.choose_next_question(rank_target_objects, target_objects ,tmp_weights, asked_questions, questions, initial_questions)
			print('question_id: %d' %(question_id) )
			print('rank:')
			game.print_top(rank_target_objects, target_objects)

			if( question_id == -1):
				choosen = game.guess(rank_target_objects, target_objects)

				if(choosen == None):
					weights = add_target(asked_questions, target_objects, weights, questions, question_features, asked_questions_data, answer_target_data)
					break

				print( 'Estas pensando en: ' + choosen['name'])
				success = input('Es correcto (y/n) : ').strip()
				if(success == 'y'):
					game.learn(asked_questions, choosen['id'], weights, asked_questions_data, answer_target_data)

				else:
					weights = add_target(asked_questions, target_objects, weights, questions, question_features, asked_questions_data, answer_target_data)

				break

			elif( question_number == 12 or question_number == 16 or question_number == 20 or question_number == 24 or question_number == 30):
				choosen = game.guess(rank_target_objects, target_objects)
				if(choosen == None):
					weights = add_target(asked_questions, target_objects, weights, questions, question_features, asked_questions_data, answer_target_data)
					break

				print( 'Estas pensando en: ' + choosen['name'])
				success = input('Es correcto (y/n) : ').strip()
				if(success == 'y'):
					game.learn(asked_questions, choosen['id'], weights, asked_questions_data, answer_target_data)
					break

				else:
					rank_target_objects[choosen['id']] = -float('inf')
					continuar = input('continuar:')
					if continuar == 'n':
						weights = add_target(asked_questions, target_objects, weights, questions, question_features, asked_questions_data, answer_target_data)
						break







			question_number += 1

			print('Pregunta ' + str(question_number) + ': '+ questions[question_id] + '?' )
			answer = int(input().strip())
			game.update_local_weights(question_id, answer, asked_questions, tmp_weights, rank_target_objects)

			if answer != 0:
				question_count += 1

		answer = input('Seguir jugando (y/n): ')
		if answer == 'n':
			break




if __name__ == '__main__' :
	main()

