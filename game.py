import numpy as np
import pickle, random

NEW_QUESTION_SCALE = 5
DISAGREE_SCALE = 5
AGREE_SCALE = 2

TARGET_DATA_FILE = 'data/target_data'
QUESTION_DATA_FILE = 'data/question_data'
FEATURE_DATA_FILE = 'data/feature_data'
WEIGHTS_DATA_FILE = 'data/weight_data.npy'
ASKED_QUESTIONS_DATA_FILE = 'data/asked_questions_data'
ANSWER_TARGET_DATA_FILE = 'data/answer_target_data'


def init():
	question_features = {}
	target_objects = []
	questions = []
	weights = np.zeros((0,0), dtype = int)

	update_target_objects_data(target_objects)
	update_questions_data(questions)
	update_weights_data(weights)
	update_question_features_data(question_features)

def get_data():
	asked_questions_data = []
	answer_target_data = []
	weights = target_objects = load_weights()
	n, m = weights.shape

	for i in range(m):
		asked_questions = {}
		for j in range(n):
			value = weights[j][i]
			if value > 0:
				asked_questions[j] = 1
			elif value < 0:
				asked_questions[j] = -1
		asked_questions_data.append(asked_questions)
		answer_target_data.append(i)
		update_asked_questions_data(asked_questions_data)
		update_answer_target_data(answer_target_data)



def load_question_features():
	with open(FEATURE_DATA_FILE, 'rb') as file:
		question_features = pickle.load(file)
	return question_features

def load_questions():
	with open(QUESTION_DATA_FILE, 'rb') as file:
		questions = pickle.load(file)
	return questions

def load_target_objects():
	with open(TARGET_DATA_FILE, 'rb') as file:
		target_objects = pickle.load(file)
	return target_objects

def load_weights():
	with open(WEIGHTS_DATA_FILE, 'rb') as file:
		weights = np.load(file)
	return weights

def load_asked_questions_data():
	with open(ASKED_QUESTIONS_DATA_FILE, 'rb') as file:
		asked_questions_data = pickle.load(file)
	return asked_questions_data

def load_answer_target_data():
	with open(ANSWER_TARGET_DATA_FILE, 'rb') as file:
		answer_target_data = pickle.load(file)
	return answer_target_data

def update_asked_questions_data(asked_questions_data):
	with open(ASKED_QUESTIONS_DATA_FILE, 'wb') as file:
		pickle.dump(asked_questions_data, file)

def update_answer_target_data(answer_target_data):
	with open(ANSWER_TARGET_DATA_FILE, 'wb') as file:
		pickle.dump(answer_target_data, file)

def update_target_objects_data(target_objects):
	with open(TARGET_DATA_FILE, 'wb') as file:
		pickle.dump(target_objects, file)

def update_questions_data(questions):
	with open(QUESTION_DATA_FILE, 'wb') as file:
		pickle.dump(questions, file)

def update_question_features_data(question_features):
	with open(FEATURE_DATA_FILE, 'wb') as file:
		pickle.dump(question_features, file)

def update_weights_data(weights):
	with open(WEIGHTS_DATA_FILE, 'wb') as file:
		weights.dump(file)

def load_initial_questions(questions, count = 3):
	n = len(questions)
	count = min(count, n)
	initial_questions = []
	while count > 0 :
		question_id = random.randint(0, n - 1)
		if question_id in [82,87,49,113,19, 114, 116, 141, 143, 145] :
			continue
		initial_questions.append(question_id)
		count -= 1

	return initial_questions

def add_target_object(name, target_objects, current_weights):
	name = name.strip()
	target_objects.append(name)
	update_target_objects_data(target_objects)
	n, m = current_weights.shape
	aux = np.zeros( (n, m + 1), dtype = int)
	aux[ :, :-1] = current_weights.copy() 
	current_weights = aux.copy()

	return {'target_object_id': m, 'weights': current_weights}

def add_question(type, value, questions, question_features, current_weights):
	if type == 'ingredients':
		question = 'Contiene ' + value
	elif type == 'flavor':
		question = 'Es de un sabor ' + value
	elif type == 'target_type':
		question = 'Es un(a) ' + value
	elif type == 'country':
		question = 'Es un plato de ' + value
	else:
		question = 'Esta relacionado con ' + value

	question_id = len(questions)
	questions.append(question)
	question_features[value] = question_id

	update_questions_data(questions)
	update_question_features_data(question_features)
	n, m = current_weights.shape
	aux = np.zeros( (n + 1, m), dtype = int)
	aux[ :-1, :] = current_weights.copy()
	current_weights = aux.copy()
	return {'question_id': question_id, 'weights': current_weights}

def add_features(features, questions, question_features, current_weights, asked_questions):
	for key, values in features.items():
		for value in values:
			value = value.strip()
			if value == '':
				continue

			answer = 1
			try:
			 	question_id =question_features[value]
			 	asked_questions[question_id] = answer
			except KeyError as e:
				result = add_question(key, value, questions, question_features, current_weights)
				question_id = result['question_id'] 
				current_weights = result['weights']
				asked_questions[question_id] = answer * NEW_QUESTION_SCALE

	return current_weights


def guess(rank_target_objects, target_objects):
	top_target_objects = get_top_targets(rank_target_objects, target_objects, top_count = 1)
	if len(top_target_objects) == 0:
		return None
	target = top_target_objects[0]
	return {'id': target[1], 'name':target[2]}

def softmax(x):
	#x = np.array(x,dtype = int)
	return np.exp(x)/ np.sum(np.exp(x))

def normalize(x):
	x = np.array(x,dtype = int)
	return (x - np.min(x)) / (np.max(x) - np.min(x))

def check_finish(rank_target_objects, target_objects, top_count = 10):
	top_targets = get_top_targets(rank_target_objects, target_objects, top_count)
	score = np.array(top_targets)
	# print('Score: ')
	# print(type(score))
	rank = softmax(normalize(score[:,0]))
	if( rank[0] > 0.172):
		return True
	return False

def print_top(rank_target_objects, target_objects, top_count = 10):
	top_targets = get_top_targets(rank_target_objects, target_objects, top_count)
	print(type(top_targets))
	score = np.array(top_targets)
	print('Score: ')
	print(type(score))
	rank = softmax(normalize(score[:,0]))
	for i in range( len(top_targets) ):
		top_targets[i] = rank[i],top_targets[i][1], top_targets[i][2]
	print( top_targets )


def get_top_targets(rank_target_objects, target_objects, top_count = 10):
	top_count = min(top_count, len(rank_target_objects))
	return sorted([[value, key, target_objects[key]] for key, value in rank_target_objects.items()], reverse = True)[:top_count]

def update_local_weights(question_id, answer, asked_questions, tmp_weights, rank_target_objects):

	n, m = tmp_weights.shape

	for target_object_id in range(m):
		value = tmp_weights[question_id][target_object_id]
		if value == 0:
			continue

		if (value > 0 and answer < 0) or (value < 0 and answer > 0):
			tmp_weights[question_id][target_object_id] -= answer
		else:
			tmp_weights[question_id][target_object_id] += answer

		rank_target_objects[target_object_id] += answer * tmp_weights[question_id][target_object_id]

	asked_questions[question_id] = answer

def information_gain_entropy(top_target_objects, question_id, current_weights):
	question_weights = current_weights[question_id]
	question_weights_for_top_objects = question_weights[top_target_objects]

	positives = (question_weights_for_top_objects > 0).sum()
	negatives = (question_weights_for_top_objects < 0).sum()
	unknows = (question_weights_for_top_objects == 0).sum()

	total = positives + negatives + unknows

	h_positives = 0
	h_negatives = 0

	if positives != 0 :
		h_positives = -(positives/total)*math.log(positives/total, 2)
	
	if negatives != 0:
		h_negatives = -(negatives/total)*math.log(negatives/total, 2)

	entropy = h_negatives + h_positives
	entropy *= (positives + negatives)/total

	if entropy != 0:
		return 1 / entropy

	return float('inf')

def entropy(top_target_objects, question_id, current_weights):
	question_weights = current_weights[question_id]
	question_weights_for_top_objects = question_weights[top_target_objects]

	positives = (question_weights_for_top_objects > 0).sum()
	negatives = (question_weights_for_top_objects < 0).sum()
	unknows = (question_weights_for_top_objects == 0).sum()

	return abs(positives - negatives) + 5 * unknows

def choose_next_question(rank_target_objects, target_objects, current_weights, asked_questions, questions, initial_questions):

	# if len(initial_questions) > 0 :
	# 	question_id = initial_questions.pop()
	# 	return question_id

	# print('Inside Baby')

	top_target_objects = [i for i in range( len(target_objects) )]

	n = len(asked_questions)

	if n > 0:
		count = 64 // (2 ** n)
		count = max(count, 10)
		aux = get_top_targets(rank_target_objects, target_objects)
		top_target_objects = [ target_object[1] for target_object in aux]
		print('count: %d' %count)

	best_entropy = float('inf')
	best_question_id = -1
	for question_id in range(len(questions)):
		if question_id in asked_questions:
			continue

		if question_id in [82,87,86, 49,113,19]:
			continue

		question_entropy = entropy(top_target_objects, question_id, current_weights)
		#print( questions[question_id], question_entropy)
		if(question_entropy < best_entropy):
			best_question_id, best_entropy = question_id, question_entropy

	return best_question_id


def learn(asked_questions, target_object_id, current_weights, asked_questions_data, answer_target_data):
	for question_id, answer in asked_questions.items():

		value = current_weights[question_id, target_object_id]

		if (value > 0 and answer < 0) or (value < 0 and answer > 0) :
			current_weights[question_id, target_object_id] -= answer
		else:
			current_weights[question_id, target_object_id] += answer

	asked_questions_data.append(asked_questions)
	answer_target_data.append(target_object_id)
	update_weights_data(current_weights)
	update_asked_questions_data(asked_questions_data)
	update_answer_target_data(answer_target_data)


def learn_new_target(asked_questions, name, target_objects, current_weights, asked_questions_data, answer_target_data):
	name = name.strip()

	if( name == '' ):
		return

	result = add_target_object(name, target_objects, current_weights)
	target_object_id = result['target_object_id']
	current_weights = result['weights']
	learn(asked_questions, target_object_id, current_weights, asked_questions_data, answer_target_data)
	return current_weights
