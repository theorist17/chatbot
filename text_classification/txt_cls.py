import nltk
#nltk.download('nps_chat')
#nltk.download('punkt')
posts = nltk.corpus.nps_chat.xml_posts()[:10000]

def show_all():
    print(set(post.get('class') for post in posts))

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

def prepare():
    global classifier, dialogue_act_features

    featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

def append_punc(raw_text, txt_cls):
    if raw_text[-1] in '.?!':
        return raw_text

    if txt_cls in ['Emphasis', 'yAnswer', 'Statement', 'Clarify', 'nAnswer', 'Continuer', 'Reject', 'System', 'Accept', 'Emotion', 'Other']:
        raw_text += '.'
    elif txt_cls in ['Greet', 'Bye']:
        raw_text += '!'
    elif txt_cls in ['ynQuestion', 'whQuestion']:
        raw_text += '?'

    return raw_text

def infer(raw_text):
    global classifier, dialogue_act_features
    return classifier.classify(dialogue_act_features(raw_text))

