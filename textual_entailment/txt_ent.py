import logging
import json

logging.basicConfig(level=logging.ERROR)
logging.getLogger("allennlp").setLevel(logging.ERROR)

def prepare():
    global predictor
    from allennlp.predictors.predictor import Predictor
    import allennlp_models.nli
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/snli-roberta-large-2020.02.27.tar.gz", predictor_name="textual-entailment")

    with open("agents.json") as jsonfile:
        agents = json.load(jsonfile)
    return {'message': [[] for _ in agents["agent"]], "reply": [[] for _ in agents["agent"]] }

def load_documents():
    documents = [agent["document"] for agent in agents["agent"]]
    return documents

def run(hypothesis, premise):
    return predictor.predict(hypothesis=hypothesis, premise=premise)

if __name__=='__main__':
    print(prepare())
    print('loaded.')
    result = run("Two women are sitting on a blanket near some rocks talking about politics.", "Two women are wandering along the shore drinking iced tea.")
    print(result)
