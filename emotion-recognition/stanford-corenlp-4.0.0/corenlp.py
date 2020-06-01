import os
import psutil
import sys
from pycorenlp import StanfordCoreNLP
from multiprocessing import Process

def start_java():
    if os.fork() != 0:
        return
    curpath = os.getcwd()
    dirpath = os.path.abspath(__file__).rsplit('/', 1)[0]

    os.chdir(dirpath)
    if not "java" in (p.name() for p in psutil.process_iter()):
        os.system('java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000 &')
    os.chdir(curpath)

def prepare():
    global nlp
    child = Process(target=start_java)
    child.start()
    child.join()
    nlp = StanfordCoreNLP('http://localhost:9000')

def infer(raw_text):
    global nlp
    #res = nlp.annotate("I love you. I hate him. You are nice. He is dumb. I want to know if moeny can buy happiness.",
    res = nlp.annotate(raw_text,
            properties={
                'annotators': 'sentiment',
                'outputFormat': 'json',
                'timeout': 1000,
                })
    sentimentValue = 0
    for s in res["sentences"]:
        sentimentValue += int(s["sentimentValue"]) - 2
        print("%d: '%s': %s %s" % (
            s["index"],
            " ".join([t["word"] for t in s["tokens"]]),
            int(s["sentimentValue"]) - 2, s["sentiment"]))
    print('sentimentValue', sentimentValue)
    sentimentValue = float(sentimentValue) / len(res["sentences"])
    return sentimentValue

def _cleanup():
    print('cleanup')
    PROCNAME = 'java'
    for proc in psutil.process_iter():
        try:
            if proc.name() == PROCNAME:
                p = psutil.Process(proc.pid)
                p.kill()
        except:
              pass

if __name__ == '__main__':
    cleanup = _cleanup
    try:
        prepare()
        print(infer(input()))
    except:
        pass
    else:
        cleanup = lambda: None
        pass
    cleanup()

