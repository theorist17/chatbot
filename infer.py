import socket
import sys
import os
import re

print('loading txtcls..')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "chatbot/text_classification"))
import txt_cls as txtcls
txtcls.prepare()
print('done.')

print('loading emorec..')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "chatbot/emotion-recognition"))
import emotion_recognition as emorec 
print('done.')

print('loading qna..')
import qna
documents, histories, confidences = qna.prepare()
print('done.')

print('loading persona..')
import persona
personalities = persona.prepare()
print('done.')

print('loading dialo..')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "chatbot/dialogpt"))
import dialo
chat_histories_ids = dialo.prepare()
print('done.')

print('loading txtent..')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "chatbot/textual_entailment"))
import txt_ent as txtent
ent_histories = txtent.prepare()
print('done.')

print('booting server..')
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('203.252.157.78', 10000)
sock.bind(server_address)
sock.listen(1)
print('done.')

def update_agents():
    global documents, personalities, confidences

    print('updating agents.json..')
    documents = qna.load_documents()
    confidences = qna.load_confidences()
    personalities = persona.load_personalities()
    print('done.')
    print()

while True:
    print('waiting for a connection')
    connection, client_address = sock.accept()
    try:
        print('\nconnection from', client_address, '\n')

        message = connection.recv(1024).decode()
        if message:
            # spliting message into agent id and actual message
            aid, message = message.split(' ', 1)
            
            # parsing agent update command
            if message.rstrip() == 'I love you.':
                update_agents()
                connection.sendall(''.encode())
                continue

            # text classfication for user message
            msg_cls = txtcls.infer(message)
            message = txtcls.append_punc(message, msg_cls)


            # loading per-agent runtime data
            aid = int(aid)
            personality = personalities[aid]
            chat_history_ids = chat_histories_ids[aid]

            nlg = False
            if msg_cls in ['ynQuestion', 'whQuestion']:
                ans = qna.run(message, documents[aid])
                if float(ans['confidence']) > float(confidences[aid]):
                    reply = ans['reply']
                    print('*'*10, 'qna', '*'*10)
                else:
                    nlg = True
            elif msg_cls in ['Greet']:
                reply = persona.run(message, personality, histories[aid])
                print('*'*10, 'greetings', '*'*10)
            elif msg_cls in ['Bye']:
                import random
                reply = random.choice(['bye.', 'bye, bye.', 'have a good day.', 'good bye.', 'have a nice day.', 'take care.', 'good day.', 'good luck.'])
                print('*'*10, 'farewells ', '*'*10)
            else: # Statement
                nlg = True

            if nlg:
                dialo_rpl = dialo.run(message, chat_history_ids)
                a = float(txtent.run(documents[aid], dialo_rpl)['probs'][0])
                b = float(txtent.run(' '.join(ent_histories['message'][aid])+' '+message, dialo_rpl)['probs'][0])
                c = 0.5*float(txtent.run(message, dialo_rpl)['probs'][0])
                dialo_scr = a + b + c
                print('dialog', dialo_scr, a, b, c, dialo_rpl)

                persona_rpl = persona.run(message, personality, histories[aid])
                a = float(txtent.run(documents[aid], persona_rpl)['probs'][0])
                b = float(txtent.run(' '.join(ent_histories['message'][aid])+' '+message, persona_rpl)['probs'][0])
                c = 0.5*float(txtent.run(message, persona_rpl)['probs'][0])
                persona_scr = a + b + c
                print('persona', persona_scr, a, b, c, persona_rpl)

                if dialo_scr > persona_scr:
                    reply = dialo_rpl
                    print('*'*10, 'dialo', '*'*10)
                else:
                    reply = persona_rpl
                    print('*'*10, 'persona', '*'*10)

            # text classification for agent reply
            rpl_cls = txtcls.infer(reply)
            reply = txtcls.append_punc(reply, rpl_cls)

            # sentiment analysis & classification
            emotion = emorec.run(reply)

            # sending reply to client
            reply_sent = reply + " " + emotion 
            connection.sendall(reply_sent.encode())

            # updating history
            histories[aid] = persona.add_history(message, reply, personality, histories[aid])
            chat_histories_ids[aid] = dialo.add_history(message, reply, chat_history_ids)
            ent_histories['message'][aid].append(message)
            ent_histories['reply'][aid].append(reply)
            
            # for debugging
            print('message :', message)
            print('message class :', msg_cls)
            print('reply :', reply)
            print('reply class :', rpl_cls)
            if msg_cls in ['ynQuestion', 'whQuestion']:
                if float(ans['confidence']) > float(confidences[aid]):
                    print("qna confident answer :", ans['answer'])
                else:
                    print("qna unvalid answer :", ans['answer'])
                print("qna confidence :", ans['confidence'])
            print('sent :', reply_sent)
            print('emotion :', emotion)
            print('agent id :', aid)
            print('history :', persona.translate(histories[aid]))
            print('history :', histories[aid])
            print('chat_history :', chat_histories_ids[aid])
            print('messages :', ' '.join(ent_histories['message'][aid]))
            print('replies :', ' '.join(ent_histories['reply'][aid]))
            print('persona :', persona.translate(personality))
            print("document :", ''.join(documents[aid]).replace('.', ".\n"))
            print()
        else:
            print('no data from', client_address)
    finally:
        connection.close()
