from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

#def find(tensor, values):
#    return torch.nonzero(tensor[..., None] == values)

def prepare():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")

    model.eval()
    model.to('cuda')

    chat_histories_ids = [torch.Tensor().to('cuda', dtype=torch.long) for x in range(37)]
    return chat_histories_ids

def run(raw_message, chat_history_ids):
    chat_history_ids = chat_history_ids.clone().detach()

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(raw_message + tokenizer.eos_token, return_tensors='pt').to('cuda', dtype=torch.long)

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1).to('cuda')

    # rewriting history while generating a response while limiting the total chat history to 50 tokens
    chat_history_ids = model.generate(bot_input_ids, max_length=bot_input_ids.shape[-1]+50, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=10).to('cuda')

    # parsing the last sentence from whole history
    last_eos_idx = (chat_history_ids[:,bot_input_ids.shape[-1]:][0]==50256).nonzero()[0].squeeze()
    last_eos_idx += bot_input_ids.shape[-1]
    chat_history_ids = chat_history_ids[:,:last_eos_idx+1]
   
    # pretty print last ouput tokens from bot
    reply = tokenizer.decode(chat_history_ids[:,bot_input_ids.shape[-1]:last_eos_idx+1][0], skip_special_tokens=True)
    
    #print("HISTORY {}".format(tokenizer.decode(chat_history_ids[:,:][0], skip_special_tokens=False)))
    return reply

def add_history(raw_message, raw_reply, chat_history_ids):
    # adding user's message to given history
    user_input_ids = tokenizer.encode(raw_message + tokenizer.eos_token, return_tensors='pt').to('cuda', dtype=torch.long)
    chat_history_ids = torch.cat([chat_history_ids, user_input_ids], dim=-1).to('cuda')
    
    # adding chatbot's reply to given history
    chat_reply_ids = tokenizer.encode(raw_reply+ tokenizer.eos_token, return_tensors='pt').to('cuda', dtype=torch.long)
    chat_history_ids = torch.cat([chat_history_ids, chat_reply_ids], dim=-1).to('cuda')

    #print("HISTORY {}".format(tokenizer.decode(chat_history_ids[:,:][0], skip_special_tokens=False)))
    #print('len(chat_history_ids)', len(chat_history_ids[0]))
    #print('size', chat_history_ids.size())
    if len(chat_history_ids[0]) > 100:
        chat_history_ids = torch.narrow(chat_history_ids, 1, len(chat_history_ids[0]) - 100, 100)
    #print('len(chat_history_ids)', len(chat_history_ids[0]))

    return chat_history_ids

if __name__=='__main__':
    chat_histories_ids = prepare()
    print('ready when you are.')
    chat_history_ids = chat_histories_ids[0] # choose your chatbot by id

    while True:
        message = input()
        reply = run(message, chat_history_ids)
        print(reply)
        chat_history_ids = add_history(message, reply, chat_history_ids)
