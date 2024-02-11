from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from spacy import displacy


def predict_triggers(pretrained_model, sentence):
    model = AutoModelForTokenClassification.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    token_classifier = pipeline(
        "token-classification", model=model, aggregation_strategy="none", tokenizer=tokenizer)

    tagged_sentence = ""
    for token in token_classifier(sentence):
        word = token['word']
        if word.startswith('##'):
            tagged_sentence += word[2:]
            continue
        if token['entity'] == 'Target':
            word += '*'
        tagged_sentence +=  ' ' + word
    
    #tagged_sentence = tagged_sentence.split()
    
    #for i, word in enumerate(tagged_sentence):
        #if '*' in word and not word.endswith('*'):
            #tagged_sentence[i] = word.replace("*", "") + "*"

    #return ' '.join(tagged_sentence)
    return tagged_sentence.strip()

def predict_frames(pretrained_model, sentence, visualize):
    model = AutoModelForTokenClassification.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    token_classifier = pipeline(
        "token-classification", model=model, aggregation_strategy="simple", tokenizer=tokenizer)

    ents = [{'start': d['start'], 'end': d['end'], 'label': d['entity_group']} for d in token_classifier(sentence) if d['entity_group'] != 'None']

    if visualize:
        dic_ents = {
            "text": sentence,
            "ents": ents,
            "title": None
        }

        displacy.render(dic_ents, manual=True, style="ent")
    else:
        return token_classifier(sentence)