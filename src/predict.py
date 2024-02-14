from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from spacy import displacy


def predict(pretrained_model, sentence, task, visualize):
    model = AutoModelForTokenClassification.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if visualize:
        token_classifier = pipeline(
            "token-classification", model=model, aggregation_strategy="average" if task == 'frames' else 'simple', tokenizer=tokenizer)
        ents = [{'start': d['start'], 'end': d['end'], 'label': d['entity_group']} for d in token_classifier(sentence) if d['entity_group'] not in ['None', 'Not a target']]
        dic_ents = {
            "text": sentence,
            "ents": ents,
            "title": None
        }

        displacy.render(dic_ents, manual=True, style="ent")
    else:
        token_classifier = pipeline(
            "token-classification", model=model, aggregation_strategy="none", tokenizer=tokenizer)
        return token_classifier(sentence)