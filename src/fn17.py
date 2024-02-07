from datasets import load_dataset, Dataset, DatasetDict
from copy import deepcopy
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.corpus import framenet as fn
import nltk

# from https://github.com/swabhs/open-sesame/blob/master/sesame/globalconfig.py
SESAME_TEST_FILES = [
    "ANC__110CYL067.xml",
    "ANC__110CYL069.xml",
    "ANC__112C-L013.xml",
    "ANC__IntroHongKong.xml",
    "ANC__StephanopoulosCrimes.xml",
    "ANC__WhereToHongKong.xml",
    "KBEval__atm.xml",
    "KBEval__Brandeis.xml",
    "KBEval__cycorp.xml",
    "KBEval__parc.xml",
    "KBEval__Stanford.xml",
    "KBEval__utd-icsi.xml",
    "LUCorpus-v0.3__20000410_nyt-NEW.xml",
    "LUCorpus-v0.3__AFGP-2002-602187-Trans.xml",
    "LUCorpus-v0.3__enron-thread-159550.xml",
    "LUCorpus-v0.3__IZ-060316-01-Trans-1.xml",
    "LUCorpus-v0.3__SNO-525.xml",
    "LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml",
    "Miscellaneous__Hound-Ch14.xml",
    "Miscellaneous__SadatAssassination.xml",
    "NTI__NorthKorea_Introduction.xml",
    "NTI__Syria_NuclearOverview.xml",
    "PropBank__AetnaLifeAndCasualty.xml",
]

SESAME_DEV_FILES = [
    "ANC__110CYL072.xml",
    "KBEval__MIT.xml",
    "LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
    "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml",
    "Miscellaneous__Hijack.xml",
    "NTI__NorthKorea_NuclearOverview.xml",
    "NTI__WMDNews_062606.xml",
    "PropBank__TicketSplitting.xml",
]


def flatten(df):
    """combine a dataframe group to a one-line instance.

    Args:
        df ([dataframe]): a dataframe, represents a group

    Returns:
        a one-line dict
    """
    label_values = np.stack(df['frame_tags'].values)
    processed = np.zeros_like(label_values)

    for token_id in range(label_values.shape[1]):
        labels = label_values[:, token_id]
        for i in range(len(labels)):
            if labels[i] != 0:
                processed[i, token_id] = labels[i]
                break
    aggregated_tags = processed.sum(axis=0)
    result = df.iloc[0].to_dict()
    result['frame_tags'] = aggregated_tags

    return result


def combine(ds, group_column, ):
    ds = ds.to_pandas()
    combined = []
    for sent_id, group in ds.groupby(group_column):
        combined.append(flatten(group))
    return Dataset.from_pandas(pd.DataFrame(combined))


def load_dataset_hf(flatten=True):
    ds = load_dataset('liyucheng/FrameNet_v17')
    if flatten:
        flat_ds = deepcopy(ds)
        for k, v in flat_ds.items():
            flat_ds[k] = combine(v, 'sent_id')
        return flat_ds
    else:
        return ds



def get_token_indices(sentence):
    token_indices = {}
    start_index = 0
    for word in sentence.split():
        start = sentence.find(word, start_index)
        end = start + len(word)
        token_indices[(start, end)] = word
        start_index = end
    return token_indices


def parse_annotated_sentence_from_framenet_sentence(
    fn_sentence
):
    sentence_text = fn_sentence["text"]
    tokens = sentence_text.split()
    token_indices = get_token_indices(sentence_text)
    whitelist_sentence = False
    for fn_annotation in fn_sentence["annotationSet"]:
        if (
            "FE" in fn_annotation
            and "Target" in fn_annotation
            and "frame" in fn_annotation
        ):
            trigger_locs = fn_annotation["Target"]
            # filter out broken annotations
            for trigger_loc in trigger_locs:
                if trigger_loc[0] >= len(sentence_text):
                    return None
            for i, target in enumerate(trigger_locs):
                if i > 0:
                    token_indices[target] = 'I-' + fn_annotation["frame"]["name"]
                else:
                    token_indices[target] = 'B-' + fn_annotation["frame"]["name"]
            whitelist_sentence = True
    
    tagged_sentence = ' '.join(token_indices[indices] for indices in sorted(token_indices))
    frames = ['O' if t[:2] not in ['B-', 'I-'] else t for t in tagged_sentence.split()]
    if whitelist_sentence:
        return {
            "text": sentence_text,
            "tokens": tokens,
            "frames": frames,
            #really bad
            "frame_tags": [{fe['name']: fe["ID"] for fe in fn.frames()}[f[2:]] if f[:2] in ['B-', 'I-'] else 0 for f in frames],
            "tagged_sentence": tagged_sentence,
        }
    return None

def parse_annotated_sentences_from_framenet_doc(fn_doc):
    annotated_sentences = []
    for sentence in fn_doc["sentence"]:
        annotated_sentence = parse_annotated_sentence_from_framenet_sentence(sentence)
        if annotated_sentence:
            annotated_sentences.append(annotated_sentence)
    return annotated_sentences

def load_framenet_samples(include_docs=None, exclude_docs=None):
    samples = []
    for doc in tqdm(fn.docs()):
        if exclude_docs and doc["filename"] in exclude_docs:
            continue
        if include_docs and doc["filename"] not in include_docs:
            continue
        samples += parse_annotated_sentences_from_framenet_doc(doc)
    return samples


def load_training_data():
    return load_framenet_samples(
        exclude_docs=SESAME_DEV_FILES + SESAME_TEST_FILES
    )

def load_test_data():
    return load_framenet_samples(include_docs=SESAME_TEST_FILES)

def load_validation_data():
    return load_framenet_samples(include_docs=SESAME_DEV_FILES)


def load_dataset_nltk():
    nltk.download("framenet_v17")

    ds = DatasetDict()

    val_data = load_validation_data()
    test_data = load_test_data()
    train_data = load_training_data()

    ds['train'] = Dataset.from_list(train_data)
    ds['dev'] = Dataset.from_list(val_data)
    ds['test'] = Dataset.from_list(test_data)

    return ds