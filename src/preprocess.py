# adapted from https://github.com/swabhs/open-sesame

from __future__ import division

'''
Reads XML files containing FrameNet 1.7 annotations, and converts them to a CoNLL 2009-like format.
'''

import codecs
import os.path
import sys

import tqdm
import xml.etree.ElementTree as et

from modeling import FrameAnnotation, SentenceAnnotation


VERSION = '1.7'
FNDIR = f'data/framenet_v{VERSION[0]+VERSION[-1]}/'
FULLTEXT_DIR = FNDIR + 'fulltext/'
OUTDIR = 'data/fn' + VERSION + '_parsed/'

TEST_FILES = [
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

DEV_FILES = [
        "ANC__110CYL072.xml",
        "KBEval__MIT.xml",
        "LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
        "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml",
        "Miscellaneous__Hijack.xml",
        "NTI__NorthKorea_NuclearOverview.xml",
        "NTI__WMDNews_062606.xml",
        "PropBank__TicketSplitting.xml",
        ]

logger = open(OUTDIR + "preprocess-fn." + VERSION + ".log", "w")

trainf = OUTDIR + 'fn' + VERSION + '.train.conll'
devf = OUTDIR + 'fn' + VERSION + '.dev.conll'
testf = OUTDIR + 'fn' + VERSION + '.test.conll'


relevantfelayers = ["Target", "FE"]
relevantposlayers = ["BNC", "PENN"]
ns = {'fn' : 'http://framenet.icsi.berkeley.edu'}

firsts = {devf: True,
          testf: True,
          trainf: True}

sizes = {devf: 0,
         testf: 0,
         trainf: 0}

totsents = numsentsreused = fspno = numlus = 0.0
isfirst = isfirstsent = True


def write_to_conll(outf, fsp, firstex, sentid):
    mode = "a"
    if firstex:
        mode = "w"

    with codecs.open(outf, mode, "utf-8") as outf:
        for i in range(fsp.sent.size()):
            token, postag, nltkpostag, nltklemma, lu, frm, role = fsp.info_at_idx(i)

            outf.write(str(i + 1) + "\t")  # ID = 0
            outf.write(str(token.encode('utf-8')) + "\t")  # TOKEN = 1
            outf.write(nltklemma + "\t")  # LEMMA = 2
            outf.write(postag + "\t" + nltkpostag + "\t")  # POS, PPOS = 3,4
            outf.write(str(sentid - 1) + "\t")  # SENT_NUM = 5
            outf.write(lu + "\t" + frm + "\t")  # LU, FRAME = 6,7
            outf.write(role + "\n")  #FRAME_ROLE = 8

        outf.write("\n")  # end of sentence
        outf.close()



def process_xml_labels(label, layertype):
    try:
        st = int(label.attrib["start"])
        en = int(label.attrib["end"])
    except KeyError:
        logger.write("\t\tIssue: start and/or end labels missing in " + layertype + "\n")
        return
    return (st, en)


def process_sent(sent):
    senttext = ""
    for t in sent.findall('fn:text', ns):  # not a real loop
        senttext = t.text

    sentann = SentenceAnnotation(senttext)

    for anno in sent.findall('fn:annotationSet', ns):
        for layer in anno.findall('fn:layer', ns):
            layertype = layer.attrib["name"]
            if layertype in relevantposlayers:
                for label in layer.findall('fn:label', ns):
                    startend = process_xml_labels(label, layertype)
                    sentann.add_token(startend)
                    sentann.add_postag(label.attrib["name"])
                if sentann.normalize_tokens(logger) is None:
                    logger.write("\t\tSkipping: incorrect tokenization\n")
                    return
                break
        if sentann.foundpos:
            break

    if not sentann.foundpos:
        # TODO do some manual tokenization
        logger.write("\t\tSkipping: missing POS tags and hence tokenization\n")
        return
    return sentann


def get_all_fsps_in_sent(sent, sentann, fspno, lex_unit, frame, isfulltextann, corpus):
    numannosets = 0
    fsps = {}
    fspset = set([])

    # get all the FSP annotations for the sentece : it might have multiple targets and hence multiple FSPs
    for anno in sent.findall('fn:annotationSet', ns):
        numannosets += 1
        if numannosets == 1:
            continue
        anno_id = anno.attrib["ID"]
        if isfulltextann: # happens only for fulltext annotations
            if "luName" in anno.attrib:
                if anno.attrib["status"] == "UNANN" and "test" not in corpus: # keep the unannotated frame-elements only for test, to enable comparison
                    continue
                lex_unit = anno.attrib["luName"]
                frame = anno.attrib["frameName"]
                if frame == "Test35": continue # bogus frame
            else:
                continue
            logger.write("\tannotation: " + str(anno_id) + "\t" + frame + "\t" + lex_unit + "\n")
        fsp = FrameAnnotation(lex_unit, frame, sentann)

        for layer in anno.findall('fn:layer', ns):  # not a real loop
            layertype = layer.attrib["name"]
            if layertype not in relevantfelayers:
                continue
            if layertype == "Target" :
                for label in layer.findall('fn:label', ns):  # can be a real loop
                    startend = process_xml_labels(label, layertype)
                    if startend is None:
                        break
                    fsp.add_target(startend, logger)
            elif layer.attrib["name"] == "FE" and layer.attrib["rank"] == "1":
                for label in layer.findall('fn:label', ns):
                    startend = process_xml_labels(label, layertype)
                    if startend is None:
                        if "itype" in label.attrib:
                            logger.write("\t\tIssue: itype = " + label.attrib["itype"] + "\n")
                            continue
                        else:
                            break
                    fsp.add_fe(startend, label.attrib["name"], logger)

        if not fsp.foundtarget:
            logger.write("\t\tSkipping: missing target\n")
            continue
        if not fsp.foundfes:
            logger.write("\t\tIssue: missing FSP annotations\n")
        if fsp not in fspset:
            fspno += 1
            fsps[anno_id] = fsp
            fspset.add(fsp)
        else:
            logger.write("\t\tRepeated frames encountered for same sentence\n")

    return numannosets, fspno, fsps


def get_annoids(filelist, outf):
    annos = []
    isfirstex = True
    fspno = 0
    numsents = 0
    invalidsents = 0
    repeated = 0
    totfsps = 0
    sents = set([])

    for tfname in tqdm.tqdm(filelist):
        tfname = os.path.join(FULLTEXT_DIR, tfname)
        logger.write("\n" + tfname + "\n")
        if not os.path.isfile(tfname):
            continue
        with codecs.open(tfname, 'rb', 'utf-8') as xml_file:
            tree = et.parse(xml_file)

        root = tree.getroot()
        for sentence in root.iter('{http://framenet.icsi.berkeley.edu}sentence'):
            numsents += 1
            logger.write("sentence:\t" + str(sentence.attrib["ID"]) + "\n")
            for annotation in sentence.iter('{http://framenet.icsi.berkeley.edu}annotationSet'):
                if "luName" in annotation.attrib and "frameName" in annotation.attrib:
                    annos.append(annotation.attrib["ID"])
            # get the tokenization and pos tags for a sentence
            sentann = process_sent(sentence)
            if sentann is None:
                invalidsents += 1
                logger.write("\t\tIssue: Token-level annotations not found\n")
                continue

            # get all the different FSP annotations in the sentence
            x, fspno, fsps = get_all_fsps_in_sent(sentence, sentann, fspno, None, None, True, outf)
            totfsps += len(fsps)
            if len(fsps) == 0: invalidsents += 1
            if sentann.text in sents:
                repeated += 1
            for fsp in list(fsps.values()):
                sents.add(sentann.text)
                write_to_conll(outf, fsp, isfirstex, numsents)
                sizes[outf] += 1
                isfirstex = False
        xml_file.close()
    sys.stderr.write("# total sents processed = %d\n" % numsents)
    sys.stderr.write("# repeated sents        = %d\n" % repeated)
    sys.stderr.write("# invalid sents         = %d\n" % invalidsents)
    sys.stderr.write("# sents in set          = %d\n" % len(sents))
    sys.stderr.write("# annotations           = %d\n" % totfsps)
    return annos


def process_fulltext():
    sys.stderr.write("\nReading {} fulltext data ...\n".format(VERSION))

    # read and write all the test examples in conll
    logger.write("\n\nTEST\n\n")
    sys.stderr.write("TEST\n")
    test_annos = get_annoids(TEST_FILES, testf)

    # read and write all the dev examples in conll
    logger.write("\n\nDEV\n\n")
    sys.stderr.write("DEV\n")
    dev_annos = get_annoids(DEV_FILES, devf)

    # read all the full-text train examples in conll
    train_files = []
    for f in os.listdir(FULLTEXT_DIR):
        if f not in TEST_FILES and f not in DEV_FILES and not f.endswith("xsl"):
            train_files.append(f)
    logger.write("\n\nTRAIN\n\n")
    sys.stderr.write("TRAIN\n")
    get_annoids(train_files, trainf)

    return dev_annos, test_annos



if __name__ == "__main__":
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    dev, test = process_fulltext()

    logger.close()
