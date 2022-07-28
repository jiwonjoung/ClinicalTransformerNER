# we will use DDI data set at https://github.com/isegura/DDICorpus
training_root = "./ddi/DDICorpusBrat/Train/MedLine"
testing_root = "./ddi/DDICorpusBrat/Test/MedLine"

# link pacakge to python path
# import necessary functions
import sys
sys.path.append("./NLPreprocessing")
sys.path.append("./NLPreprocessing/text_process")
sys.path.append("./NLPpreprocessing/text_process/sentence_tokenization.py")

import logging
from annotation2BIO import generate_BIO, pre_processing, read_annotation_brat, BIOdata_to_file, logger
from sentence_tokenization import logger as logger1

# change log level to error to avoid too much log information in jupyter notebook
logger1.setLevel(logging.ERROR)
logger.setLevel(logging.ERROR)

file_ids = set()

for fn in Path(training_root).glob("*.ann"):
    file_ids.add(fn.stem)
    
len(file_ids)
# generate BIO from brat annotation
train_root = Path(training_root)
# train_bio = "./2018n2c2/bio/trains"
train_bio = "./ddi/ddi_bio/"
output_root = Path(train_bio)
output_root.mkdir(parents=True, exist_ok=True)

for fid in file_ids:
    txt_fn = train_root / (fid + ".txt")
    ann_fn = train_root / (fid + ".ann")
    bio_fn = output_root / (fid + ".bio.txt")
    
    txt, sents = pre_processing(txt_fn)
    e2idx, entities, rels = read_annotation_brat(ann_fn)
    nsents, sent_bound = generate_BIO(sents, entities, file_id=fid, no_overlap=False)
    
    BIOdata_to_file(bio_fn, nsents)

 # now we have to split the train and dev sets
# for transformer NER, we need to name these two datasets as train.txt and dev.txt
from sklearn.model_selection import train_test_split

file_ids = list(file_ids)
train_ids, dev_ids = train_test_split(file_ids, train_size=0.9, random_state=13, shuffle=True)
len(train_ids), len(dev_ids)
import fileinput

merged = output_root / "merge" # this will the final data dir we use for training
merged.mkdir(exist_ok=True, parents=True)

# train
with open(merged / "train.txt", "w") as f:
    for fid in train_ids:
        f.writelines(fileinput.input(output_root / (fid + ".bio.txt")))
    fileinput.close()
        
# dev
with open(merged /"dev.txt", "w") as f:
    for fid in dev_ids:
        f.writelines(fileinput.input(output_root / (fid + ".bio.txt")))
    fileinput.close()