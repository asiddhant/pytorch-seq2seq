from __future__ import print_function
import argparse
import os
import shutil
from string import punctuation

parser = argparse.ArgumentParser()
parser.add_argument('--inputdir', help="data directory", default="/home/ubuntu/ijcnlp_dailydialog")
parser.add_argument('--storedir', help="data directory", default="../datasets")
args = parser.parse_args()

def preprocess_dataset(root, name, file):
    path = os.path.join(root, name)
    if not os.path.exists(path):
        os.mkdir(path)
    final_output=[]
    with open(os.path.join(args.inputdir,file)) as f:
        for line in f:
            conv = line.strip().lower().split('__eou__')
            conv = [''.join(c for c in s if c not in punctuation) for s in conv]
            conv = [' '.join(s.split()) for s in conv]
            output = [conv[i].strip()+'\t'+conv[i+1].strip() 
                      for i in range(len(conv)-2) 
                      if len(conv[i].strip()) and len(conv[i+1].strip())]
            final_output += output
            
    with open(os.path.join(path,'data.txt'),'w+') as of:
        for item in final_output:
            of.write("%s\n" % item)
    
    words=set()
    for sentence in final_output:
        words.update(sentence.replace('\t',' ').split(' '))        
    words=list(words)
    
    src_vocab = os.path.join(path, 'vocab.source')
    with open(src_vocab, 'w+') as fout:
        fout.write("\n".join(words))
    tgt_vocab = os.path.join(path, 'vocab.target')
    shutil.copy(src_vocab, tgt_vocab)
    
if __name__ == '__main__':
    data_dir = args.storedir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    daily_dialog = os.path.join(data_dir, 'daily_dialog')
    if not os.path.exists(daily_dialog):
        os.mkdir(daily_dialog)

    preprocess_dataset(daily_dialog, 'train', 'train/dialogues_train.txt')
    preprocess_dataset(daily_dialog, 'dev', 'validation/dialogues_validation.txt')
    preprocess_dataset(daily_dialog, 'test', 'test/dialogues_test.txt')