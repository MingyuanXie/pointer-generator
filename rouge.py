# -*- coding: utf-8 -*-
import sys
import os
import decode
# import pyrouge
import shutil


def abstract2id(article, wd2idx):
	text = article.split(" ");
	result = []
	for w in text:
		if w not in wd2idx:
			wd2idx[w] = len(wd2idx)
		result.append(str(wd2idx[w]))
	return " ".join(result)

def word2id(wd2idx, in_path, out_path):
	files = os.listdir(in_path);
	for file in files:
		f = open(os.path.join(in_path, file))
		text = f.readline().strip();
		result = abstract2id(text, wd2idx)
		with open(os.path.join(out_path, file), 'wb') as writer:
			writer.write(result)

if __name__ == '__main__':
	# if len(sys.argv) != 2: # prints a message if you've entered flags incorrectly#
	# 	raise Exception("please enter two argv, one ref_dir, one dec_dir")

	vocab_dict = {}
	ref_dir = sys.argv[1]
	dec_dir = sys.argv[2]
	temp_ref_dir = "temp_ref"
	temp_dec_dir = "temp_dec"

	shutil.rmtree(temp_ref_dir)
	os.mkdir(temp_ref_dir)
	shutil.rmtree(temp_dec_dir)
	os.mkdir(temp_dec_dir)

	wd2idx = {}
	word2id(wd2idx, ref_dir, temp_ref_dir);
	word2id(wd2idx, dec_dir, temp_dec_dir);

	results_dict = decode.rouge_eval(temp_ref_dir, temp_dec_dir)  #一个ref_dir，一个dec_dir
	decode.rouge_log(results_dict, dec_dir)