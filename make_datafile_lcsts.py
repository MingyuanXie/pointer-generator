# -*- coding: utf-8 -*-
import sys
import os
import hashlib
import struct
import subprocess
import collections
import jieba
from tensorflow.core.example import example_pb2

# 我们使用如下符号在.bin数据文件中对句子进行分割
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

finished_files_dir = "finished_files_test"
chunks_dir = os.path.join(finished_files_dir, "chunked")
lcsts_dir = "/home/xmy/LCSTS/DATA"
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data
VOCAB_SIZE = 50000



# 把指定的bin进行分块
def chunk_file(set_name):
    in_file = 'finished_files/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1

def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunks_dir)


# 根据url_file文件中列出的url，读取分词后的.story文件，并将它们写入输出文件中
def write_to_bin(in_file, out_file, label_offset_line, abstract_offset_line, article_offset_line, offset_line, makevocab=False, scoreFilter=False, highScore = 0):

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        i = 0;
        with open(in_file, "r") as f:
            for line in f:
                i = i + 1;
                label = 0
                if i % offset_line == label_offset_line and scoreFilter:
                    label = int(line.strip()[13:14])

                if not scoreFilter or label >= highScore:
                    if i % offset_line == abstract_offset_line:
                        seg_list = jieba.cut(line.strip());
                        abstract = SENTENCE_START + " " + " ".join(seg_list) + " " + SENTENCE_END
                    if i % offset_line == article_offset_line:
                        seg_list = jieba.cut(line.strip());
                        article = " ".join(seg_list)

                        if makevocab:
                            art_tokens = article.split()
                            abs_tokens = abstract.split()
                            abs_tokens = [t for t in abs_tokens if
                                          t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                            tokens = art_tokens + abs_tokens
                            tokens = [t.strip() for t in tokens]  # strip
                            tokens = [t for t in tokens if t != ""]  # remove empty
                            vocab_counter.update(tokens)

                        # Write to tf.Example
                        tf_example = example_pb2.Example()
                        tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
                        tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
                        tf_example_str = tf_example.SerializeToString()
                        str_len = len(tf_example_str)
                        writer.write(struct.pack('q', str_len))
                        writer.write(struct.pack('%ds' % str_len, tf_example_str))

    print("Finished writing file %s\n" % out_file)
    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':
    # write_to_bin(os.path.join(lcsts_dir, "PART_I.txt"), os.path.join(finished_files_dir, "train.bin"), 0, 3, 6, 8, scoreFilter=False, makevocab=True)
    # write_to_bin(os.path.join(lcsts_dir, "PART_II.txt"), os.path.join(finished_files_dir, "val.bin"), 2, 4, 7, 9, scoreFilter=False, makevocab=False)
    # write_to_bin(os.path.join(lcsts_dir, "PART_III.txt"), os.path.join(finished_files_dir, "test.bin"), 2, 4, 7, 9, scoreFilter=False, makevocab=False)

    write_to_bin(os.path.join(lcsts_dir, "PART_III.txt"), os.path.join(finished_files_dir, "test.bin"), 2, 4, 7, 9, scoreFilter=True, makevocab=False, highScore=3)

    # 这里吧test.bin train.bin val.bin三个进行拆分成子bin，每个包含1000个example
    chunk_all()