import sys
import re
import os


def get_linear_names(fpath):
    node_names = []
    shapes = []
    with open(fpath, 'r') as f:
        lines = f.readlines()
        lines = list(filter(lambda x: 'Quantize Node name' in x, lines))
        lines = list(filter(lambda x: 'Dot' in x, lines))
        lines = list(filter(lambda x: 'IsLinear:1' in x, lines))
        for line in lines:
            line = re.split(' ', line)
            line = list(filter(lambda x:len(x)>0, line))
            print(line)
            name = line[3]
            # import pdb; pdb.set_trace()
            shape = (int(re.split('{', line[7])[1][:-1]), int(line[8][:-2]))
            # shape = re.split(',', shape)
            print(shape)
            shapes.append((int(shape[0]), int(shape[1])))
            node_names.append(name)
    return node_names, shapes

def get_conv_names(fpath):
    node_names = []
    shapes = []
    with open(fpath, 'r') as f:
        lines = f.readlines()
        lines = list(filter(lambda x: 'Quantize Node name' in x, lines))
        lines = list(filter(lambda x: 'Conv2d' in x or 'Convolution' in x, lines))
        for line in lines:
            ori_line = line
            line = re.split(' ', line)
            line = list(filter(lambda x : len(x)>0, line))
            print(line)
            name = line[3]
            if('Convolution') in line:
                M = int(line[6])
                K = int(line[7])
                N = int(line[8])
                shapes.append((M,K,N))
            else:
                shapes.append(None)
            node_names.append(name)

    return node_names, shapes


def generate_cfg_bert(fpath, outf):
    names, _ = get_linear_names(fpath)
    with open(outf, 'w') as f:
        for name in names:
            f.write('%s %d %d\n' % (name, 8, 0))


def generate_cfg_bert_block(fpath, outf):
    names, shapes = get_linear_names(fpath)
    with open(outf, 'w') as f:
        for name, shape in zip(names, shapes):
            f.write('%s %d %d %s %s %s\n' % (name, 8, 0, './bert_block_sparse_data/row_%d_%d.bin' %
                                             (shape[0], shape[1]), './bert_block_sparse_data/col_%d_%d.bin' % (shape[0], shape[1]),
                                              './bert_block_sparse_data/val_%d_%d.bin' % (shape[0], shape[1])))




def generate_cfg_mobilenet(fpath, outf):
    names, _ = get_conv_names(fpath)
    with open(outf, 'w') as f:
        for name in names:
            f.write('%s %d 0\n'%(name, 8))


def generate_cfg_mobilenet_block(fpath, outf):
    names, shapes = get_conv_names(fpath)
    with open(outf, 'w') as f:
        for name, shape in zip(names, shapes):
            if shape:
                prefix='./sparse_data/%d_%d_%d/' %(shape[0], shape[1], shape[2])
                if os.path.exists(prefix) and len(os.listdir(prefix))>2:
                    f.write('%s %d 0 %s %s %s\n'%(name, 8, prefix+'row.bin', prefix+'col.bin',prefix+'val.bin'))
                else:
                    f.write('%s %d 0\n'%(name, 8))    
            else:
                f.write('%s %d 0\n'%(name, 8))

if __name__ == '__main__':
    # generate_cfg_bert(sys.argv[1], sys.argv[2])
    # generate_cfg_mobilenet(sys.argv[1], sys.argv[2])
    generate_cfg_mobilenet_block(sys.argv[1], sys.argv[2])