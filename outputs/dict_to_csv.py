import torch
import csv
import os
import argparse
parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default=None, type=str)

parser.add_argument('--arch', dest='arch',
                    help='Architecture of model',
                    default="cganet", type=str)

parser.add_argument('--norm', dest='norm',
                    help='The directory used to save the trained models',
                    default="evonorm", type=str)

parser.add_argument('--lr', 
                    help='The directory used to save the trained models',
                    default=0.01, type=float)

parser.add_argument('--seed', dest='seed',
                    help='The directory used to save the trained models',
                    default=1234, type=int)

parser.add_argument('--output-file', dest='output_file',
                    help='The directory used to save the trained models',
                    default='output.tsv', type=str)
parser.add_argument('-world_size', '--world_size', default=5, type=int, help='total number of nodes')
parser.add_argument('--graph', '-g',  default='ring', help = 'graph structure - [ring, torus]' )
parser.add_argument('--skew', default=1.0, type=float,     help='obelongs to [0,1] where 0= completely iid and 1=completely non-iid')

args = parser.parse_args()

if args.save_dir is None:
    if args.norm is not None:
        args.save_dir = args.arch+"_nodes_"+str(args.world_size)+"_"+ args.norm+"_lr_"+ str(args.lr)+"_seed_"+str(args.seed)+"_skew_"+str(args.skew)+"_"+args.graph
    
else:
    args.save_dir += args.arch+"_nodes_"+str(args.world_size)+"_"+ args.norm+"_lr_"+ str(args.lr)+"_seed_"+str(args.seed)+"_skew_"+str(args.skew)+"_"+args.graph
 
def average(input):
    return sum(input)/len(input)
dict_data = torch.load(os.path.join(args.save_dir, "excel_data","dict"))
fields = dict_data.keys()
dict_data["avg test acc"] = average(dict_data["avg test acc"])
dict_data["avg test acc final"] = average(dict_data["avg test acc final"])
dict_data["data transferred"] = average(dict_data["data transferred"])

if not( os.path.isfile(args.output_file) ):
    with open(args.output_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames= fields, delimiter='\t' )
        writer.writeheader()


with open(args.output_file, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames= fields, delimiter='\t' )
    writer.writerow(dict_data)