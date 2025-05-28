from trainers.train import Trainer
from trainers.train_label_shift import Trainer_LS


import argparse
parser = argparse.ArgumentParser()

if __name__ == "__main__":

    # ========  Experiments Phase ================
    parser.add_argument('--phase',               default='train',         type=str, help='train')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name',               default='EXP1',         type=str, help='experiment name')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method',              default='DANN',               type=str, help='DANN, DIRT, DSAN, HoMM, CoDATS, SASA')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path',              default=r'./data',                  type=str, help='Path containing datase2t')
    parser.add_argument('--dataset',                default='HAR',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs',               default=1,                          type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device',                 default= "cuda",                   type=str, help='cpu or cuda')

    # ========= Additional Setting ================
    parser.add_argument('--label_shift',          default='False',                   type=str, help='False or True')    

    # arguments
    args = parser.parse_args()

    if args.label_shift == "True":
        args.exp_name = args.exp_name+'_label_shift'
        trainer = Trainer_LS(args)
    else:
        # create trainier object
        trainer = Trainer(args)
    
    # train and test
    if args.phase == 'train':
        trainer.fit()

