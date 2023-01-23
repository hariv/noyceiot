import json
import argparse

from input_info import InputInfo
from sparsity import Sparsity
from batchnorm_adaptation import BatchnormAdaptation
from sparsity_params import SparsityParams


CATEGORIES_MAP = {"inputinfo": "input_info",
                  "sparsity": "sparsity",
                  "batchnormadaptation": "batchnorm_adaptation",
                  "sparsityparams": "sparsity_params"}

SPARSITY_ALGORITHMS = ['magnitude_sparsity', 'rb_sparsity', 'const_sparsity']
SPARSITY_LEVEL_SETTING_MODES = ['local', 'global']
SCHEDULES = ['polynomial', 'exponential', 'adaptive', 'multistep']
WEIGHT_IMPORTANCES = ['abs', 'normed_abs']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_json', type=str, dest='save_json', required=True,
                        help="Name of json file to save output.")
    
    #input_info
    parser.add_argument('--sample_size', type=str, dest='sample_size', required=True,
                        help=("Shape of the tensor expected as input to the model."))
    parser.add_argument('--input_type', type=str, dest='inputinfo_type', required=False,
                        help="Data type of the model input tensor.")
    parser.add_argument('--filler', type=str, dest='inputinfo_filler', required=False,
                        help=("Determines what the tensor will be filled with when"
                              " passed to the model during tracing and exporting."))
    parser.add_argument('--keyword', type=str, dest='inputinfo_keyword', required=False,
                        help=("Keyword to be used when passing the tensor to the"
                              " model's 'forward' method."))
    
    # compression_sparsity
    parser.add_argument('--algorithm', type=str, dest='algorithm', required=True,
                        choices=SPARSITY_ALGORITHMS,
                        help="compression algorithm name")
    
    # initializer batchnorm_adaptation
    parser.add_argument('--num_bn_adaptation_samples',type=int, dest='batchnormadaptation_num_bn_adaptation_samples',
                        required=False, help=("Number of samples from the training dataset to use"
                                              " for model inference during the BatchNorm statistics"
                                              " adaptation procedure for the compressed model."))
    # params
    parser.add_argument('--sparsity_level_setting_mode', type=str, dest='sparsityparams_sparsity_level_setting_mode',
                        required=False, choices=SPARSITY_LEVEL_SETTING_MODES, help="The mode of sparsity level setting.")
    parser.add_argument('--schedule', type=str, dest='sparsityparams_schedule', required=False,
                        choices=SCHEDULES, help="The type of scheduling to use for adjusting the targetsparsity level.")
    parser.add_argument('--sparsity_target', type=float, dest='sparsityparams_sparsity_target', required=False,
                        help="Target sparsity level for the model, to be reached at the end of the compression schedule.")
    parser.add_argument('--sparsity_target_epoch', type=int, dest='sparsityparams_sparsity_target_epoch', required=False,
                        help=("Index of the epoch upon which the sparsity level of the model is scheduled to become equal"
                              " to sparsity_target."))
    parser.add_argument('--sparsity_freeze_epoch', type=int, dest='sparsityparams_lr_sparsity_freeze_epoch', required=False,
                        help="Duration, in epochs, of the learning rate dropping process.")
    parser.add_argument('--update_per_optimizer_step', dest='sparsityparams_update_per_optimizer_step', required=False,
                        action='store_true', help=("Whether the function-based sparsity level schedulers should update"
                                                   " the sparsity level after each optimizer step instead of each epoch"
                                                   " step."))
    parser.add_argument('--steps_per_epoch', type=int, dest='sparsityparams_steps_per_epoch', required=False,
                        help="Number of optimizer steps in one epoch.")
    parser.add_argument('--multistep_steps', type=str, dest='sparsityparams_lstnum_multistep_steps', required=False,
                        help="A list of scheduler steps at which to transition to the next scheduled sparsity level")
    parser.add_argument('--multistep_sparsity_levels', type=str, dest="sparsityparams_lstfloat_multistep_sparsity_levels",
                        required=False, help="Multistep scheduler only")
    parser.add_argument('--patience', type=int, dest='sparsityparams_patience', required=False,
                        help="A conventional patience parameter for the scheduler, as for any other standard scheduler.")
    parser.add_argument('--power', type=float, dest='sparsityparams_power', required=False,
                        help="For polynomial scheduler - determines the corresponding power value.")
    parser.add_argument('--no_concave', dest='sparsityparams_notinp_concave', required=False, action='store_true',
                        help=("For polynomial scheduler - if false, then the target sparsity level will be approached in"
                        " concave manner, and in a convex manner otherwise"))
    parser.add_argument('--weight_importance', type=str, dest='sparsityparams_weight_importance', required=False,
                        choices=WEIGHT_IMPORTANCES, help=("Determines the way in which the weight values will be sorted"
                                                          " after being aggregated in order to determine the sparsity"
                                                          " threshold corresponding to a specific sparsity level."))

    parser.add_argument('--sparsity_init', type=float, dest='sparsity_sparsity_init', required=False,
                        help="Initial value of the sparsity level applied to the model")
    parser.add_argument('--ignored_scopes', type=str, dest='sparsity_lststr_ignored_scopes', required=False,
                        help="A list of model control flow graph node scopes to be ignored for this operation.")
    parser.add_argument('--target_scopes', type=str, dest='sparsity_lststr_target_scopes', required=False,
                        help="A list of model control flow graph node scopes to be considered for this operation.")
    parser.add_argument('--compression_lr_multiplier', type=int, dest='sparsity_compression_lr_multiplier', required=False,
                        help="Used to increase/decrease gradients for compression algorithms' parameters.")

    return parser.parse_args()

def generate_dict(obj):
    generic_dict = {}
    for arg_name in obj.arg_names:
        if arg_name in obj.__dict__:
            generic_dict[arg_name] = obj.__dict__[arg_name]
    return generic_dict

def main():
    args = parse_args()
    
    input_info = InputInfo(args.sample_size)
    sparsity = Sparsity(args.algorithm)
    save_json_name = args.save_json
    
    batchnorm_adaptation = BatchnormAdaptation()
    sparsity_params = SparsityParams()
    
    for arg in args.__dict__:
        # if this input was specified
        if args.__dict__[arg] is not None and args.__dict__[arg] is not False:
            arg_category = arg.split('_')[0]
            arg_feat = '_'.join(arg.split('_')[1:])
            lststr = False
            lstnum = False
            lstfloat = False
            notinp = False
            
            if len(arg.split('_')) > 2 and arg.split('_')[1] == 'lststr':
                arg_feat = '_'.join(arg.split('_')[2:])
                lststr = True

            if len(arg.split('_')) > 2 and arg.split('_')[1] == 'lstnum':
                arg_feat = '_'.join(arg.split('_')[2:])
                lstnum = True

            if len(arg.split('_')) > 2 and arg.split('_')[1] == 'lstfloat':
                arg_feat = '_'.join(arg.split('_')[2:])
                lstfloat = True

            if len(arg.split('_')) > 2 and arg.split('_')[1] == 'notinp':
                arg_feat = '_'.join(arg.split('_')[2:])
                notinp = True
                
            if arg_category in CATEGORIES_MAP.keys():
                locals()[CATEGORIES_MAP[arg_category]].__dict__[arg_feat] = args.__dict__[arg]
                if lststr:
                    locals()[CATEGORIES_MAP[arg_category]].__dict__[arg_feat] = args.__dict__[arg].split(',')

                if lstnum:
                    locals()[CATEGORIES_MAP[arg_category]].__dict__[arg_feat] = list(map(lambda x: int(x),
                                                                                         args.__dict__[arg].split(',')))
                if lstfloat:
                    locals()[CATEGORIES_MAP[arg_category]].__dict__[arg_feat] = list(map(lambda x: float(x),
                                                                                         args.__dict__[arg].split(',')))
                if notinp:
                    locals()[CATEGORIES_MAP[arg_category]].__dict__[arg_feat] = not args.__dict__[arg]
                locals()[CATEGORIES_MAP[arg_category]].present = True
                            
    res_dict = {}
    res_dict['input_info'] = generate_dict(input_info)
    res_dict['compression'] = generate_dict(sparsity)

    if batchnorm_adaptation.present:
        res_dict['compression']['initializer'] = {}
        res_dict['compression']['initializer']['batchnorm_adaptation'] = generate_dict(batchnorm_adaptation)

    if sparsity_params.present:
        res_dict['compression']['params'] = generate_dict(sparsity_params)

    with open(save_json_name, 'w') as f:
        json.dump(res_dict, f)

if __name__ == '__main__':
    main()
