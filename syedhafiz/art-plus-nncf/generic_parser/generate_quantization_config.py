import json
import argparse

from input_info import InputInfo
from quantization import Quantization
from batchnorm_adaptation import BatchnormAdaptation
from quantization_range import QuantizationRange
from range_params import RangeParams
from precision import Precision
from quantization_params import QuantizationParams
from weights import Weights
from activations import Activations

CATEGORIES_MAP = {"inputinfo": "input_info",
                  "quantization": "quantization",
                  "batchnormadaptation": "batchnorm_adaptation",
                  "quantizationrange": "quantization_range",
                  "rangeparams": "range_params",
                  "precision": "precision",
                  "quantizationparams": "quantization_params",
                  "weights": "weights",
                  "activations": "activations"}

QUANTIZATION_ALGORITHMS = ['quantization', 'experimental_quantization']
RANGE_TYPES = ['mixed_min_max', 'min_max', 'mean_min_max', 'threesigma', 'percentile', 'mean_percentile']
TARGET_QUANTIZER_GROUPS = ['activations', 'weights']
PRESETS = ['performance', 'mixed']
MODES = ['symmetric', 'asymmetric']
OVERFLOW_FIXES = ['enable', 'disable', 'first_layer_only']
PRECISION_TYPES = ['hawq', 'autoq', 'manual']
BITWIDTH_ASSIGNMENT_MODES = ['strict', 'liberal']

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
    
    # compression_quantization
    parser.add_argument('--algorithm', type=str, dest='algorithm', required=True,
                        choices=QUANTIZATION_ALGORITHMS,
                        help="compression algorithm name")
    
    # initializer batchnorm_adaptation
    parser.add_argument('--num_bn_adaptation_samples',type=int, dest='batchnormadaptation_num_bn_adaptation_samples',
                        required=False, help=("Number of samples from the training dataset to use"
                                              " for model inference during the BatchNorm statistics"
                                              " adaptation procedure for the compressed model."))
    # initializer range
    parser.add_argument('--num_init_samples', type=int, dest='quantizationrange_num_init_samples', required=False,
                        help=("Number of samples from the training dataset to consume as sample model"
                              " inputs for purposes of setting initial minimum and maximum quantization"
                              " ranges."))
    parser.add_argument('--range_type', type=str, dest='quantizationrange_type', required=False,
                        choices=RANGE_TYPES, help=("Type of the initializer - determines which statistics"
                                                   " gathered during initialization will be used to"
                                                   " initialize the quantization ranges."))
    
    parser.add_argument('--range_ignored_scopes', type=str, dest='quantizationrange_lststr_ignored_scopes', required=False,
                        help="A list of model control flow graph node scopes to be ignored for this operation.")
    parser.add_argument('--range_target_scopes', type=str, dest='quantizationrange_lststr_target_scopes', required=False,
                        help="A list of model control flow graph node scopes to be considered for this operation.")
    parser.add_argument('--target_quantizer_group', type=str, dest='quantizationrange_target_quantizer_group',
                        required=False, choices=TARGET_QUANTIZER_GROUPS,
                        help="The target group of quantizers for which the specified type of range initialization will be applied.")
    
    
    # initializer range params
    parser.add_argument('--min_percentile', type=float, dest='rangeparams_min_percentile',
                        required=False, help=("For 'percentile' and 'mean_percentile' types - specify"
                                              " the percentile of input value histograms to be set as"
                                              " the initial value for the quantizer input minimum."))
    parser.add_argument('--max_percentile', type=float, dest='rangeparams_max_percentile',
                        required=False, help=("For 'percentile' and 'mean_percentile' types - specify"
                                              " the percentile of input value histograms to be set as"
                                              " the initial value for the quantizer input maximum."))
    
    # initializer precision
    parser.add_argument('--precision_type', type=str, dest='precision_type', choices=PRECISION_TYPES,
                        required=False, help=("Type of precision initialization."))
    parser.add_argument('--bits', type=str, dest='precision_lstnum_bits', required=False,
                        help=("A list of bitwidth to choose from when performing precision initialization."
                              " Overrides bits constraints specified in weight and activation sections."))
    parser.add_argument('--num_data_points', type=int, dest='precision_num_data_points', required=False,
                        help="Number of data points to iteratively estimate Hessian trace.")
    parser.add_argument('--iter_number', type=int, dest='precision_iter_number', required=False,
                        help=("Maximum number of iterations of Hutchinson algorithm to Estimate"
                              " Hessian trace."))
    parser.add_argument('--tolerance', type=float, dest='precision_tolerance', required=False,
                        help=("Minimum relative tolerance for stopping the Hutchinson algorithm."
                              " It's calculated between mean average trace from the previous iteration and"
                              " the current one."))
    parser.add_argument('--compression_ratio', type=float, dest='precision_compression_ratio', required=False,
                        help=("The desired ratio between bit complexity of a fully INT8 model and a"
                              " mixed-precision lower-bit one for HAWQ type. The target model size after"
                              " quantization, relative to total parameters size in FP32 for AUTOQ type."))
    parser.add_argument('--eval_subset_ratio', type=float, dest='precision_eval_subset_ratio', required=False,
                        help=("The desired ratio of dataloader to be iterated during each search"
                              " iteration of AutoQ precision initialization."))
    parser.add_argument('--warmup_iter_number', type=int, dest='precision_warmup_iter_number', required=False,
                        help=("The number of random policy at the beginning of of AutoQ precision"
                              " initialization to populate replay buffer with experiences."))
    parser.add_argument('--bitwidth_per_scope', type=str, dest='precision_bitwidth_per_scope', required=False,
                        help="Manual settings for the quantizer bitwidths.")
    parser.add_argument('--traces_per_layer_path', type=str, dest='precision_traces_per_layer_path', required=False,
                        help=("Path to serialized PyTorch Tensor with average Hessian traces per quantized"
                              "modules."))
    parser.add_argument('--dump_init_precision_data', dest='precision_dump_init_precision_data', required=False,
                        action='store_true', help=("Whether to dump data related to Precision Initialization"
                                                   " algorithm."))
    parser.add_argument('--bitwidth_assignment_mode', type=str, dest='precision_bitwidth_assignment_mode',
                        choices=BITWIDTH_ASSIGNMENT_MODES, required=False,
                        help=("The mode for assignment bitwidth to activation quantizers."))
    
    # weights
    parser.add_argument('--w_mode', type=str, dest='weights_mode', choices=MODES, required=False,
                        help="Mode of quantization.")
    parser.add_argument('--w_bits', type=int, dest='weights_bits', required=False, help="Bitwidth to quantize to")
    parser.add_argument('--w_signed', dest='weights_signed', required=False, action='store_true',
                        help="Whether to use signed or unsigned input/output values for quantization.")
    parser.add_argument('--w_per_channel', dest='weights_per_channel', required=False, action='store_true',
                        help="Whether to quantize inputs of this quantizer per each channel of input tensor")
    parser.add_argument('--w_ignored_scopes', type=str, dest='weights_lststr_ignored_scopes', required=False,
                        help="A list of model control flow graph node scopes to be ignored for this operation.")
    parser.add_argument('--w_target_scopes', type=str, dest='weights_lststr_target_scopes', required=False,
                        help="A list of model control flow graph node scopes to be considered for this operation.")
    parser.add_argument('--w_logarithm_scale', dest='weights_logarithm_scale', required=False, action='store_true',
                        help="Whether to use log of scale as the optimization parameter instead of the scale itself.")
    
    # activations
    parser.add_argument('--a_mode', type=str, dest='activations_mode', choices=MODES, required=False,
                        help="Mode of quantization.")
    parser.add_argument('--a_bits', type=int, dest='activations_bits', required=False, help="Bitwidth to quantize to")
    parser.add_argument('--a_signed', dest='activations_signed', required=False, action='store_true',
                        help="Whether to use signed or unsigned input/output values for quantization.")
    parser.add_argument('--a_per_channel', dest='activations_per_channel', required=False, action='store_true',
                        help="Whether to quantize inputs of this quantizer per each channel of input tensor")
    parser.add_argument('--a_ignored_scopes', type=str, dest='activations_lststr_ignored_scopes', required=False,
                        help="A list of model control flow graph node scopes to be ignored for this operation.")
    parser.add_argument('--a_target_scopes', type=str, dest='activations_lststr_target_scopes', required=False,
                        help="A list of model control flow graph node scopes to be considered for this operation.")
    parser.add_argument('--a_logarithm_scale', dest='activations_logarithm_scale', required=False, action='store_true',
                        help="Whether to use log of scale as the optimization parameter instead of the scale itself.")
    parser.add_argument('--a_unified_scale_ops', type=str, dest='activations_lststr_unified_scale_ops', required=False,
                        help="Specifies operations in the model which will share the same quantizer module for activations.")
    
    # params
    parser.add_argument('--batch_multiplier', type=int, dest='quantizationparams_batch_multiplier', required=False,
                        help="Gradients will be accumulated for this number of batches before doing a 'backward' call.")
    parser.add_argument('--activations_quant_start_epoch', type=int, dest='quantizationparams_activations_quant_start_epoch', required=False,
                        help="A zero-based index of the epoch, upon reaching which the activations will start to be quantized.")
    parser.add_argument('--weights_quant_start_epoch', type=int, dest='quantizationparams_weights_quant_start_epoch', required=False,
                        help="Epoch index upon which the weights will start to be quantized.")
    parser.add_argument('--lr_poly_drop_start_epoch', type=int, dest='quantizationparams_lr_poly_drop_start_epoch', required=False,
                        help="Epoch index upon which the learning rate will start to be dropped.")
    parser.add_argument('--lr_poly_drop_duration_epochs', type=int, dest='quantizationparams_lr_poly_drop_duration_epochs', required=False,
                        help="Duration, in epochs, of the learning rate dropping process.")
    parser.add_argument('--disable_wd_start_epoch', type=int, dest='quantizationparams_disable_wd_start_epoch', required=False,
                        help="Epoch to disable weight decay in the optimizer.")
    parser.add_argument('--base_lr', type=float, dest='quantizationparams_base_lr', required=False, help="Initial value of learning rate.")
    parser.add_argument('--base_wd', type=float, dest='quantizationparams_base_wd', required=False, help="Initial value of weight decay.")
    
    
    parser.add_argument('--preset', type=str, dest='quantization_preset', required=False, choices=PRESETS,
                        help=("The preset defines the quantization schema for weights and activations"))
    parser.add_argument('--no_quantize_inputs', dest='quantization_notinp_quantize_inputs', required=False, action='store_true',
                        help=("Whether the model inputs should be immediately quantized prior to any other model"
                             " operations."))
    parser.add_argument('--quantize_outputs', dest='quantization_quantize_outputs', required=False, action='store_true',
                        help="Whether the model outputs should be additionally quantized.")
    parser.add_argument('--export_to_onnx_standard_ops', dest='quantization_export_to_onnx', required=False, action='store_true',
                        help=("Determines how should the additional quantization operations be exported into the"
                             " ONNX format."))
    parser.add_argument('--overflow_fix', type=str, dest='quantization_overflow_fix', required=False,
                        choices=OVERFLOW_FIXES,
                        help=("Controls whether to apply the overflow issue fix for the appropriate NNCF config"
                             " or not."))
    parser.add_argument('--ignored_scopes', type=str, dest='quantization_lststr_ignored_scopes', required=False,
                        help="A list of model control flow graph node scopes to be ignored for this operation.")
    parser.add_argument('--target_scopes', type=str, dest='quantization_lststr_target_scopes', required=False,
                        help="A list of model control flow graph node scopes to be considered for this operation.")
    parser.add_argument('--compression_lr_multiplier', type=int, dest='quantization_compression_lr_multiplier', required=False,
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
    quantization = Quantization(args.algorithm)
    save_json_name = args.save_json
    
    batchnorm_adaptation = BatchnormAdaptation()
    quantization_range = QuantizationRange()
    range_params = RangeParams()
    precision = Precision()
    quantization_params = QuantizationParams()
    weights = Weights()
    activations = Activations()
    
    for arg in args.__dict__:
        # if this input was specified
        if args.__dict__[arg] is not None and args.__dict__[arg] is not False:
            arg_category = arg.split('_')[0]
            arg_feat = '_'.join(arg.split('_')[1:])
            lststr = False
            lstnum = False
            notinp = False
            
            if len(arg.split('_')) > 2 and arg.split('_')[1] == 'lststr':
                arg_feat = '_'.join(arg.split('_')[2:])
                lststr = True

            if len(arg.split('_')) > 2 and arg.split('_')[1] == 'lstnum':
                arg_feat = '_'.join(arg.split('_')[2:])
                lstnum = True

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

                if notinp:
                    locals()[CATEGORIES_MAP[arg_category]].__dict__[arg_feat] = not args.__dict__[arg]
                locals()[CATEGORIES_MAP[arg_category]].present = True
                            

    res_dict = {}
    res_dict['input_info'] = generate_dict(input_info)
    res_dict['compression'] = generate_dict(quantization)

    if batchnorm_adaptation.present:
        if 'initializer' not in res_dict['compression']:
            res_dict['compression']['initializer'] = {}
            
        res_dict['compression']['initializer']['batchnorm_adaptation'] = generate_dict(batchnorm_adaptation)

    if quantization_range.present:
        if 'initializer' not in res_dict['compression']:
            res_dict['compression']['initializer'] = {}

        res_dict['compression']['initializer']['range'] = generate_dict(quantization_range)

        if range_params.present:
            res_dict['compression']['initializer']['range']['params'] = generate_dict(range_params)
    
    if precision.present:
        if 'initializer' not in res_dict['compression']:
            res_dict['compression']['initializer'] = {}
            
        res_dict['compression']['initializer']['precision'] = generate_dict(precision)

    if quantization_params.present:
        res_dict['compression']['params'] = generate_dict(quantization_params)

    if weights.present:
        res_dict['compression']['weights'] = generate_dict(weights)

    if activations.present:
        res_dict['compression']['activations'] = generate_dict(activations)


    with open(save_json_name, 'w') as f:
        json.dump(res_dict, f)

if __name__ == '__main__':
    main()
