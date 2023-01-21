import json
import argparse

INPUT_INFO_OPT_ARG_NAMES = ['input_type', 'filler', 'keyword']
COMPRESSION_QUANT_OPT_ARG_NAMES = ['algorithm', 'preset', 'quantize_inputs', 'quantize_outputs',
                                   'export_to_onnx_standard_ops', 'overflow_fix', 'ignored_scopes',
                                   'target_scopes', 'compression_lr_multiplier']

INPUT_INFO_MAP = {
    "input_type": "type",
    "filler": "filler",
    "keyword": "keyword"
}

def parse_args():
    parser = argparse.ArgumentParser(description='Compression json generator')
    # input_info
    parser.add_argument('--sample_size', type=str, dest='sample_size', required=True,
                       help=("Shape of the tensor expected as input to the model."))
    parser.add_argument('--input_type', type=str, dest='input_type', required=False,
                       help="Data type of the model input tensor.")
    parser.add_argument('--filler', type=str, dest='filler', required=False,
                       help=("Determines what the tensor will be filled with when"
                             " passed to the model during tracing and exporting."))
    parser.add_argument('--keyword', type=str, dest='keyword', required=False,
                       help=("Keyword to be used when passing the tensor to the"
                             " model's 'forward' method."))


    # compression_quantization
    parser.add_argument('--algorithm', type=str, dest='algorithm', required=True,
                       help="compression algorithm name")

    # initializer batchnorm_adaptation
    parser.add_argument('--num_bn_adaptation_samples',type=int, dest='num_bn_adaptation_samples',
                       required=False, help=("Number of samples from the training dataset to use"
                                             " for model inference during the BatchNorm statistics"
                                             " adaptation procedure for the compressed model."))
    # initializer range
    parser.add_argument('--num_init_samples', type=int, dest='num_init_samples', required=False,
                       help=("Number of samples from the training dataset to consume as sample model"
                             " inputs for purposes of setting initial minimum and maximum quantization"
                             " ranges."))
    parser.add_argument('--init_range_type', type=str, dest='init_range_type', required=False,
                       help=("Type of the initializer - determines which statistics gathered during"
                             " initialization will be used to initialize the quantization ranges."))
    # initializer range params
    parser.add_argument('--min_percentile', type=float, dest='min_percentile',
                       required=False, help=("For 'percentile' and 'mean_percentile' types - specify"
                                             " the percentile of input value histograms to be set as"
                                             " the initial value for the quantizer input minimum."))
    parser.add_argument('--max_percentile', type=float, dest='max_percentile',
                       required=False, help=("For 'percentile' and 'mean_percentile' types - specify"
                                             " the percentile of input value histograms to be set as"
                                             " the initial value for the quantizer input maximum."))
    # initializer precision
    parser.add_argument('--init_precision_type', type=str, dest='init_precision_type',
                       required=False, help=("Type of precision initialization."))
    parser.add_argument('--bits', type=str, dest='bits', required=False,
                       help=("A list of bitwidth to choose from when performing precision initialization."
                             " Overrides bits constraints specified in weight and activation sections."))
    parser.add_argument('--num_data_points', type=int, dest='num_data_points', required=False,
                       help="Number of data points to iteratively estimate Hessian trace.")
    parser.add_argument('--iter_number', type=int, dest='iter_number', required=False,
                       help=("Maximum number of iterations of Hutchinson algorithm to Estimate"
                             " Hessian trace."))
    parser.add_argument('--tolerance', type=float, dest='tolerance', required=False,
                       help=("Minimum relative tolerance for stopping the Hutchinson algorithm."
                             " It's calculated between mean average trace from the previous iteration and"
                             " the current one."))
    parser.add_argument('--compression_ratio', type=float, dest='compression_ratio', required=False,
                       help=("The desired ratio between bit complexity of a fully INT8 model and a"
                             " mixed-precision lower-bit one for HAWQ type. The target model size after"
                             " quantization, relative to total parameters size in FP32 for AUTOQ type."))
    parser.add_argument('--eval_subset_ratio', type=float, dest='eval_subset_ratio', required=False,
                       help=("The desired ratio of dataloader to be iterated during each search"
                             " iteration of AutoQ precision initialization."))
    parser.add_argument('--warmup_iter_number', type=int, dest='warmup_iter_number', required=False,
                       help=("The number of random policy at the beginning of of AutoQ precision"
                             " initialization to populate replay buffer with experiences."))
    parser.add_argument('--bitwidth_per_scope', type=str, dest='bitwidth_per_scope', required=False,
                       help="Manual settings for the quantizer bitwidths.")
    parser.add_argument('--traces_per_layer_path', type=str, dest='traces_per_layer_path', required=False,
                       help=("Path to serialized PyTorch Tensor with average Hessian traces per quantized"
                             "modules."))
    parser.add_argument('--dump_init_precision_data', dest='dump_init_precision_data', required=False,
                       action='store_true', help=("Whether to dump data related to Precision Initialization"
                                                  " algorithm."))
    parser.add_argument('--bitwidth_assignment_mode', type=str, dest='bitwidth_assignment_mode',
                       required=False, help=("The mode for assignment bitwidth to activation quantizers."))

    # weights
    parser.add_argument('--w_mode', type=str, dest='w_mode', required=False, help="Mode of quantization.")
    parser.add_argument('--w_bits', type=int, dest='w_bits', required=False, help="Bitwidth to quantize to")
    parser.add_argument('--w_signed', dest='w_signed', required=False, action='store_true',
                       help="Whether to use signed or unsigned input/output values for quantization.")
    parser.add_argument('--w_per_channel', dest='w_per_channel', required=False, action='store_true',
                       help="Whether to quantize inputs of this quantizer per each channel of input tensor")
    parser.add_argument('--w_ignored_scopes', type=str, dest='w_ignored_scopes', required=False,
                       help="A list of model control flow graph node scopes to be ignored for this operation.")
    parser.add_argument('--w_target_scopes', type=str, dest='w_target_scopes', required=False,
                       help="A list of model control flow graph node scopes to be considered for this operation.")
    parser.add_argument('--w_logarithm_scale', dest='w_logarithm_scale', required=False, action='store_true',
                       help="Whether to use log of scale as the optimization parameter instead of the scale itself.")

    # activations
    parser.add_argument('--a_mode', type=str, dest='a_mode', required=False, help="Mode of quantization.")
    parser.add_argument('--a_bits', type=int, dest='a_bits', required=False, help="Bitwidth to quantize to")
    parser.add_argument('--a_signed', dest='a_signed', required=False, action='store_true',
                       help="Whether to use signed or unsigned input/output values for quantization.")
    parser.add_argument('--a_per_channel', dest='a_per_channel', required=False, action='store_true',
                       help="Whether to quantize inputs of this quantizer per each channel of input tensor")
    parser.add_argument('--a_ignored_scopes', type=str, dest='a_ignored_scopes', required=False,
                       help="A list of model control flow graph node scopes to be ignored for this operation.")
    parser.add_argument('--a_target_scopes', type=str, dest='a_target_scopes', required=False,
                       help="A list of model control flow graph node scopes to be considered for this operation.")
    parser.add_argument('--a_logarithm_scale', dest='a_logarithm_scale', required=False, action='store_true',
                       help="Whether to use log of scale as the optimization parameter instead of the scale itself.")
    parser.add_argument('--a_unified_scale_ops', type=str, dest='a_unified_scale_ops', required=False,
                       help="Specifies operations in the model which will share the same quantizer module for activations.")

    # params
    parser.add_argument('--batch_multiplier', type=int, dest='batch_multiplier', required=False,
                       help="Gradients will be accumulated for this number of batches before doing a 'backward' call.")
    parser.add_argument('--activations_quant_start_epoch', type=int, dest='activations_quant_start_epoch', required=False,
                       help="A zero-based index of the epoch, upon reaching which the activations will start to be quantized.")
    parser.add_argument('--weights_quant_start_epoch', type=int, dest='weights_quant_start_epoch', required=False,
                       help="Epoch index upon which the weights will start to be quantized.")
    parser.add_argument('--lr_poly_drop_start_epoch', type=int, dest='lr_poly_drop_start_epoch', required=False,
                       help="Epoch index upon which the learning rate will start to be dropped.")
    parser.add_argument('--lr_poly_drop_duration_epochs', type=int, dest='lr_poly_drop_duration_epochs', required=False,
                       help="Duration, in epochs, of the learning rate dropping process.")
    parser.add_argument('--disable_wd_start_epoch', type=int, dest='disable_wd_start_epoch', required=False,
                       help="Epoch to disable weight decay in the optimizer.")
    parser.add_argument('--base_lr', type=float, dest='base_lr', required=False, help="Initial value of learning rate.")
    parser.add_argument('--base_wd', type=float, dest='base_wd', required=False, help="Initial value of weight decay.")

    
    parser.add_argument('--preset', type=str, dest='preset', required=False,
                       help=("The preset defines the quantization schema for weights and activations"))
    parser.add_argument('--no_quantize_inputs', dest='no_quantize_inputs', required=False, action='store_true',
                       help=("Whether the model inputs should be immediately quantized prior to any other model"
                       " operations."))
    parser.add_argument('--quantize_outputs', dest='quantize_outputs', required=False, action='store_true',
                       help="Whether the model outputs should be additionally quantized.")
    parser.add_argument('--export_to_onnx_standard_ops', dest='export_to_onnx', required=False, action='store_true',
                       help=("Determines how should the additional quantization operations be exported into the"
                             " ONNX format."))
    parser.add_argument('--overflow_fix', type=str, dest='overflow_fix', required=False,
                       help=("Controls whether to apply the overflow issue fix for the appropriate NNCF config"
                             " or not."))
    parser.add_argument('--ignored_scopes', type=str, dest='ignored_scopes', required=False,
                       help="A list of model control flow graph node scopes to be ignored for this operation.")
    parser.add_argument('--target_scopes', type=str, dest='target_scopes', required=False,
                       help="A list of model control flow graph node scopes to be considered for this operation.")
    parser.add_argument('--compression_lr_multiplier', type=int, dest='compression_lr_multiplier', required=False,
                       help="Used to increase/decrease gradients for compression algorithms' parameters.")

    return parser.parse_args()

def parse_input_info(args):
    input_info_dict = {}
    input_info_dict["sample_size"] = list(map(lambda x: int(x), args.sample_size.split(",")))

    for a in INPUT_INFO_OPT_ARG_NAMES:
        if args.__dict__[a]:
            input_info_dict[INPUT_INFO_MAP[a]] = args.__dict__[a]
    
    return input_info_dict

def parse_quant_comp_info(args):
    assert(args.algorithm == "quantization")
    quant_comp_dict = {}
    quant_comp_dict["algorithm"] = args.algorithm

    for a in COMPRESSION_QUANT_OPT_ARG_NAMES:
        if args.__dict__[a]:
            quant_comp_dict[COMPRESSION_QUANT_MAP[a]] = args.__dict__[a]
            
    return quant_comp_dict

def main():
    args = parse_args()

    config_dict = {}
    input_info_dict = parse_input_info(args)
    config_dict["input_info"] = input_info_dict
    
    quant_comp_dict = parse_quant_comp_info(args)
    config_dict["compression"] = quant_comp_dict

    print(json.dumps(config_dict, indent=2))
    #quant_comp_init_dict = parse_quant_comp_info(args)
    #quant_comp_preset_dict = parse_quant_preset_info(args)
    
    #sample_size = args.sample_size.split(",")
    #sample_size = list(map(lambda x: int(x), sample_size))
    #algorithm = args.algorithm

        
if __name__ == '__main__':
    main()
