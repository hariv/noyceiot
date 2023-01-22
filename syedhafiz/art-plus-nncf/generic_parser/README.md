## Sample invocation

This invocation includes all optional arguments except:
* compression->initializer->precision->bitwidth_per_scope
* compression-> scope_overrides

from [https://openvinotoolkit.github.io/nncf/](https://openvinotoolkit.github.io/nncf/)

Invocation does not include these 2 arguments since parsing them has not yet been implemented.

```
python3 generate_quantization_config.py --sample_size 1,3,224,224 --algorithm experimental_quantization --keyword batman --input_type float --preset performance --quantize_outputs --export_to_onnx_standard_ops --overflow_fix enable --compression_lr_multiplier 0 --ignored_scopes LeNet/relu_0,LeNet/relu_1 --target_scopes "UNet/ModuleList[down_path]/UNetConvBlock[1]/Sequential[block]/Conv2d[0],UNet/ModuleList[down_path]/UNetConvBlock[2]/Sequential[block]/Conv2d[0]" --num_bn_adaptation_samples 0 --range_type threesigma --num_init_samples 0 --a_mode symmetric --a_bits 0 --a_per_channel --a_ignored_scopes LeNet/relu_1 --a_target_scopes "UNet/ModuleList[down_path]/UNetConvBlock[1]/Sequential[block]/Conv2d[0]" --a_signed --a_logarithm_scale --a_unified_scale_ops a,b --w_mode asymmetric --w_bits 0 --w_signed --w_per_channel --w_ignored_scopes a,b,c,d,e --w_target_scopes a,b,c,d,e,f,g,h --w_logarithm_scale --batch_multiplier 0 --activations_quant_start_epoch 0 --weights_quant_start_epoch 0 --lr_poly_drop_start_epoch 0 --lr_poly_drop_duration_epochs 0 --disable_wd_start_epoch 0 --base_lr 0 --base_wd 0 --min_percentile 0 --max_percentile 0 --precision_type hawq --bits 2,4,5 --num_data_points 0 --iter_number 0  --tolerance 0 --compression_ratio 0 --eval_subset_ratio 0 --warmup_iter_number 0 --traces_per_layer_path hjksdgf --dump_init_precision_data --bitwidth_assignment_mode liberal --no_quantize_inputs --filler abc --target_quantizer_group weights --range_ignored_scopes a --range_target_scopes 1,2,3,4,5,6,7 --save_json config.json
```

Check `config.json` for resultant json