import nncf
import argparse
import tensorflow as tf
import tensorflow_datasets as tdfs

from nncf.common.quantization.structs import QuantizationPreset

NUM_CLASSES = 10

def transform_fn(data_item):
    images, _ = data_item
    return images

def preprocess_for_eval(image, label):
    image = tf.keras.applications.resnet50.preprocess_input(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    label = tf.one_hot(label, NUM_CLASSES)

    return image, label

def load_calibration_data(dataset_name='cifar10', dataset_split='test'):
    dataset = tdfs.load(dataset_name, split=dataset_split,
                        shuffle_files=False, as_supervised=True)
    dataset = dataset.map(preprocess_for_eval).batch(128)

    return dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, dest='experiment_name', required=True,
                        help='Unique name to give to generated model')
    parser.add_argument('--dataset_name', type=str, dest='dataset_name', required=True,
                        help='Name of dataset to use for calibration')
    parser.add_argument('--dataset_split', type=str, dest='dataset_split', required=True,
                        help='Which split (train/test/val) of the dataset to use for calibration')
    parser.add_argument('--orig_model_path', type=str, dest='orig_model_path', required=True,
                        help='Path to original unquantized model')
    # I don't think NNCF has implemented this yet: https://github.com/openvinotoolkit/nncf/blob/0ea9b48bbca2670dc28a72a13dbdf2969e7b8581/nncf/tensorflow/quantization/quantize.py#L77
    #parser.add_argument('--model_type_transformer', dest='model_type_transformer', required=False,
    #                    action='store_true',
    #help='if passed, quantization will use nncf.ModelType.Transformer for model_type')
    parser.add_argument('--use_mixed_preset', dest='use_mixed_preset', required=False,
                        action='store_true', help='if passed, uses nncf.Preset.MIXES as preset')
    # NNCF haven't implemented this either: https://github.com/openvinotoolkit/nncf/blob/0ea9b48bbca2670dc28a72a13dbdf2969e7b8581/nncf/tensorflow/quantization/quantize.py#L79
    #parser.add_argument('--no_fast_bias_correction', dest='no_fast_bias_correction', required=False,
    #                    action='store_true', help='if passed, will not use fast bias correction')
    parser.add_argument('--subset_size', type=int, dest='subset_size', required=False, default=300,
                        help='num samples to use from calibration dataset to estimate params')
    
    return parser.parse_args()
    
    
def main():
    args = parse_args()
    calibration_dataset = load_calibration_data(args.dataset_name, args.dataset_split)
    
    model = tf.keras.models.load_model(args.orig_model_path)

    calibration_dataset = nncf.Dataset(calibration_dataset, transform_fn)
    
    preset = QuantizationPreset.MIXED if args.use_mixed_preset else QuantizationPreset.PERFORMANCE
    preset_str = 'mixed' if args.use_mixed_preset else 'performance'
    
    quantized_model = nncf.quantize(model, calibration_dataset, preset=preset,
                                    subset_size=args.subset_size)
    
    save_model_name = (args.experiment_name + '_' + args.dataset_name + '_' +
                       args.dataset_split + '_preset_' + preset_str +
                       '_subset_size_' + str(args.subset_size) + '.h5')
    
    tf.keras.models.save_model(quantized_model, save_model_name)
    
if __name__ == '__main__':
    main()
