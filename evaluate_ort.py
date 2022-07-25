import argparse
import yaml
import os
import torch
import torch.utils.data as torch_data
from train_eval.initialization import initialize_prediction_model, initialize_metric,\
    initialize_dataset, get_specific_args
from train_eval.utils import convert_double_to_float
from torch_ort import ORTInferenceModule
from torch.onnx import utils as onnx_utils
import pickle


if __name__ == '__main__':


    # Initialize device:
    device = torch.device("cpu")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file with dataset parameters", required=True)
    parser.add_argument("-o", "--output_dir", help="Directory to save results", required=True)
    args = parser.parse_args()

    # Make directory
    os.makedirs('./onnx_model', exist_ok=True)

    onnx_file_path = os.path.join('onnx_model', 'testing.onnx')

    # Load config
    with open(args.config, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)


    # Initialize model
    model = initialize_prediction_model(cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
                                            cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'])
    model = model.float().to("cpu")
    model.eval()
    model = ORTInferenceModule(model)

    with open("pgp.pickle", "rb") as handle:
        data = pickle.load(handle)
    result = model(data['inputs'])
