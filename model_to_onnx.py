import argparse
import os
import pickle

import torch
import torch.utils.data as torch_data
import yaml

from train_eval.initialization import (
    get_specific_args,
    initialize_dataset,
    initialize_metric,
    initialize_prediction_model,
)
from train_eval.utils import convert_double_to_float

if __name__ == "__main__":

    # Initialize device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file with dataset parameters", required=True)
    parser.add_argument("-r", "--data_root", help="Root directory with data", required=False)
    parser.add_argument("-d", "--data_dir", help="Directory to extract data", required=False)
    parser.add_argument("-o", "--output_dir", help="Directory to save results", required=True)
    parser.add_argument("-w", "--checkpoint", help="Path to pre-trained or intermediate checkpoint", required=True)
    parser.add_argument("-p", "--load_from_pickle_file", help="Use pickle file to load the dataset", required=False)

    args = parser.parse_args()

    data_root, data_dir, checkpoint_path = args.data_root, args.data_dir, args.checkpoint

    # Make directory
    os.makedirs("./onnx_model", exist_ok=True)

    onnx_file_path = os.path.join("onnx_model", "testing.onnx")

    # Load config
    with open(args.config, "r") as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    """
    Initialize evaluator object
    :param cfg: Configuration parameters
    :param data_root: Root directory with data
    :param data_dir: Directory with extracted, pre-processed data
    :param checkpoint_path: Path to checkpoint with trained weights
    """

    # Initialize dataset
    if not args.load_from_pickle_file:
        ds_type = cfg["dataset"] + "_" + cfg["agent_setting"] + "_" + cfg["input_representation"]
        spec_args = get_specific_args(cfg["dataset"], data_root, cfg["version"] if "version" in cfg.keys() else None)
        test_set = initialize_dataset(ds_type, ["load_data", data_dir, cfg["test_set_args"]] + spec_args)

        # Initialize dataloader
        dl = torch_data.DataLoader(test_set, cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    # Initialize model
    model = initialize_prediction_model(
        cfg["encoder_type"],
        cfg["aggregator_type"],
        cfg["decoder_type"],
        cfg["encoder_args"],
        cfg["aggregator_args"],
        cfg["decoder_args"],
    )
    model = model.float().to(device)
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Load data for onnx export
    if not args.load_from_pickle_file:
        data = next(iter(dl))
        data = convert_double_to_float(data)

    else:
        with open(args.load_from_pickle_file, "rb") as infile:
            data = pickle.load(infile)

    # Export model to onnx
    torch.onnx.export(
        model,
        (data, {}),
        onnx_file_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
    )
