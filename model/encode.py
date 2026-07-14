import argparse, os, sys, glob, math, time
import ants
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from project_utils import instantiate_from_config
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from taming.data.dsc import DSCInference
from taming.data.transforms import get_transforms_DSC


@torch.no_grad()
def run_conditional(model, dloader, device, batch_size=1):
    for i, batch in enumerate(tqdm(dloader)):
        dsc_signal = batch['dsc_signal'].squeeze().to(device)
        dsc_signal_maxv = batch['dsc_signal_maxv'].squeeze().to(device)
        tumor_mask = batch['tumor_mask'].squeeze().to(device)
        patient_dir = batch['patient_dir'][0]
        
        dsc_signal_flatten = dsc_signal[tumor_mask].unsqueeze(1)

        encoded_features = []
        results = []
        for j in range(0, len(dsc_signal_flatten), batch_size):
            input = dsc_signal_flatten[j:j+batch_size]
            h = model.encoder(input)
            h = model.quant_conv(h)
            encoded_features.append(h.cpu())
            _, _, info = model.quantize(h)
            results.append(info[2])

        # DVAE Encoding
        # h shape: [N, 64, 3] (for each input, perslice per point)
        encoded_features = torch.cat(encoded_features, dim=0).to(torch.float32).cpu()
        encoded_features = encoded_features.view(encoded_features.shape[0], -1)
        
        dvae_encoded = torch.zeros(dsc_signal.shape, dtype=torch.float32)[..., :6]
        dvae_encoded[tumor_mask] = encoded_features
        dvae_encoded = dvae_encoded.numpy()
        
        save_dir = os.path.join(patient_dir, 'dsc_dvae_encoded')
        os.makedirs(save_dir, exist_ok=True)
        
        dsc_path = os.path.join(patient_dir, 'dsc_ae_encoded', 'ae_encoded.nii.gz')
        if not os.path.exists(dsc_path):
            dsc_path = os.path.join(patient_dir, 'dsc', 'dsc.nii')
            dsc_template = ants.image_read(dsc_path)
            dsc_template_sub = list([ants.slice_image(dsc_template, axis=3, idx=i) for i in range(0, 6)])
            dsc_template_sub = ants.list_to_ndimage(dsc_template, dsc_template_sub)
        else:
            dsc_template_sub = ants.image_read(dsc_path)

        dvae_encoded_nifti = dsc_template_sub.new_image_like(dvae_encoded)
        ants.image_write(dvae_encoded_nifti, os.path.join(save_dir, f'dvae_encoded.nii.gz'))
        
        # Quantization
        results = torch.cat(results, dim=0).squeeze().to(torch.float32).cpu()
        decimal_values = results @ torch.tensor([4, 2, 1], dtype=results.dtype, device=results.device) + 1

        # 2. Peak Height 기준 순서 정의 (H1이 1등, H2가 2등 ...)
        # 이미지 분석 결과 기반 순위: H1, H2, H4, H3, H8, H6, H5, H7
        # 이 리스트의 인덱스 0은 decimal_value 1(H1)이 가질 새로운 '순위' 혹은 '값'을 의미합니다.
        rank_mapping = torch.tensor([1, 2, 4, 3, 7, 6, 8, 5]) 

        # 3. Remapping 적용
        # decimal_values가 1~8이므로, 인덱스로 쓰기 위해 1을 빼고 정수형으로 변환합니다.
        remapped_values = rank_mapping[(decimal_values.long() - 1)]

        num_unique = len(np.unique(decimal_values))
        if num_unique > 8:
            print(f"{patient_dir} Unique values: {num_unique}")

        quantized_dsc = torch.zeros(tumor_mask.shape, dtype=torch.float32, device=remapped_values.device)
        quantized_dsc[tumor_mask] = remapped_values.to(torch.float32)
        quantized_dsc = quantized_dsc.cpu().numpy()
        
        save_dir = os.path.join(patient_dir, 'dsc_clusters')
        os.makedirs(save_dir, exist_ok=True)
        
        tumor_mask_path = os.path.join(patient_dir, 'tumor_mask.nii.gz')
        tumor_mask_nifti = ants.image_read(tumor_mask_path)
        
        quantized_dsc_nifti = tumor_mask_nifti.new_image_like(quantized_dsc)
        ants.image_write(quantized_dsc_nifti, os.path.join(save_dir, f'dVAE_quantization.nii.gz'))


        # num_unique = len(np.unique(decimal_values))
        # if num_unique > 8:
        #     print(f"{patient_dir} Unique values: {num_unique}")

        # quantized_dsc = torch.zeros(tumor_mask.shape, dtype=torch.float32, device=decimal_values.device)
        # quantized_dsc[tumor_mask] = decimal_values.to(torch.float32)
        # quantized_dsc = quantized_dsc.cpu().numpy()
        
        # save_dir = os.path.join(patient_dir, 'dsc_clusters')
        # os.makedirs(save_dir, exist_ok=True)
        
        # tumor_mask_path = os.path.join(patient_dir, 'tumor_mask.nii.gz')
        # tumor_mask_nifti = ants.image_read(tumor_mask_path)
        
        # quantized_dsc_nifti = tumor_mask_nifti.new_image_like(quantized_dsc)
        # ants.image_write(quantized_dsc_nifti, os.path.join(save_dir, f'dVAE_quantization.nii.gz'))
        

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        help="Name of the checkpoint to load. If not specified, will load the last checkpoint.",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Number of GPUs to use.",
    )
    return parser


def load_model_from_config(config, sd, gpu=True, device=0, eval_mode=True):
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Missing Keys in State Dict: {missing}")
        print(f"Unexpected Keys in State Dict: {unexpected}")
    if gpu:
        model = model.to(f"cuda:{device}")
    if eval_mode:
        model.eval()
    return {"model": model}

def load_model(config, ckpt, gpu, device, eval_mode):
    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
    else:
        pl_sd = {"state_dict": None}
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   device=device,
                                   eval_mode=eval_mode)["model"]
    return model

if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            if opt.ckpt_name:
                ckpt = os.path.join(logdir, "checkpoints", opt.ckpt_name)
            else:
                ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    if opt.ignore_base_data:
        for config in configs:
            if hasattr(config, "data"): del config["data"]
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True
    show_config = False
    if show_config:
        print(OmegaConf.to_container(config))

    dataset = DSCInference(dataroot=opt.dataroot, transform=get_transforms_DSC())

    config["model"]["params"]["sane_index_shape"] = True
    model = load_model(config, ckpt, gpu, opt.device, eval_mode)
    dloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    run_conditional(model, dloader, opt.device, batch_size=32768)
