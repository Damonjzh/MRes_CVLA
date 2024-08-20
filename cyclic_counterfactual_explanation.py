import os
import argparse
import pickle
import cv2
import pandas as pd
import torch

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pipeline_stable_diffusion_pie import StableDiffusionPIEPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from pathlib import Path
import matplotlib.pyplot as plt
from training.mimic import Mimic
# from lungs_segmentation.pre_trained_models import create_model
# seg_model = create_model("resnet34")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# seg_model = seg_model.to(device)
# import lungs_segmentation.inference as inference
from textwrap import fill
import cv2
from report_label import get_chexbert_labels_for_gen_reports_from_str, get_chexbert

# import matplotlib
# matplotlib.use('Agg')

path = './mimic_manipulate_r2gencmn_16k_testset_rerun/'
path1 = "./gt_finding"
path2 = "./r2gencmn"

finetuned_path = './r2gencmn_val2000/checkpoint-16000'
guidance_scale_change = 7.5
csv_name = 'mimic_manipulate_r2gencmn_16k_testset.csv'
model_type = "r2gen_cmn"
if model_type == "r2gen_cmn":
    # r2gen_cmn
    from models.models import BaseCMNModel
    from modules.dataloaders import R2DataLoader
    from modules.loss import compute_loss
    # from modules.metrics import compute_scores
    from modules.tokenizers import Tokenizer
    from modules.tester import Tester

    load_path = "./R2GenCMN-main/results/mimic_example/model_mimic_cxr.pth"
elif model_type == "r2gen":
    # r2gen
    from module_r2gen.tokenizers import Tokenizer
    from module_r2gen.dataloaders import R2DataLoader
    # from module_r2gen.metrics import compute_scores
    from module_r2gen.optimizers import build_optimizer, build_lr_scheduler
    from module_r2gen.trainer import Trainer
    from modules.loss import compute_loss
    from models.r2gen import R2GenModel
    from modules.tester import Tester

    load_path = "./R2Gen-main/checkpoints/model_mimic_cxr.pth"

elif model_type == "rgrg":
    # rgrg
    from models.models import BaseCMNModel
    from modules.dataloaders import R2DataLoader
    from modules.loss import compute_loss
    from modules.metrics import compute_scores
    from modules.tokenizers import Tokenizer
    from modules.tester import Tester

    load_path = "./R2GenCMN-main/results/mimic_example/model_mimic_cxr.pth"


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a PIE inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--finetuned_path",
        type=str,
        default=None,
        required=False,
        help="Path to domain specific finetuned unet from any healthcare text-to-image dataset",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        required=True,
        help="Path to the input instance images.",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,  # "--mask_path="./assets/example_inputs/mask.png"
        required=False,
        help="Path to mask.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument("--step", type=int, default=10,
                        help="N in the paper, Number to images / steps for PIE generation")
    parser.add_argument("--strength", type=float, default=0.5, help="Roll back ratio garmma")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./simulation",
        help="The output directory.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="The lung with",
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    # for Relational Memory r2gen
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # r2gen_cmn
    parser.add_argument('--image_dir', type=str, default='/media/NAS06/zihao/files/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str,
                        default='/media/NAS04/zihao/R2GenCMN-main/data/mimic_cxr/mimic_annotation.json',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=1000, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=7e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    # parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--load', type=str,
                        default='/media/NAS04/zihao/R2GenCMN-main/results/mimic_example/model_mimic_cxr.pth',
                        help='whether to load the pre-trained model.')

    args = parser.parse_args()
    return args


# -------------------------------------------------------------------------------------------------------------#

def dicom_id_to_report_path(db, report_path, dicom_id: str):
    db_series = db.loc[dicom_id]
    subject_id = "p" + db_series["subject_id"]
    study_id = "s" + db_series["study_id"] + ".txt"
    subject_id_prefix = subject_id[:3]

    return report_path / Path("files") / Path(subject_id_prefix) / Path(subject_id) / Path(study_id)


def dicom_id_to_img_path(db, report_path, dicom_id: str):
    db_series = db.loc[dicom_id]
    subject_id = "p" + db_series["subject_id"]
    study_id = "s" + db_series["study_id"]
    study_id_jpg = dicom_id + ".jpg"
    subject_id_prefix = subject_id[:3]

    return report_path / Path("files") / Path(subject_id_prefix) / Path(subject_id) / Path(study_id) / Path(
        study_id_jpg)


def load_report(db, report_path, dicom_id: str, parse_fun):
    report_path = dicom_id_to_report_path(db, report_path, dicom_id)
    with open(report_path, "r") as f:
        txt = f.readlines()

    return parse_fun(txt)


def parse_report_i(txt: str) -> str:
    txt = " ".join([line.strip() for line in txt if line.strip() != ""])

    try:
        _, impression = txt.strip().split("IMPRESSION:")
    except:
        raise ValueError

    return impression.strip()


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            instance_dataset,
            instance_prompt,
            tokenizer,
            class_data_root=None,
            class_prompt=None,
            class_num=None,
            size=512,
            center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_dataset = instance_dataset
        self.instance_prompt = instance_prompt

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.instance_dataset.transform = self.image_transforms

    def __len__(self):
        return self.instance_dataset.__len__()

    def __getitem__(self, index):
        example = {}
        instance_image, label, label2, meta_data = self.instance_dataset.__getitem__(index)
        example['img_path'] = meta_data['img_path']
        example["instance_images"] = instance_image
        example["instance_prompt"] = self.instance_prompt + " " + meta_data["pathologies"]
        example["instance_prompt_r2gen"] = self.instance_prompt + " " + meta_data["pathologies2"]
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt + " " + meta_data["pathologies"],
            # self.instance_prompt + "," + meta_data["pathologies"]
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["pathologies"] = label
        example["pathologies_r2gen"] = label2
        example["gt_report"] = meta_data["gt_report"]
        example["generated_report"] = meta_data["generated_report"]

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "instance_prompt_ids": [example["instance_prompt"] for example in examples],
        "instance_prompt_r2gen": [example["instance_prompt_r2gen"] for example in examples],
        "img_path": [example["img_path"] for example in examples],
        "pathologies_gt": [example["pathologies"] for example in examples],
        "pathologies_r2gen": [example["pathologies_r2gen"] for example in examples],
        "gt_report": [example["gt_report"] for example in examples],
        "generated_report": [example["generated_report"] for example in examples],

    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def main(args):
    seed = args.seed
    set_all_seeds(seed)
    args.load = load_path

    image_path = args.image_path
    mask_path = args.mask_path
    prompt = args.prompt

    model_id_or_path = args.pretrained_model_name_or_path
    # finetuned_path = args.finetuned_path
    resolution = args.resolution
    ddim_times = args.step
    strength = args.strength
    guidance_scale = guidance_scale_change
    # guidance_scale = args.guidance_scale

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    pipe_original = StableDiffusionPIEPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32,
                                                               cache_dir="./checkpoints", safety_checker=None)
    pipe_original = pipe_original.to(device)

    pipe2 = StableDiffusionPIEPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32,
                                                       cache_dir="./checkpoints", safety_checker=None)

    unet2 = UNet2DConditionModel.from_pretrained(
        finetuned_path, subfolder="unet"
    )
    pipe2.unet = unet2
    pipe2 = pipe2.to(device)

    image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution)
        ]
    )

    images = []
    step_i = 0
    init_image_original = Image.open(image_path).convert("RGB")  # The unedited image
    init_image = image_transforms(init_image_original)
    init_image.save(os.path.join(output_dir, str(step_i) + ".png"))

    if mask_path != None:
        mask = Image.open(mask_path).convert("RGB")
        mask = image_transforms(mask)
        mask.save(os.path.join(output_dir, "mask" + ".png"))
    else:
        mask = None

    step_i += 1
    img = init_image
    images.append(img)

    instance_dataset = Mimic(path=path1, path2=path2, type="both", split='test')
    # instance_dataset = Mimic(path= path2, split='test', type='r2gen')
    train_dataset = DreamBoothDataset(
        instance_dataset=instance_dataset,
        instance_prompt=args.instance_prompt,
        class_data_root=None,
        class_prompt=None,
        class_num=100,
        tokenizer=tokenizer,
        size=512,
        center_crop=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, None),
        num_workers=0,
    )

    ddim_times = 1

    # r2gen model
    tokenizer = Tokenizer(args)
    criterion = compute_loss
    # metrics = compute_scores
    # model = BaseCMNModel(args, tokenizer)
    if model_type == "r2gen_cmn":
        model = BaseCMNModel(args, tokenizer)
    elif model_type == "r2gen":
        model = R2GenModel(args, tokenizer)

    # chexbert
    chexbert = get_chexbert()
    columns = [
        'id', 'img_path', 'image_id', 'Diffusion_model', 'Manipulate', 'Target', 'Result', 'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
        'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding', 'Report'

    ]

    df = pd.DataFrame(columns=columns)
    for step, batch in enumerate(train_dataloader):

        if step != 89:
            continue

        img_path = batch['img_path'][0]
        image_id = img_path.split('/')[-1].split('.')[0]

        init_image_original = Image.open(img_path).convert("RGB")
        init_image = image_transforms(init_image_original)

        if not os.path.exists(path):
            os.makedirs(path)
        save_path = path + str(step) + '_'.join(batch['img_path'][0].split('/')[6:9]).split('.jpg')[0]
        csv_path = path + csv_name

        img = init_image
        step_i = 1
        model_number = 1
        generator = torch.Generator("cuda").manual_seed(seed)
        # fig, ax = plt.subplots(3, 3 * model_number, figsize=(20, 10))

        for model_id in range(model_number):

            pipe = pipe2
            step_i = 1

            with torch.no_grad():
                while step_i <= ddim_times:
                    img_re = pipe(prompt=batch['instance_prompt_ids'], image=img, mask=None, init_image=init_image,
                                  strength=strength, guidance_scale=guidance_scale).images[0]
                    images.append(img_re)
                    img_gen = pipe(prompt=batch['instance_prompt_r2gen'], image=img, mask=None, init_image=init_image,
                                   strength=strength, guidance_scale=guidance_scale).images[0]
                    images.append(img_gen)
                    gt_title = batch['instance_prompt_ids'][0].split('The lung with ')[1]
                    r2gen_title = batch['instance_prompt_r2gen'][0].split('The lung with ')[1]
                    base_sentence = 'The lung with '
                    # img.save(os.path.join(save_path))
                    step_i += 1
                    gt_title_wrapped = fill(gt_title, width=15)
                    r2gen_title_wrapped = fill(r2gen_title, width=15)

                    img_path = batch['img_path'][0]
                    image_id = img_path.split('/')[-1].split('.')[0]
                    pathologies = batch['pathologies_gt'][0]
                    pathologies = [str(int(p)) for p in pathologies]
                    if model_id == 0:
                        model_name = 'gt'
                    elif model_id == 1:
                        model_name = 'r2gen_cmn'
                    data = [str(step), img_path, image_id, model_name, 'Reference_gt', '', ''] + pathologies + [''] + \
                           batch['gt_report']
                    gt_df = pd.DataFrame([data], columns=columns)
                    df = pd.concat([df, gt_df], ignore_index=True)

                    # r2gen
                    pil_image = Image.fromarray((img_re * 255).astype(np.uint8))

                    r2gen_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])

                    img_gen_r2 = r2gen_transform(pil_image)
                    img_gen_r2 = img_gen_r2.unsqueeze(0)
                    tester = Tester(model, criterion, args, img_gen_r2)
                    gen_report = tester.test()

                    # get_chexbert_labels_for_gen_reports(chexbert, input_file, res_file, output_file)
                    return_dict = get_chexbert_labels_for_gen_reports_from_str(chexbert, gen_report)
                    return_label_str = ', '.join([condition for condition, labels in return_dict.items() if
                                                  condition != 'No Finding' and labels[0] == 1])

                    manipulate_data = pd.DataFrame(return_dict)
                    df = pd.concat([df, manipulate_data], ignore_index=True)
                    df.at[df.index[-1], 'id'] = str(step)
                    df.at[df.index[-1], 'img_path'] = img_path
                    df.at[df.index[-1], 'image_id'] = image_id
                    df.at[df.index[-1], 'Diffusion_model'] = model_name
                    df.at[df.index[-1], 'Manipulate'] = 'Reconstruct_gt'
                    df.at[df.index[-1], 'Target'] = ''
                    df.at[df.index[-1], 'Result'] = ''
                    df.at[df.index[-1], 'Report'] = gen_report[0]

                    img_path = batch['img_path'][0]
                    image_id = img_path.split('/')[-1].split('.')[0]
                    pathologies = batch['pathologies_r2gen'][0]
                    pathologies = [str(int(p)) for p in pathologies]
                    data = [str(step), img_path, image_id, model_name, 'Reference_r2gen_cmn', '', ''] + pathologies + [
                        ''] + batch['generated_report']
                    r2gen_df = pd.DataFrame([data], columns=columns)
                    df = pd.concat([df, r2gen_df], ignore_index=True)

                    # r2gen
                    pil_image = Image.fromarray((img_gen * 255).astype(np.uint8))

                    r2gen_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])

                    img_gen_r2 = r2gen_transform(pil_image)
                    img_gen_r2 = img_gen_r2.unsqueeze(0)
                    tester = Tester(model, criterion, args, img_gen_r2)
                    gen_report = tester.test()

                    # get_chexbert_labels_for_gen_reports(chexbert, input_file, res_file, output_file)
                    return_dict = get_chexbert_labels_for_gen_reports_from_str(chexbert, gen_report)
                    # return_label_str_r2gen = ', '.join([labels for condition, labels in return_dict.items() if
                    #                               condition != 'No Finding' and labels[0] == 1])
                    values = [str(v[0]) for v in return_dict.values()]
                    return_label_str_r2gen = ','.join(values)

                    manipulate_data = pd.DataFrame(return_dict)
                    df = pd.concat([df, manipulate_data], ignore_index=True)
                    df.at[df.index[-1], 'id'] = str(step)
                    df.at[df.index[-1], 'img_path'] = img_path
                    df.at[df.index[-1], 'image_id'] = image_id
                    df.at[df.index[-1], 'Diffusion_model'] = model_name
                    df.at[df.index[-1], 'Manipulate'] = 'Reconstruct_r2gen_cmn'
                    df.at[df.index[-1], 'Target'] = ''
                    df.at[df.index[-1], 'Result'] = ''
                    df.at[df.index[-1], 'Report'] = gen_report[0]

                    add_result = []
                    remove_result = []

                    # remove keywords
                    remove_keywords = batch['instance_prompt_r2gen'][0].split("with")[1].strip().split(",")
                    img_remove_dict = {}
                    for keyword in remove_keywords:
                        if keyword == '':
                            remove_num_keywords -= 1
                            continue
                        remaining_keywords = [k for k in remove_keywords if k != keyword]
                        new_sentence = base_sentence + ','.join(remaining_keywords)
                        # generator.manual_seed(seed)
                        img_remove = pipe(prompt=new_sentence, image=img, mask=None, init_image=init_image,
                                          strength=strength, guidance_scale=guidance_scale).images[0]
                        img_remove_dict[keyword] = img_remove

                        # r2gen
                        pil_image = Image.fromarray((img_remove * 255).astype(np.uint8))

                        r2gen_transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406),
                                                 (0.229, 0.224, 0.225))])

                        img_gen_r2 = r2gen_transform(pil_image)
                        img_gen_r2 = img_gen_r2.unsqueeze(0)
                        tester = Tester(model, criterion, args, img_gen_r2)
                        gen_report = tester.test()
                        print(gen_report)

                        # get_chexbert_labels_for_gen_reports(chexbert, input_file, res_file, output_file)
                        return_dict = get_chexbert_labels_for_gen_reports_from_str(chexbert, gen_report)
                        return_label_str = ', '.join([condition for condition, labels in return_dict.items() if
                                                      condition != 'No Finding' and labels[0] == 1])

                        manipulate_data = pd.DataFrame(return_dict)
                        df = pd.concat([df, manipulate_data], ignore_index=True)
                        df.at[df.index[-1], 'id'] = str(step)
                        df.at[df.index[-1], 'img_path'] = batch['img_path'][0]
                        df.at[df.index[-1], 'image_id'] = batch['img_path'][0].split('/')[-1].split('.')[0]
                        df.at[df.index[-1], 'Diffusion_model'] = model_name
                        df.at[df.index[-1], 'Manipulate'] = 'remove'
                        if keyword == 'Effusion':
                            keyword = 'Pleural Effusion'
                        df.at[df.index[-1], 'Target'] = keyword
                        df.at[df.index[-1], 'Result'] = df.at[df.index[-1], keyword]
                        df.at[df.index[-1], 'Report'] = gen_report[0]

                        remove_result.append(df.at[df.index[-1], keyword])

                    remove_num_keywords = 0 if remove_keywords == [''] else len(remove_keywords)

                    if model_id == 0:
                        fig, ax = plt.subplots(2 + remove_num_keywords, 3 * model_number,
                                               figsize=(20, 4 * (2 + remove_num_keywords)))

                    ax[0, 0 + model_id * 3].imshow(init_image)
                    ax[0, 0 + model_id * 3].set_title('Initial Image')
                    ax[0, 0 + model_id * 3].axis('off')

                    # 绘制 reconstructed_image
                    ax[0, 1 + model_id * 3].imshow(img_re)
                    ax[0, 1 + model_id * 3].set_title('gt' + '\n' + gt_title_wrapped)
                    ax[0, 1 + model_id * 3].axis('off')

                    init_image_np = np.array(init_image) / 255.0
                    diff = abs(init_image_np - img_re)
                    vmax = diff.max()
                    vmin = diff.min()
                    diff_2d = np.clip((diff - vmin) / (vmax - vmin), 0, 1)
                    heatmap_gt = cv2.applyColorMap(np.uint8((1 - diff_2d) * 255), cv2.COLORMAP_JET)

                    ax[0, 2 + model_id * 3].imshow(heatmap_gt)
                    ax[0, 2 + model_id * 3].set_title('heatmap_gt')
                    ax[0, 2 + model_id * 3].axis('off')

                    ax[1, 0 + model_id * 3].imshow(init_image)
                    ax[1, 0 + model_id * 3].set_title('Initial Image')
                    ax[1, 0 + model_id * 3].axis('off')

                    ax[1, 1 + model_id * 3].imshow(img_gen)
                    ax[1, 1 + model_id * 3].set_title(
                        'r2gen_cmn' + '\n' + r2gen_title_wrapped + '\n' + return_label_str_r2gen)
                    ax[1, 1 + model_id * 3].axis('off')

                    diff = abs(init_image_np - img_gen)
                    vmax = diff.max()
                    vmin = diff.min()
                    diff_2d = np.clip((diff - vmin) / (vmax - vmin), 0, 1)
                    heatmap_gen = cv2.applyColorMap(np.uint8((1 - diff_2d) * 255), cv2.COLORMAP_JET)

                    ax[1, 2 + model_id * 3].imshow(heatmap_gen)
                    ax[1, 2 + model_id * 3].set_title('heatmap_r2gen_cmn')
                    ax[1, 2 + model_id * 3].axis('off')

                    for i in range(remove_num_keywords):
                        ax[i + 2, 0 + model_id * 3].imshow(img_gen)
                        ax[i + 2, 0 + model_id * 3].set_title('r2gen_cmn' + '\n' + r2gen_title_wrapped)
                        ax[i + 2, 0 + model_id * 3].axis('off')

                        ax[i + 2, 1 + model_id * 3].imshow(img_remove_dict[remove_keywords[i]])
                        ax[i + 2, 1 + model_id * 3].set_title(
                            'Remove ' + remove_keywords[i] + ':' + str(remove_result[i]))
                        ax[i + 2, 1 + model_id * 3].axis('off')

                        diff = abs(img_remove_dict[remove_keywords[i]] - img_gen)
                        vmax = diff.max()
                        vmin = diff.min()
                        diff_2d = np.clip((diff - vmin) / (vmax - vmin), 0, 1)
                        heatmap_ill_montage = cv2.applyColorMap(np.uint8((1 - diff_2d) * 255), cv2.COLORMAP_JET)

                        ax[i + 2, 2 + model_id * 3].imshow(heatmap_ill_montage)
                        ax[i + 2, 2 + model_id * 3].set_title(remove_keywords[i])
                        ax[i + 2, 2 + model_id * 3].axis('off')

        df.to_csv(csv_path, index=False)
        plt.tight_layout()
        plt.savefig(save_path + '.png')



if __name__ == "__main__":
    args = parse_args()
    main(args)
