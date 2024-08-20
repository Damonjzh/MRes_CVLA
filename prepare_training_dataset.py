from src.path_datasets_and_weights import path_chexbert_weights

from collections import defaultdict
import csv
import pandas as pd
import os
import evaluate
import spacy
import tempfile
from src.CheXbert.src.label import label
from src.CheXbert.src.models.bert_labeler import bert_labeler

import torch
import torch.nn as nn
from src.full_model.train_full_model import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_chexbert():
    model = bert_labeler()
    model = nn.DataParallel(model)
    checkpoint = torch.load(path_chexbert_weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    return model

def convert_labels_like_miura(preds_reports: list[list[int]]):
    """
    See doc string of update_clinical_efficacy_scores function for more details.
    Miura (https://arxiv.org/pdf/2010.10042.pdf) considers blank/NaN (label 0) and negative (label 2) to be the negative class,
    and positive (label 1) and uncertain (label 3) to be the positive class.

    Thus we convert label 2 -> label 0 and label 3 -> label 1.
    """

    def convert_label(label: int):
        if label == 2:
            return 0
        elif label == 3:
            return 1
        else:
            return label

    preds_reports_converted = [[convert_label(label) for label in condition_list] for condition_list in preds_reports]

    return preds_reports_converted

def get_chexbert_labels_for_gen_reports(chexbert, input_file, res_file, output_file):
    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile)
        image_id = [str(row['image_id']) for row in reader]  # 获取 image_id 列

    res = []
    with open(res_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming each row contains one report
            res.append(row[0])

    with tempfile.TemporaryDirectory() as temp_dir:
        csv_gen_reports_file_path = os.path.join(temp_dir, "gen_reports.csv")

        header = ["Report Impression"]

        with open(csv_gen_reports_file_path, "w") as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(header)
            csv_writer.writerows([[gen_report] for gen_report in res])

        # preds_*_reports are List[List[int]] with the labels extracted by CheXbert (see doc string for details)
        preds_gen_reports = convert_labels_like_miura(label(chexbert, csv_gen_reports_file_path))

    CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
                  'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                  'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']

    df_res = pd.DataFrame({condition: labels for condition, labels in zip(CONDITIONS, preds_gen_reports)})

    df_res.insert(0, 'image_id', image_id)

    df_res.to_csv(output_file, index=False)


path_chexbert_weights = "./chexbert.pth"
input_file = './test_samples.csv'
res_file = './combined-res.csv'
output_file = './train_label.csv'

chexbert = get_chexbert()
get_chexbert_labels_for_gen_reports(chexbert, input_file, res_file, output_file)