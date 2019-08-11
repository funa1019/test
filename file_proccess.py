import logging
import os
import re
from random import random


def remove_other_files(extension, archive_dir):
    for root, dirs, files in os.walk(archive_dir):
        for current_file in files:

            if not current_file.lower().endswith(extension):

                try:
                    logging.debug("Removing resource: File [%s]", os.path.join(root, current_file))
                    os.remove(os.path.join(root, current_file))
                except OSError:
                    logging.error("Could not remove resource: File [%s]", os.path.join(root, current_file))


def create_category_folder(category0, category1, source_dir):
    for new_dir in ['train', 'predict']:
        for new_category in [category0, category1]:
            abspath_dir = os.path.abspath(os.path.join(source_dir, new_dir, new_category))

            logging.info(
                "Creating resource: Directory [%s]", abspath_dir)
            os.makedirs(abspath_dir)


def label_files(directory, category_rules, source_dir):
    predict_ratio = 0.1

    for root, dirs, files in os.walk(directory):
        for file in files:

            if re.compile(list(category_rules.values())[0]).match(file):

                if random() < predict_ratio:
                    train_test_dir = 'predict/'

                else:
                    train_test_dir = 'train/'

                try:
                    logging.debug(
                        "Moving %s from %s to %s", file, root,
                        os.path.join(source_dir, train_test_dir, list(category_rules.keys())[0]))

                    os.rename(
                        os.path.join(root, file),
                        os.path.join(source_dir, train_test_dir, list(category_rules.keys())[0], file))

                except OSError:
                    logging.error("Could not move %s ", os.path.join(root, file))

            elif re.compile(list(category_rules.values())[1]).match(file):

                if random() < predict_ratio:
                    train_test_dir = 'predict/'

                else:
                    train_test_dir = 'train/'

                try:
                    logging.debug("Moving %s from %s to %s", file, root,
                                  os.path.join(source_dir, train_test_dir, list(category_rules.keys())[1]))

                    os.rename(
                        os.path.join(root, file),
                        os.path.join(source_dir, train_test_dir, list(category_rules.keys())[1], file))

                except OSError:
                    logging.error("Could not move %s ", os.path.join(root, file))

            else:
                logging.error("No files matching category regex")


data_org_directory = '/media/cs/data/BreaKHis_v1'  # raw data path
data_mdy_directory = '/media/cs/data/BreaKHis_v1/dataset'  # modified data stored path

remove_other_files('.png', data_org_directory)
create_category_folder('benign', 'malignant', data_mdy_directory)
label_files(data_org_directory, {'benign': 'SOB_B_.*.png', 'malignant': 'SOB_M_.*.png'}, data_mdy_directory)
