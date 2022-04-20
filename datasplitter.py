import pandas as pd
import random
from tqdm import tqdm
import cv2 as cv
import os


class DataSplitter:
    def __init__(self, filelist: str, split_dict: dict):
        self.data_df = self.get_dataframe(filelist)
        self.split_dict = split_dict

        self.orig_df = self.data_df[self.data_df.multiscale == False]
        self.multiscale_df = self.data_df[self.data_df.multiscale == True]

        # get original splits
        print('====GETTING ORIGINAL SPLITS====')
        self.orig_train_df, self.orig_val_df, self.orig_test_df = self.get_orig_scale_split()

        # get multiscale splits
        print('====GETTING FINAL MULTISCALE SPLITS====')
        self.final_splits = self.get_final_splits()

    def write_files(self, target_folder):
        for key, value in self.final_splits.items():
            print(f"==Processing {key} datasplit==")
            for f in tqdm(value.filepath.values):
                img = cv.imread(f)
                filename = f.split('/')[-1]
                save_path = f'{target_folder}/{key}'
                self._create_folders(save_path)

                cv.imwrite(f"{save_path}/{filename}", img)

    def get_final_splits(self):
        output_dict = {
            'train': None,
            'val': None,
            'test': None
        }
        for key, value in tqdm(output_dict.items()):
            df_name = f"self.orig_{key}_df"
            multi_df = self.multiscale_df[self.multiscale_df.base_filename.isin(eval(df_name).base_filename)]
            output_df = pd.concat([eval(df_name), multi_df])
            output_dict[key] = output_df
        return output_dict

    def get_orig_scale_split(self):

        out_train = []
        out_val = []
        out_test = []
        for key, value in tqdm(split_dict.items()):
            sds = self.orig_df[self.orig_df.dataset == key]
            if self.split_dict[key][3] != None:
                train_cap = int(min(len(sds) * self.split_dict[key][0], self.split_dict[key][3]))
            else:
                train_cap = int(len(sds) * self.split_dict[key][0])
            val_cap = train_cap + int(len(sds) * self.split_dict[key][1])
            test_cap = val_cap + int(len(sds) * self.split_dict[key][2])

            # create splits
            train_df = sds[:train_cap]
            val_df = sds[train_cap:val_cap]
            test_df = sds[val_cap:test_cap]

            # append
            out_train.append(train_df)
            out_val.append(val_df)
            out_test.append(test_df)

        out_train_df = pd.concat(out_train)
        out_val_df = pd.concat(out_val)
        out_test_df = pd.concat(out_test)

        return out_train_df, out_val_df, out_test_df

    @staticmethod
    def _create_folders(target_folder: str) -> None:
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

    @staticmethod
    def get_dataframe(filelist) -> pd.DataFrame:
        out_dict = {
            'filename': [],
            'filepath': [],
            'multiscale': [],
            'dataset': [],
            'base_filename': []
        }
        for i in filelist:
            # set values
            multiscale = False
            filename = i.split('/')[-1]
            filepath = i
            if '_multiscale' in i:
                multiscale = True
            dataset = i.split('/')[-2]
            base_filename = filename.split('.')[0]
            if multiscale == True:
                base_filename = filename.split('.')[0][:-2]
            # append values
            for key, _ in out_dict.items():
                out_dict[key].append(eval(key))
        return pd.DataFrame(out_dict)

    @property
    def orig_splits(self):
        return self.orig_train_df, self.orig_val_df, self.orig_test_df

    @property
    def debug(self):
        return self.multiscale_df, self.final_splits


if __name__ == '__main__':
    import glob
    filelist = glob.glob('/home/adithya/adithya/Real-ESRGAN/data/**/*')
    split_dict = {
        'klothed-bloomingdales-images': [0.8, 0.1, 0.1, 20000],
        'klothed-brunellocucienelli-image': [0.8, 0.1, 0.1, None],
        'klothed-oldnavy-images': [0.8, 0.1, 0.1, None],
        'klothed-uniqlo-images': [0.8, 0.1, 0.1, None]
    }
    output = DataSplitter(filelist, split_dict)
    output.write_files('data/super_res_dataset')
