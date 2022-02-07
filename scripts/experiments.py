import os
import evaluate_lsd.main as lsd
import pandas as pd

def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)

def log(path,name,comment,avg_lsd,min_lsd):
    """
    Created by Leander Maben.
    """
    df=pd.load_csv(path)
    df.loc[len(df.index)] = [name,comment,avg_lsd,min_lsd]
    df.to_csv(path,index=False)

## Experiment with data quantity
def vary_data(train_percents, test_percent, names, csv_path, n_epochs=75, n_epochs_decay=25, data_cache='/content/Pix2Pix-VC/data_cache',results_dir='/content/Pix2Pix-VC/results'):
    """
    Experiment with different size training sets and document lsd scores.

    Created by Leander Maben.
    """
    
    for train_p, name in zip(train_percents,names):
        print('#'*25)
        print(f'Training {name} with train_percent {train_p}')
        run(f'python datasets/fetchData.py --train_percent {train_p} --test_percent {test_percent}')
        run(f'python train.py --dataroot {data_cache} --name {name} --model pix2pix --direction AtoB --input_nc 1 --output_nc 1 --lambda_L1 200 --netG resnet_9blocks   --dataset_mode audio')
        run(f'python test.py --dataroot {data_cache} --name {name} --model pix2pix --direction AtoB --netG resnet_9blocks  --input_nc 1 --output_nc 1 --dataset_mode audio')
        avg_lsd, min_lsd = lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'))
        log(csv_path, name,'Training with {}',avg_lsd,min_lsd)
        shutil.rmtree(data_cache)
        print('Finished experiment with {name}')
        print('#'*25)
        

if __name__ == '__main__':
    csv_path = '/content/drive/MyDrive/NTU - Speech Augmentation/pix2pix.csv'
    if not os.path.exists(csv_path):
        cols=['name','comment','avg_lsd','min_lsd']
        df=pd.DataFrame(columns=cols)
        pd.to_csv(df,index=False)

    train_percents =[10,25,50,75]
    vary_data(train_percents,15,[f'pix_noisy_{i}' for i in train_percents], csv_path)
    
