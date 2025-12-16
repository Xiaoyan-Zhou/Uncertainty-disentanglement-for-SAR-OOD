### rewrite excel based on many excels under different seeds.
import os
import pandas as pd


def read_write_data(folder_path, writer, sheet_name, save_sheet_name, loss='EDL', decomposition_flag=True):
    # 获取所有xlsx文件
    # excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    lambdas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
    decomposition = ['entropy', 'evidence', 'variance']
    if decomposition_flag:
        excel_files = ([f"ours_kl_lambda_{l}_{loss}.xlsx" for l in lambdas] +
            [f"{d}_{loss}_baseline.xlsx" for d in decomposition])
    else:
        excel_files = [f"ours_kl_lambda_{l}_{loss}.xlsx" for l in lambdas]

    dfs=[]
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        if sheet_name == 'accuracy_dataset':
            if ('_EDL_' in file_path) or ('_UMSE_' in file_path):
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                df['source_file'] = file_path.split('/')[-1]
                dfs.append(df)
            elif 'ours' in file_path:
                df = pd.read_excel(file_path, sheet_name='AU '+ sheet_name)
                df['source_file'] = file_path.split('/')[-1]
                dfs.append(df)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df['source_file'] = file_path.split('/')[-1]
            dfs.append(df)

    combined_df = pd.concat(dfs, axis=1)
    # combined_df.to_excel(save_xlsx_path, sheet_name=save_sheet_name, index=False)
    combined_df.to_excel(writer, sheet_name=save_sheet_name, index=False)

if __name__ == '__main__':
    # 所有Excel文件所在的文件夹路径
    folder_path = 'models_trained_final/bs8_deactivate/seed_3407'  # 请修改为你的文件夹路径
    save_path = 'results/models_trained_final/bs8_deactivate/'
    save_sheet_name='seed_3407'
    decomposition_flag=False
    loss='UMSE' #'UMSE', 'EDL'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ### data processing of accuracy of AU branch
    output_path_AU_acc = os.path.join(save_path, 'AU_acc.xlsx')
    sheet_name='accuracy_dataset'
    mode = "a" if os.path.exists(output_path_AU_acc) else "w"
    if mode == 'a':
        with pd.ExcelWriter(output_path_AU_acc, engine='openpyxl', mode=mode, if_sheet_exists="replace") as writer:
            read_write_data(folder_path, writer, sheet_name, save_sheet_name, loss, decomposition_flag)
    else:
        with pd.ExcelWriter(output_path_AU_acc, engine='openpyxl', mode=mode) as writer:
            read_write_data(folder_path, writer, sheet_name, save_sheet_name, loss, decomposition_flag)

     ### data processing of accuracy of EU branch on OOD detection task
    output_path_OOD_EU = os.path.join(save_path, 'OOD_EU.xlsx')
    sheet_name='epistemic uncertainty'
    mode = "a" if os.path.exists(output_path_OOD_EU) else "w"
    if mode == 'a':
        with pd.ExcelWriter(output_path_OOD_EU, engine='openpyxl', mode=mode, if_sheet_exists="replace") as writer:
            read_write_data(folder_path, writer, sheet_name, save_sheet_name, loss, decomposition_flag)
    else:
        with pd.ExcelWriter(output_path_OOD_EU, engine='openpyxl', mode=mode) as writer:
            read_write_data(folder_path, writer, sheet_name, save_sheet_name, loss, decomposition_flag)


    ### data processing of accuracy of AU branch on misclassification detection task
    output_path_misclassification_AU = os.path.join(save_path, 'misclassification_AU.xlsx')
    sheet_name='aleatoric uncertainty'    
    mode = "a" if os.path.exists(output_path_misclassification_AU) else "w"
    if mode == 'a': 
        with pd.ExcelWriter(output_path_misclassification_AU, engine='openpyxl', mode=mode, if_sheet_exists="replace") as writer:
            read_write_data(folder_path, writer, sheet_name, save_sheet_name, loss, decomposition_flag)
    else:
        with pd.ExcelWriter(output_path_misclassification_AU, engine='openpyxl', mode=mode) as writer:
            read_write_data(folder_path, writer, sheet_name, save_sheet_name, loss, decomposition_flag)
