
import os


# 2023-10-26
#     12-15
#     03-20
# 2023-10-22
#     00-16
#     00-10
def get_latest_model_checkpoint_path(parent_dir):
    file_name = 'models/last.ckpt'
    date_folders = os.listdir(parent_dir)
    print("date_folders:", date_folders)
    latest_checkpoint_path = None

    for date_folder in date_folders:
        time_folders = os.listdir(os.path.join(parent_dir, date_folder))
        time_folders_with_file = [time_folder for time_folder in time_folders if
                                  os.path.isfile(os.path.join(parent_dir, date_folder, time_folder, file_name))]
        sorted_time_folders_with_file = sorted(time_folders_with_file, reverse=True)
        if sorted_time_folders_with_file:
            latest_checkpoint_path = os.path.join(parent_dir, date_folder, sorted_time_folders_with_file[0])
            break

    return str(latest_checkpoint_path)+"/"+file_name