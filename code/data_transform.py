import pandas as pd
import numpy as np
import os

# Iterate on a single column and check how many videos exist
def getTotalRows(task):
    total_new_rows = 2
    for val in task:
        if (val != 0):
            if check_exist[3] in str(val).split(', '):
                total_new_rows = 4
                break
            elif check_exist[2] in str(val).split(', '):
                total_new_rows = 3
            elif (check_exist[1] in str(val).split(', ')) and total_new_rows != 3:
                total_new_rows = 2

    return total_new_rows


root_dir = '../dsp_intent_analyzer_dataset/forms'

csv_filenames = ['001',
                '002',
                '003']

codename = {
    'Quiatchon, Pauline Rose': '001',
    'Adamos, Jedd': '002',
    'Rafael, Angelique': '003'}

gt_tasks = { '001': {'Task1': 'Undetermined', 'Task2': 'Eat', 'Task3': 'Spontaneous', 'Task4': 'Go Outside', 'Task5': 'Study', 'Task6': 'Drink'},
            '002': {'Task1': 'Eat', 'Task2': 'Spontaneous', 'Task3': 'Drink', 'Task4': 'Study', 'Task5': 'Undetermined', 'Task6': 'Go Outside'},
            '003': {'Task1': 'Go Outside', 'Task2': 'Drink', 'Task3': 'Study', 'Task4': 'Spontaneous', 'Task5': 'Undetermined', 'Task6': 'Eat'}
          }



# Check how many videos exist
check_exist = ['Video 1', 'Video 2', 'Video 3', 'Video 4']

# Range of tasks
task_ranges = [  [i for i in range(2,15)],
            [i for i in range(15,28)],
            [i for i in range(28,41)],
            [i for i in range(41,54)],
            [i for i in range(54,67)],
            [i for i in range(67,80)]]

# Create data which would be the resulting final data
cols = ['subject','observer','backpack','umbrella','racket','utensils','bowl','bottle','cup','fruits','sandwich','laptop','clock','book','human_task','gt_task']
master_data = pd.DataFrame(columns=cols)
error_data = []

for name in csv_filenames:
    subject = name
    csv_file = os.path.join(root_dir,f'{subject}.csv')
    subject_data = pd.read_csv(csv_file)

    # Set all NaN to 0
    # copy = subject_data.copy(deep=True)
    subject_data.fillna(0)

    for num_observer in range(len(subject_data)):
        observer = subject_data.iloc[num_observer,:]
        observer_proc_data = pd.DataFrame(columns=cols)

        for num_task,task_range in enumerate(task_ranges):
            task_data = observer[task_range].fillna(0)
            num_rows = getTotalRows(task_data)
            tasks = []

            for video_iter in range(num_rows):
                # Initialize an instance
                # Each instance contains a video and its labels
                vid_data = {}
                vid_data['subject'] = subject
                vid_data['observer'] = codename[observer[1]]
                vid = check_exist[video_iter]

                for col,data in zip(cols[2:15],task_data[:13]):
                    if vid in str(data).split(', '):
                        vid_data[col] = 1
                    else:
                        vid_data[col] = 0

                vid_data['human_task'] = task_data[12]
                vid_data['gt_task'] = gt_tasks[subject][f'Task{num_task+1}']
                if gt_tasks[subject][f'Task{num_task+1}'] != task_data[12]:
                    error_data.append((subject, codename[observer[1]], num_task+1))

                tasks.append(vid_data)


            observer_proc_data = observer_proc_data.append(pd.DataFrame(tasks), ignore_index=True)


        master_data = master_data.append(observer_proc_data, ignore_index=True)

print(master_data.tail())
