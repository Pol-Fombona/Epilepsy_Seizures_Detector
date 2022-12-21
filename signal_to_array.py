import pandas as pd
import numpy as np
import os

    
def get_signal_array(size, seizure_start, seizure_end, shared_array, sampled = 128):

    size *= sampled

    seizure_start *= sampled
    seizure_end *= sampled

    pre_seizure_start = seizure_start - 30 * sampled
    pre_seizure_end = seizure_start

    post_seizure_start = seizure_end
    post_seizure_end = seizure_end + 30 * sampled

    if shared_array.size == 0:
        # Only one seizure in file
        signal_array = np.zeros(size, dtype= np.int64)

    else:
        # Two or more seizure in file
        signal_array = shared_array

    set_pre_seizure_array(signal_array, pre_seizure_start, pre_seizure_end)
    set_seizure_array(signal_array, seizure_start, seizure_end)
    set_post_seizure_array(signal_array, post_seizure_start, post_seizure_end, size)

    return signal_array


def set_pre_seizure_array(signal_array, pre_seizure_start, pre_seizure_end):
    # Set pre_seizure window in the array to 2

    if pre_seizure_start >= 0:
        signal_array[pre_seizure_start : pre_seizure_end] = 2

    elif pre_seizure_end >= 0:
        signal_array[: pre_seizure_end] = 2

    return signal_array


def set_post_seizure_array(signal_array, post_seizure_start, post_seizure_end, size):
    # Set post_seizure window in the array to 3

    if post_seizure_end < size:
        signal_array[post_seizure_start : post_seizure_end] = 3

    elif post_seizure_start < size:
        signal_array[post_seizure_start :] = 3

    return signal_array


def set_seizure_array(signal_array, seizure_start, seizure_end):
    # Set seizure window in the array to 1

    signal_array[seizure_start : seizure_end] = 1

    return signal_array


def signal2window(groundtruth, signal, split_size, overlap):

    windows = []

    interval_index = dict.fromkeys(set(groundtruth), 0)
    index_change_type = np.where(groundtruth[:-1] != groundtruth[1:])[0] + 1
    index_change_type = np.append(index_change_type, len(groundtruth))

    start_index = 0

    for end_index in index_change_type:

        signal_type = int(groundtruth[start_index])
        interval = interval_index[signal_type]


        window = get_window(start_index, end_index, signal, split_size, overlap, signal_type, interval)
        windows = windows + window
    
        interval_index[signal_type] += 1
        start_index = end_index

    columns = list(signal.columns[2:])
    columns = ["Timestamp(Hz)", "Type", "Intervals"] + columns
    windows_df = pd.DataFrame(windows, columns=columns)

    return windows_df


def get_window(start_index, end_index, signals, split_size, overlap, signal_type, interval):
   
    window = []

    if signal_type == 1 and overlap != 0:
        
        timestamps = [x for x in range(start_index, end_index, overlap)]
        types = [signal_type for x in range(start_index, end_index, overlap)]
        intervals = [interval for x in range(start_index, end_index, overlap)]


        for col in signals.columns[2:]:
            signal = list(signals[col])
            signal = signal[0][start_index:end_index]

            col_windows = [signal[i:i+split_size] for i in range(0, len(signal), overlap) if len(signal[i:i+split_size])==split_size]
            
            window.append(col_windows)

        len_windows = len(window[0])

        window.insert(0, timestamps[:len_windows])
        window.insert(1, types[:len_windows])
        window.insert(2, intervals[:len_windows])


    else:
        timestamps = [x for x in range(start_index, end_index, split_size)]
        types = [signal_type for x in range(start_index, end_index, split_size)]
        intervals = [interval for x in range(start_index, end_index, split_size)]

        window.append(timestamps)
        window.append(types)
        window.append(intervals)

        for col in signals.columns[2:]:
            signal = list(signals[col])
            signal = signal[0][start_index:end_index]
            
            col_windows = np.split(signal, np.arange(split_size, len(signal), split_size))

            if (len(signal) % (split_size)) != 0:
                col_windows = col_windows[:-1]

            window.append(col_windows)


    window_2 = list(zip(*window))

    return window_2






def get_window2(signal_partition, split_size, sampled, overlap, window_set, timestamp):

    length = split_size * sampled
    signal_type = int(signal_partition[0])

    if length > signal_partition.size:
        print("Partition bigger than section")
        exit()

    if signal_partition[0] == 1 and overlap:
        # If seizure data and overlap
        #overlap_size = 128 * 0.25
        overlap_ratio = 0.125
        overlap_size = sampled * overlap_ratio

        start = 0
        end = length

        while end <= signal_partition.size:

            window_set.append((timestamp, signal_type, overlap_ratio, signal_partition[start:end]))


            timestamp += split_size * overlap_ratio
            start += int(overlap_size)
            end += int(overlap_size) 

        if (end - overlap_size) < signal_partition.size:
            
            timestamp += (signal_partition.size - (end - overlap_size)) - length * (overlap_size / split_size)
            window_set.append((timestamp, signal_type, overlap_ratio, signal_partition[-length:]))

            timestamp += split_size

        else:
            timestamp -= split_size*overlap_ratio
            timestamp += split_size

        return window_set, timestamp
        
    
    else:
        partitions = int(signal_partition.size / length)

        start, end = 0, 1

        for i in range(partitions):
            
            window_set.append((timestamp, signal_type, 0, signal_partition[start * length : end * length]))

            timestamp += split_size
            start = end
            end += 1

        if (signal_partition.size % length) != 0:
            # If last split size is smaller than length
            #window_set.append(signal_partition[-length:])

            # size = 320
            # split_size = 2
            # sampled = 128
            # length = 256

            timestamp += (signal_partition.size % length) / length * split_size - split_size
            window_set.append((timestamp, signal_type, 0, signal_partition[-length:]))
            timestamp += split_size

        return window_set, timestamp


def signal2array(path, path_to_save):

    df_annotation = pd.read_excel(path)

    data = df_annotation.loc[df_annotation["type"] == "seizure"]
    data = data.groupby("filename")

    signals_arrays = []

    for name, group in data:

        shared_array = np.array([])

        for index in group.index:
    
            size = group["rec_duration"][index]
            seizure_start, seizure_end = group["seizure_start"][index], group["seizure_end"][index]

            signal_array = get_signal_array(size, seizure_start, seizure_end, shared_array, sampled=128)

            shared_array = signal_array

        
        file = group["filename"][index]
        patID = group["PatID"][index]
        signal_type = group["type"][index]
        
        signals_arrays.append((patID, file, signal_type, signal_array))
    
    signals_arrays = pd.DataFrame(signals_arrays, columns=["PatID", "File", "Type", "SignalArray"])
    signals_arrays.to_pickle(path_to_save)

    return signals_arrays


def transform_dataset(path):


    df_list = []

    for file in os.listdir(path):

        if ".parquet" in file:
            
            print(file, "is being transformed")

            df =pd.read_parquet(path+file, engine="fastparquet")
            
            new_df = df.loc[df['type'] == "seizure"].drop(["type"], axis=1).groupby(['filename', "PatID"]).agg(list)
            
            if not new_df.empty:

                new_df = new_df.reset_index()
                df_list.append(new_df)


    df = pd.concat(df_list, ignore_index=True)
    df.to_pickle(path + "Data\\signals_combination.pkl")

    return df


def get_transformed_dataset(df):

    new_df = df.loc[df['type'] == "seizure"].drop(["type"], axis=1).groupby(['filename', "PatID"]).agg(list)
    new_df = new_df.reset_index()
    return new_df


if __name__ == "__main__":

    #df_combination = transform_dataset(path = "C:\\Users\\Admin\\Desktop\\parquet\\")
    #groundtruth_df = signal2array(path = "C:\\Users\\Admin\\Desktop\\parquet\\df_annotation_full.xlsx",
    #                                path_to_save="C:\\Users\\Admin\\Desktop\\parquet\\Data\\groundtruth.pkl")

    groundtruth_df = pd.read_pickle("files\\groundtruth.pkl")
    parquet_files =  list([x for x in os.listdir("parquetFiles/") if x.endswith(".parquet")])
    
    sampled = 128
    overlap_ratio = 1/8
    window_size = 1 * sampled
    overlap = int(overlap_ratio * sampled)

    for index, row in groundtruth_df.iterrows():

        filename = row["File"]
        patID = row["PatID"]
        
        parquet_file_name = patID + "_raw_eeg_128.parquet"

        if parquet_file_name in parquet_files:

            df = pd.read_parquet("parquetFiles/" + parquet_file_name, engine="fastparquet")

            if filename in df["filename"].unique():
                print("Processing File", filename)
                df = get_transformed_dataset(df)
                df = df[df["filename"] == filename]

                groundtruth_array = row["SignalArray"]
                window = signal2window(groundtruth_array, df, window_size, overlap)
                
                path_to_save = ("window_array\\" + filename.split(".")[0]
                                + "-Sampled_"+str(sampled)
                                + "-WinSize_"+str(window_size)
                                + "-Overlap_" + str(overlap) 
                                + ".pkl")

                window.to_pickle(path_to_save)
                print(filename, "done")
                print("#############")

    
    
    
