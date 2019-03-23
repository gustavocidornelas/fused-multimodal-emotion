"""
Created on Fri March 22, 2019

@author: Gustavo Cid Ornelas
"""
import scipy.io.wavfile
import csv

if __name__ == "__main__":
    """
    Script that gets the raw audio from the IEMOCAP dataset. Should be executed only once to get the FC_raw_audio.csv 
    file, which contains the ids and audio samples for the data that is used by our model.
    """

    # output file
    out_file = "../data/processed-data/FC_raw_audio.csv"

    # reading all of the ids that are going to be used
    with open("../data/processed-data/FC_ordered_ids.txt") as f:
        ordered_ids = f.readlines()

    file_count = 0

    with open(out_file, "w") as f:
        # finding the corresponding .wav files specified in ordered_ids
        for id in ordered_ids:
            current_session = id[4]
            partial_id = id[0:-6]

            audio_file = "../data/raw-data/IEMOCAP_full_release/Session" + current_session + "/sentences/wav/" + partial_id + "/" + id[0:-1] + ".wav"

            # reading the audio file
            fs, samples = scipy.io.wavfile.read(audio_file)

            # writing the results to the csv file
            csv_writer = csv.writer(f)
            csv_writer.writerow([id, samples])

            file_count += 1

            if file_count % 100 == 0:
                print(str(100 * file_count/len(ordered_ids)) + " % of the files read...")

    print("Done!")

