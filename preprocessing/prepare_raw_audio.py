"""
Created on Fri March 22, 2019

@author: Gustavo Cid Ornelas
"""
import scipy.io.wavfile
import numpy as np

from sklearn import preprocessing


if __name__ == "__main__":
    """
    Script that gets the raw audio from the IEMOCAP dataset. Should be executed only once to get the FC_raw_audio.csv 
    file, which contains the ids and audio samples for the data that is used by our model. We truncated/zero-padded
    everything to 150.000 samples
    """

    # output file
    out_file = "../data/processed-data/FC_raw_audio.npy"

    # reading all of the ids that are going to be used
    with open("../data/processed-data/FC_ordered_ids.txt") as f:
        ordered_ids = f.readlines()

    file_count = 0

    # every audio should have the same length (150.000) for the batches
    audio_data = np.zeros((len(ordered_ids), 150000))

    with open(out_file, "w") as f:
        # finding the corresponding .wav files specified in ordered_ids
        for row, id in enumerate(ordered_ids):
            current_session = id[4]
            partial_id = id[0:-6]

            audio_file = "../data/raw-data/IEMOCAP_full_release/Session" + current_session + "/sentences/wav/" + \
                         partial_id + "/" + id[0:-1] + ".wav"

            # reading the audio file
            _, samples = scipy.io.wavfile.read(audio_file)

            # standardizing the audio samples to have zero mean and unit variance
            samples = preprocessing.scale(samples.astype(float))

            # zero padding the audio samples
            if samples.shape[0] < 150000:
                len_pad = 150000 - samples.shape[0]
                zero_pad = np.zeros(len_pad)
                padded_samples = np.concatenate((samples, zero_pad))
                audio_data[row, :] = padded_samples
            elif samples.shape[0] > 150000:
                samples = samples[:150000]
                audio_data[row, :] = samples

            file_count += 1

            if file_count % 100 == 0:
                print(str(round(100 * file_count/len(ordered_ids), 2)) + "% of the files read...")

    print("Done!")

    # saving the padded audio data
    np.save(out_file, audio_data)


