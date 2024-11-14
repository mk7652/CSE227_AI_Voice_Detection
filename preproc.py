#Essential module import

import praatio
import praatio.audio
from src.Charsiu import charsiu_chain_attention_aligner, charsiu_forced_aligner, charsiu_attention_aligner
import os
import sys
import librosa
import soundfile as sf
import torch
import torchaudio
import matplotlib.pyplot as plt

def audio_loader(path: str) -> tuple[list[str], list[str]]:
    """
        Load audio file (.wav) and transcripts (.txt)
    """
    audios = []
    transcripts = []
    for f in os.listdir(path):
        if f.split('.')[1] == 'wav':
            audios.append(f)
        elif f.split('.')[1] == 'txt':
            transcripts.append(f)

    return audios, transcripts

def generate_alignments(path: str, audios: list[str], transcripts: list[str]) -> list[tuple]:
    """
        Generate alignments of phonemes from dictionaries of phonemes using attention-based aligner
    """
    alignments = []
    try:
        charsiu = charsiu_attention_aligner('charsiu/en_w2v2_fs_10ms')
        for voice, transcript in zip(audios, transcripts):
            script = open(path + transcript).read()
            alignment = charsiu.align(path + voice, script)
            charsiu.serve(audio=path + voice, text=script, save_to=path + voice.split('.')[0] + '.TextGrid')
            alignments.append(alignment)
    except Exception as e:
        import nltk
        nltk.download("averaged_perceptron_tagger")
        charsiu = charsiu_attention_aligner('charsiu/en_w2v2_fs_10ms')
        alignments = []
        for voice, transcript in zip(audios, transcripts):
            script = open(path + transcript).read()
            alignment = charsiu.align(path + voice, script)
            charsiu.serve(audio=path + voice, text=script, save_to=path + voice.split('.')[0] + '.TextGrid')
            alignments.append(alignment)
    return alignments

def filter_fricatives(alignments: list[tuple]) -> list[tuple]:
    fricatives = set(['F', 'Z', 'V', 'S', 'DH'])
    fricative_timestamps = []
    for alignment in alignments:
        filtered = [f for f in alignment if f[-1] in fricatives]
        fricative_timestamps.append(filtered)
    return fricative_timestamps

def generate_subwaves(path: str, audios: list[str], fricative_timestamps: list[tuple], outputdir: str) -> None:

    try:
        torch.set_default_device("cuda")
    except Exception as e:
        torch.set_default_device("cpu")
    finally:
        tensor_generator = lambda x: x.cpu() if torch.cuda.is_available else x

        for alignment, audio in zip(fricative_timestamps, audios):

            audio_name = audio.split('.')[0]

            waveform, sample_rate = torchaudio.load(path + audio)
            waveform = tensor_generator(waveform).numpy()
            num_channels, _ = waveform.shape
            target_dir = f"{outputdir}{audio_name}/"
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)
            
            for ind, fricative in enumerate(alignment):

                start, end = fricative[0], fricative[1]
                phoneme = fricative[-1]
                
                time_axis = torch.arange(start * sample_rate, end * sample_rate) / sample_rate
                subwave = waveform[:, int(start * sample_rate):int(end * sample_rate)] 
                figure, axes = plt.subplots(num_channels, 1)

                axes = [axes]
                axes[0].plot(tensor_generator(time_axis), subwave[0], linewidth=1)
                axes[0].grid(True)

                plt.ioff()        
                figure.suptitle(f"{audio_name}_{phoneme}_{ind}_Waveform")
                figure.savefig(target_dir + f"{audio_name}_{phoneme}_{ind}_Waveform.png")
                plt.close(figure)
                
                praatio.audio.extractSubwav(fn=path + audio, 
                                            outputFN=target_dir + f"{audio_name}_{phoneme}_{ind}_Waveform.wav",
                                            startTime=start, endTime=end)


if __name__ == "__main__":
    
    wav_path = "../sample_data/original/"

    output_path = "../sample_data/segments/"

    audios, transcripts = audio_loader(wav_path)

    alignments = generate_alignments(wav_path, audios, transcripts)

    fricative_timestamps = filter_fricatives(alignments)

    generate_subwaves(path=wav_path, audios=audios, fricative_timestamps=fricative_timestamps, outputdir=output_path)