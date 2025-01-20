import random
import sys
from importlib.resources import files
import numpy as np

import soundfile as sf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tqdm

from src.utils import (
    hop_length,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    transcribe,
    target_sample_rate
)

from model import DiT, UNetT
from model.utils import seed_everything
from src.utils import chunk_text

class TTS:
    def __init__(
        self,
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_name="vocos",
        local_path=None,
        device=None,
        hf_cache_dir=None,
    ):
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.mel_spec_type = vocoder_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=120,
            chunk_overlap=0,
            length_function=len,
            separators=[", ", " ", ". "],
        )

        # Set device
        if device is not None:
            self.device = device
        else:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Load models
        self.load_vocoder_model(vocoder_name, local_path=local_path, hf_cache_dir=hf_cache_dir)
        self.load_ema_model(
            ckpt_file, vocoder_name, vocab_file, ode_method, use_ema, hf_cache_dir=hf_cache_dir
        )

    def load_vocoder_model(self, vocoder_name, local_path=None, hf_cache_dir=None):
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device, hf_cache_dir)

    def load_ema_model(self, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, hf_cache_dir=None):
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        model_cls = DiT

        self.ema_model = load_model(
            model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, self.device
        )

    def transcribe(self, ref_audio, language=None):
        return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spect, file_spect):
        save_spectrogram(spect, file_spect)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spect=None,
        seed=-1,
        language="vi"
    ):
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text, duration = preprocess_ref_audio_text(ref_file, ref_text, device=self.device)

        if language == "vi":
            gen_text_length = len(gen_text.split())
            if gen_text_length >= 5:
                fix_duration = None
            elif gen_text_length > 1:
                fix_duration = duration + 0.3 * gen_text_length
            else:
                fix_duration = duration + 0.4

        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text.lower(),
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
            language=language
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spect is not None:
            self.export_spectrogram(spect, file_spect)

        return wav, sr, spect
    
    def chunk_text(self, text, max_chars=135):
        """
        Splits the input text into chunks, each with a maximum number of characters.

        Args:
            text (str): The text to be split.
            max_chars (int): The maximum number of characters per chunk.

        Returns:
            List[str]: A list of text chunks.
        """
        text = text.replace(" , ", ", ")
        text = text.replace(" . ", ", ")
        chunks = self.text_splitter.split_text(text)
        chunks = [sentence.strip() for sentence in chunks]
        chunks = [", ".join(sentence.split(", ")[1: ]) if sentence.startswith(", ") else sentence for sentence in chunks]
        chunks = [", ".join(sentence.split(". ")[1: ]) if sentence.startswith(", ") else sentence for sentence in chunks]
        chunks = [sentence.replace(", ", " , ") for sentence in chunks]

        return chunks
    
    def vie_inference(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        remove_silence=False,
        file_wave=None,
        file_spect=None,
        seed=-1,
        language="vi"
    ):

        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        all_wav = []
        chunks = self.chunk_text(gen_text)

        ref_file, ref_text, duration = preprocess_ref_audio_text(ref_file, ref_text, device=self.device)
        speaker_speed = duration / len(ref_text.split())

        for sentence in tqdm.tqdm(chunks):
            gen_text_length = len(sentence.replace(" , ", " ").split())
            if gen_text_length > 5:
                fix_duration = None
            elif gen_text_length > 1:
                fix_duration = duration + 0.3 * gen_text_length
            else:
                fix_duration = duration + 0.4
            
            wav, sr, spect = infer_process(
                ref_file,
                ref_text,
                sentence.lower(),
                self.ema_model,
                self.vocoder,
                self.mel_spec_type,
                show_info=show_info,
                progress=progress,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
                fix_duration=fix_duration,
                device=self.device,
                language="vi"
            )

            all_wav.append(wav)
        
        all_wav = np.concatenate(all_wav)

        if file_wave is not None:
            self.export_wav(all_wav, file_wave, remove_silence)

        return all_wav, sr, spect


if __name__ == "__main__":
    # tts = TTS(
    #     ckpt_file="model_checkpoints/model_eng.pt",
    #     vocab_file="model_checkpoints/vocab_eng.txt"
    # )

    # wav, sr, spect = tts.infer(
    #     ref_file="samples/reference_eng.wav",
    #     ref_text="laughing and uproarious, utterly unmindful of the companionship of men upon whom lay the shadow of an impending shameful death",
    #     gen_text="With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural",
    #     file_wave="output.wav",
    #     language="en"
    # )

    tts = TTS(
        ckpt_file="model_checkpoints/model_vie.pt",
        vocab_file="model_checkpoints/vocab_vie.txt"
    )

    wav, sr, spect = tts.vie_inference(
        ref_file="samples/reference_1.wav",
        ref_text="vừa rồi xem qua danh sách quà tặng của hồi môn của cô dâu chỉ riêng tiền mặt đã là một tỉ.",
        gen_text="tiểu cân cười đắc ý chạy đi, không quên châm chọc lâm ngự, đừng nằm mơ, tao sẽ đi báo cáo chúng mày, hôm nay có đội trưởng giám ngục đi tuần, mày có hối lộ thuốc lá cũng vô dụng, ngày chết của mày tới rồi, chờ bị trừng phạt đi, thế là tiểu cân nhanh chóng chạy đến mách lẻo đội quỷ đang đi tuần ở phía trước, đội trưởng tiên sinh, giám ngục trưởng lạnh giọng hỏi, xảy ra chuyện gì, sao ồn ào vậy hả, tiểu cân liền đáp, đội trưởng tiên sinh, tôi có chuyện muốn nói, tên đó, hắn lén lút vào sâu bên trong trộm khoáng thạch làm của riêng, đội trưởng tiên sinh, ngài nhất định phải trừng phạt hắn, vừa nói hắn vừa chỉ tay về phía lâm ngự đang chậm rãi bước đến, nào ngờ giám ngục trưởng nhìn thấy lâm ngự lại rất khách sáo hỏi, quản lý đại nhân, sao ngài lại một mình đi vào xem vậy ạ, sao không nói để tôi kêu bọn lão hồng đi cùng ngài, ngài yên tâm, chất lượng khoáng thạch ở hai chỗ này vô cùng tốt, tuyệt đối còn thua kém bên ngoài,",
        file_wave="output.wav",
        language="vi"
    )