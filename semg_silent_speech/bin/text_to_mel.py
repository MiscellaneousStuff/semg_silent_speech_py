import os
import sys

# NOTE: These need to be added so tacotron2 and waveglow submodules can properly import
#       modules
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../waveglow/')))
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../waveglow/tacotron2')))

from semg_silent_speech.models.text_to_mel import infer, text_to_mel

from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string("waveglow_checkpoint", None, "Path to a waveglow *.pt model checkpoint")
flags.DEFINE_string("tacotron2_checkpoint", None, "Path to a tacotron2 *.pt model checkpoint")
flags.DEFINE_string("text", None, "Target text to generate mel_spectogram for")
flags.DEFINE_string("audio_path", None, "(Optional) Generate and save audio of the mel_spectogram")
flags.DEFINE_bool("denoise", None, "(Optional) Removes noise from audio output")
flags.mark_flag_as_required("waveglow_checkpoint")
flags.mark_flag_as_required("tacotron2_checkpoint")

def save_output(waveglow_path, mel_spectogram, sr, audio_path, denoise=False):
    infer(waveglow_path, mel_spectogram, sr, audio_path, denoise)

def main(unused_argv):
    text = FLAGS.text
    if not text == None:
        waveglow_checkpoint_path = FLAGS.waveglow_checkpoint
        tacotron2_checkpoint_path = FLAGS.tacotron2_checkpoint
        audio_path = FLAGS.audio_path
        denoise = FLAGS.denoise
        save_output(waveglow_checkpoint_path,
                    text_to_mel(tacotron2_checkpoint_path, text),
                    22_050,
                    audio_path,
                    denoise)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)