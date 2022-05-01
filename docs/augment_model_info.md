# Definitions

T  := Transduction Model
A  := EMG Augmentation Model

Ev := 8 channels of 1D EMG recordings (8, time_length)
Av := Audio features (Mel spectrograms)
Tv := Text representation of an utterance

Tt := Target text

# Training and Evaluation

## Ground Truth Training

T(E'v) -> ^A'v (Transduction Model, Ground Truth EMG)
L(A'v, ^A'v)   (MSELoss(Pred Audio, Ground Truth Audio))

A(A'v) -> ^E'v (Vocalised EMG Augmentation Model)
L(E'v, ^E'v)   (MSELoss(Pred EMG, Ground Truth EMG))

## Ground Truth Evaluation

^Av = WaveGlow(^A'v))
wer = WER(Tv, ASR(^Av))

## Augmented Training

A'v = Tacotron2(Tt)
A(A'v) -> ^E'v
L(E'v, ^E'v)

## Augmented Evaluation

^Av = WaveGlow(^A'v))
wer = WER(Tt, ASR(^Av))

## Augmented Training Regime

wer_ground_only = WER(Tv, ASR(^Av))
wer_synth_only  = WER(Tt, ASR(^Av))
wer_combined    = 
    WER({Tv + Tt}, {ASR(T(^Av)) + ASR(A(^Av)))