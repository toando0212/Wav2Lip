import cv2
import numpy as np
import torch
import argparse
import audio
from models import Wav2Lip
import os

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', type=str, default='best.pt')
parser.add_argument('--face', type=str, required=True)
parser.add_argument('--audio', type=str, required=True)
parser.add_argument('--img_size', type=int, default=96)
parser.add_argument('--fps', type=float, default=25.)
parser.add_argument('--wav2lip_batch_size', type=int, default=16)
parser.add_argument('--display_size', type=int, default=384, help='Size of the displayed output window (output will be upscaled to this size)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _load(checkpoint_path):
    try:
        if device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    except Exception as e:
        print(f"Standard torch.load failed with error: {e}\nTrying torch.jit.load...")
        import io
        with open(checkpoint_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        if device == 'cuda':
            model = torch.jit.load(buffer)
        else:
            model = torch.jit.load(buffer, map_location='cpu')
        return {'state_dict': model.state_dict()}  # mimic expected output

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

def main():
    # Load face image
    frame = cv2.imread(args.face)
    frame = cv2.resize(frame, (args.img_size, args.img_size))
    frames = [frame]  # static image

    # Load audio and mel
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    mel_step_size = 16
    fps = args.fps
    mel_idx_multiplier = 80. / fps

    mel_chunks = []
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > mel.shape[1]:
            mel_chunks.append(mel[:, mel.shape[1] - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    model = load_model(args.checkpoint_path)
    print("Model loaded. Press 'q' to quit.")

    for idx, m in enumerate(mel_chunks):
        img_batch = []
        mel_batch = []
        frame_batch = []
        # Static image, so always use the same frame
        face = frame.copy()
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(face.copy())

        img_batch = np.asarray(img_batch)
        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch[0].shape[0], mel_batch[0].shape[1], 1])

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        p = cv2.resize(pred[0].astype(np.uint8), (args.img_size, args.img_size))


        # Upscale for display
        display_p = cv2.resize(p, (args.display_size, args.display_size), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Realtime Lip Sync', display_p)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
