import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from models import Wav2Lip
import audio
import face_detection

parser = argparse.ArgumentParser(description='Kaggle-ready Wav2Lip inference')
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to Wav2Lip checkpoint')
parser.add_argument('--face', type=str, required=True, help='Path to input face video/image')
parser.add_argument('--audio', type=str, required=True, help='Path to input audio file (wav)')
parser.add_argument('--outfile', type=str, default='/kaggle/working/result_voice.mp4', help='Output video path')
args = parser.parse_args()

args.img_size = 96
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} for inference.')

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
    batch_size = 16
    predictions = []
    for i in tqdm(range(0, len(images), batch_size)):
        predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
    results = []
    for rect, image in zip(predictions, images):
        if rect is None:
            raise ValueError('Face not detected!')
        y1, y2, x1, x2 = rect[1], rect[3], rect[0], rect[2]
        results.append([x1, y1, x2, y2])
    boxes = np.array(results)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    del detector
    return results

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    face_det_results = face_detect(frames)
    for i, m in enumerate(mels):
        idx = i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()
        face = cv2.resize(face, (args.img_size, args.img_size))
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        if len(img_batch) >= 128:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch

def load_model(path):
    print(f"Load checkpoint from: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model = Wav2Lip()
            s = checkpoint["state_dict"]
            new_s = {k.replace('module.', ''): v for k, v in s.items()}
            model.load_state_dict(new_s)
            model = model.to(device)
            return model.eval()
        else:
            # Nếu là TorchScript, load trực tiếp
            model = torch.jit.load(path, map_location=device)
            return model.eval()
    except Exception as e:
        print("Error loading checkpoint:", e)
        raise

def main():
    # Read frames
    if args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = 25
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                break
            full_frames.append(frame)
        video_stream.release()
    print(f"Number of frames: {len(full_frames)}")

    # Audio
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)
    mel_step_size = 16
    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    print(f"Length of mel chunks: {len(mel_chunks)}")
    full_frames = full_frames[:len(mel_chunks)]
    gen = datagen(full_frames.copy(), mel_chunks)
    # Model
    model = load_model(args.checkpoint_path)
    print("Model loaded")
    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp_result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/128)))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            out.write(f)
    out.release()
    # Mux audio and video using moviepy
    from moviepy.editor import VideoFileClip, AudioFileClip
    video_clip = VideoFileClip('temp_result.avi')
    audio_clip = AudioFileClip(args.audio)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(args.outfile, codec='libx264', audio_codec='aac')
    print(f"Output saved to {args.outfile}")

if __name__ == '__main__':
    main()