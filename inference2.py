"""
Combine video and audio into an mp4 file using moviepy (no ffmpeg command-line required).
Usage example:
	python inference2.py --video temp/result.avi --audio sample_audio.wav --output results/result_voice_moviepy.mp4
"""
import argparse
from moviepy.editor import VideoFileClip, AudioFileClip
import os

def mux_video_audio(video_path, audio_path, output_path):
	# Load video and audio clips
	video_clip = VideoFileClip(video_path)
	audio_clip = AudioFileClip(audio_path)
	# Set audio to video
	final_clip = video_clip.set_audio(audio_clip)
	# Write the result
	final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Mux video and audio using moviepy.")
	parser.add_argument('--video', type=str, required=True, help='Path to input video file (no audio)')
	parser.add_argument('--audio', type=str, required=True, help='Path to input audio file (wav or other)')
	parser.add_argument('--output', type=str, required=True, help='Path to output mp4 file')
	args = parser.parse_args()

	if not os.path.isfile(args.video):
		raise FileNotFoundError(f"Video file not found: {args.video}")
	if not os.path.isfile(args.audio):
		raise FileNotFoundError(f"Audio file not found: {args.audio}")

	mux_video_audio(args.video, args.audio, args.output)
	print(f"Muxed video and audio saved to {args.output}")
