import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from moviepy.editor import ImageSequenceClip, AudioFileClip


# === CONFIGURATION ===
FRAME_RATE = 30
AUDIO_FILE = input("Enter the path to the audio file (wav): ")
OUTPUT_VIDEO = "musicvideo.mp4"
TEMP_DIR = "frames"
os.makedirs(TEMP_DIR, exist_ok=True)



if not os.path.exists(AUDIO_FILE):
    raise FileNotFoundError(f"File not found: {AUDIO_FILE}")


# === LOAD AUDIO ===
audio = AudioSegment.from_file(AUDIO_FILE).set_channels(1)
samples = np.array(audio.get_array_of_samples())
sample_rate = audio.frame_rate
samples_per_frame = int(sample_rate / FRAME_RATE)
total_frames = int(len(samples) / samples_per_frame)

# === GENERATE FRAME IMAGES ===
print("Generating visualization frames...")

for i in range(total_frames):
    start = i * samples_per_frame
    end = start + samples_per_frame
    frame = samples[start:end]

    # Plot waveform
    plt.figure(figsize=(10, 4))
    plt.plot(frame, color='mediumslateblue')
    plt.ylim(-2**15, 2**15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{TEMP_DIR}/frame_{i:05d}.png", dpi=100)
    plt.close()

# === CREATE VIDEO ===
print("Creating video...")

clip = ImageSequenceClip(TEMP_DIR, fps=FRAME_RATE)
clip = clip.set_audio(AudioFileClip(AUDIO_FILE))
clip.write_videofile(OUTPUT_VIDEO, codec="libx264", audio_codec="aac", ffmpeg_params=["-loglevel", "quiet"])

# Cleanup (optional)
# import shutil; shutil.rmtree(TEMP_DIR)

print(f"✅ Done! Saved to {OUTPUT_VIDEO}")
