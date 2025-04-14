import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pydub import AudioSegment
from moviepy.editor import ImageSequenceClip, AudioFileClip
import matplotlib.patheffects as path_effects
from tqdm import tqdm




# === CONFIGURATION ===
FRAME_RATE = 30
AUDIO_FILE = input("Enter the path to the audio file (wav): ")
AUTHOR = input("Enter your name: ")
TITLE = input("Enter the title: ")
OUTPUT_VIDEO = "visualization.mp4"
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

# === HELPER FUNCTIONS ===
def rainbow_line(ax, y_vals):
    x_vals = np.linspace(0, 1, len(y_vals))
    for i in range(len(y_vals) - 1):
        c = cm.hsv(i / len(y_vals))
        ax.plot(x_vals[i:i+2], y_vals[i:i+2], color=c, linewidth=2.5)

def draw_starfield(ax, star_count=300):
    np.random.seed()  # Different stars each frame
    x = np.random.rand(star_count)
    y = np.random.rand(star_count)
    sizes = np.random.rand(star_count) * 2
    ax.scatter(x, y, s=sizes, c='white', alpha=0.3, zorder=0, transform=ax.transAxes)

# === GENERATE FRAMES ===
print("🪐 Generating galaxy frames...")

for i in tqdm(range(total_frames), desc="🌌 Generating Frames", unit="frame", dynamic_ncols=True):
    start = i * samples_per_frame
    end = start + samples_per_frame
    frame = samples[start:end]
    norm_frame = frame / (2**15)

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    draw_starfield(ax)

    # Apply small motion to make it dynamic
    zoom = 1 + 0.003 * np.sin(i / 20)
    ax.set_xlim(0, 1 / zoom)
    ax.set_ylim(-1.1 / zoom, 1.1 / zoom)

    rainbow_line(ax, norm_frame)

    # === TITLE & AUTHOR with Glow Effect ===
    glow = [path_effects.Stroke(linewidth=4, foreground='black'), path_effects.Normal()]

    # Fade in/out: fade in for first 1 sec, fade out for last 1 sec
    fade_duration = FRAME_RATE  # 1 second
    alpha = 1.0
    if i < fade_duration:
        alpha = i / fade_duration
    elif i > total_frames - fade_duration:
        alpha = (total_frames - i) / fade_duration

    # Title
    txt1 = ax.text(0.01, 0.06, f"{TITLE}", fontsize=18, color='white',
                ha='left', va='bottom', alpha=alpha, transform=ax.transAxes)
    txt1.set_path_effects(glow)

    # Author
    txt2 = ax.text(0.01, 0.02, f"{AUTHOR}", fontsize=13, color='lightgrey',
                ha='left', va='bottom', alpha=alpha, transform=ax.transAxes)
    txt2.set_path_effects(glow)


    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f"{TEMP_DIR}/frame_{i:05d}.png", facecolor='black')
    plt.close()
    


# === CREATE VIDEO ===
print("🎬 Composing video with audio...")

clip = ImageSequenceClip(TEMP_DIR, fps=FRAME_RATE)
clip = clip.set_audio(AudioFileClip(AUDIO_FILE))
clip.write_videofile(OUTPUT_VIDEO, codec="libx264", audio_codec="aac", ffmpeg_params=["-loglevel", "quiet"])

# Optional: Clean up frames
# import shutil; shutil.rmtree(TEMP_DIR)

print(f"✅ Done! Saved to {OUTPUT_VIDEO}")
