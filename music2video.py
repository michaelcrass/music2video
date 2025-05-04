import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patheffects as path_effects
from moviepy.editor import ImageSequenceClip, AudioFileClip
from tqdm import tqdm
import os
from PIL import Image
import wave
import gc
from multiprocessing import Pool, cpu_count
import multiprocessing

# === AUDIO LOADING ===

def load_audio_wav(file_path):
    with wave.open(file_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    if n_channels == 2:
        audio_array = audio_array[::2]

    return audio_array, framerate

# === HELPER FUNCTIONS ===

def draw_starfield(ax, star_count=300):
    np.random.seed()
    x = np.random.rand(star_count)
    y = np.random.rand(star_count)
    sizes = np.random.rand(star_count) * 2
    ax.scatter(x, y, s=sizes, c='white', alpha=0.3, zorder=0, transform=ax.transAxes)

def rainbow_line(ax, y_vals):
    x_vals = np.linspace(0, 1, len(y_vals))
    for i in range(len(y_vals) - 1):
        c = cm.hsv(i / len(y_vals))
        ax.plot(x_vals[i:i+2], y_vals[i:i+2], color=c, linewidth=2.5)

# === SHARED GLOBAL STATE FOR MULTIPROCESSING ===

samples = None
samples_per_frame = None
FRAME_RATE = 30
AUTHOR = ""
TITLE = ""
VISUAL_OPTION = 0
total_frames = 0

def init_worker(audio_data, spf, author, title, total):
    global samples, samples_per_frame, AUTHOR, TITLE, total_frames
    samples = audio_data
    samples_per_frame = spf
    AUTHOR = author
    TITLE = title
    total_frames = total

def generate_and_save_frame(args):
    generate_func, frame_idx = args
    image_array = generate_func(frame_idx)
    image = Image.fromarray(image_array)
    image.save(f"frames/frame_{frame_idx:05d}.png")
    del image, image_array
    gc.collect()

def generate_wave_frame(i):
    start = i * samples_per_frame
    end = start + samples_per_frame
    frame = samples[start:end]
    norm_frame = frame / (2 ** 15)

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    draw_starfield(ax)

    zoom = 1 + 0.003 * np.sin(i / 20)
    ax.set_xlim(0, 1 / zoom)
    ax.set_ylim(-1.1 / zoom, 1.1 / zoom)
    rainbow_line(ax, norm_frame)

    fade_duration = FRAME_RATE
    alpha = 1.0
    if i < fade_duration:
        alpha = i / fade_duration
    elif i > total_frames - fade_duration:
        alpha = (total_frames - i) / fade_duration

    glow = [path_effects.Stroke(linewidth=4, foreground='black'), path_effects.Normal()]
    txt1 = ax.text(0.01, 0.06, f"{TITLE}", fontsize=18, color='white',
                   ha='left', va='bottom', alpha=alpha, transform=ax.transAxes)
    txt1.set_path_effects(glow)
    txt2 = ax.text(0.01, 0.02, f"{AUTHOR}", fontsize=13, color='lightgrey',
                   ha='left', va='bottom', alpha=alpha, transform=ax.transAxes)
    txt2.set_path_effects(glow)

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # Drop alpha
    plt.close(fig)
    plt.close('all')
    plt.close()
    del fig, ax, canvas, frame, norm_frame
    gc.collect()
    return img

def generate_circle_frame(i):
    start = i * samples_per_frame
    end = start + samples_per_frame
    frame = samples[start:end]
    norm_frame = frame / (2 ** 15)

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    draw_starfield(ax)

    N = len(norm_frame)
    theta = np.linspace(0, 2 * np.pi, N)
    radius = 0.3 + 0.2 * norm_frame
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    for j in range(N - 1):
        c = cm.hsv(j / N)
        ax.plot(x[j:j+2], y[j:j+2], color=c, linewidth=2.5)

    fade_duration = FRAME_RATE
    alpha = 1.0
    if i < fade_duration:
        alpha = i / fade_duration
    elif i > total_frames - fade_duration:
        alpha = (total_frames - i) / fade_duration

    glow = [path_effects.Stroke(linewidth=4, foreground='black'), path_effects.Normal()]
    ax.text(0.0, -0.85, f"{TITLE}", fontsize=18, color='white',
            ha='center', va='top', alpha=alpha, path_effects=glow)
    ax.text(0.0, -0.9, f"{AUTHOR}", fontsize=13, color='lightgrey',
            ha='center', va='top', alpha=alpha, path_effects=glow)

    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]
    plt.close(fig)
    plt.close("all")
    plt.close()
    del fig, ax, canvas, frame, norm_frame
    gc.collect()
    return img

# === MAIN ===

def main():
    global samples, samples_per_frame, total_frames

    if not os.path.exists("frames"):
        os.makedirs("frames")

    AUDIO_FILE = input("Enter path to WAV file: ")
    AUTHOR_NAME = input("Enter your name: ")
    TITLE_NAME = input("Enter the title: ")
    VISUAL_OPTION = int(input("Enter visual option (0 = wave, 1 = circle): "))
    OUTPUT_VIDEO = "visualization.mp4"

    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(f"File not found: {AUDIO_FILE}")

    samples, sample_rate = load_audio_wav(AUDIO_FILE)
    samples_per_frame = int(sample_rate / FRAME_RATE)
    total_frames = int(len(samples) / samples_per_frame)

    generate_frame = generate_wave_frame if VISUAL_OPTION == 0 else generate_circle_frame

    print("Generating frames...")  #processes=4
    multiprocessing.set_start_method('spawn')




    with Pool(int(cpu_count()-1), initializer=init_worker,
              initargs=(samples, samples_per_frame, AUTHOR_NAME, TITLE_NAME, total_frames)) as pool:
        list(tqdm(pool.imap(generate_and_save_frame,
                            [(generate_frame, i) for i in range(total_frames)]),
                  total=total_frames, desc="Generating Frames"))





    print("Composing video with audio...")
    frame_files = [f"frames/frame_{i:05d}.png" for i in range(total_frames)]
    clip = ImageSequenceClip(frame_files, fps=FRAME_RATE)
    clip = clip.set_audio(AudioFileClip(AUDIO_FILE))
    clip.write_videofile(OUTPUT_VIDEO, codec="libx264", audio_codec="aac", ffmpeg_params=["-loglevel", "quiet"])

    print(f"Done! Saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()




