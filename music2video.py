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
import multiprocessing
from multiprocessing import Pool, cpu_count
import gc


# === CONFIG ===
FRAME_RATE = 24


# === VISUAL HELPERS ===

def rainbow_line(ax, y_vals):
    x_vals = np.linspace(0, 1, len(y_vals))
    for i in range(len(y_vals) - 1):
        c = cm.hsv(i / len(y_vals))
        ax.plot(x_vals[i:i+2], y_vals[i:i+2], color=c, linewidth=2.5)

def draw_starfield(ax, star_count=300):
    np.random.seed()
    x = np.random.rand(star_count)
    y = np.random.rand(star_count)
    sizes = np.random.rand(star_count) * 2
    ax.scatter(x, y, s=sizes, c='white', alpha=0.3, zorder=0, transform=ax.transAxes)


# === FRAME GENERATORS ===

def generate_wave_frame(i, samples, spf, title, author, total_frames):
    start = i * spf
    end = start + spf
    frame = samples[start:end]
    norm_frame = frame / (2**15)

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    draw_starfield(ax)

    zoom = 1 + 0.003 * np.sin(i / 20)
    ax.set_xlim(0, 1 / zoom)
    ax.set_ylim(-1.1 / zoom, 1.1 / zoom)

    rainbow_line(ax, norm_frame)

    glow = [path_effects.Stroke(linewidth=4, foreground='black'), path_effects.Normal()]
    fade_duration = FRAME_RATE
    alpha = 1.0
    if i < fade_duration:
        alpha = i / fade_duration
    elif i > total_frames - fade_duration:
        alpha = (total_frames - i) / fade_duration

    ax.text(0.01, 0.06, title, fontsize=18, color='white', ha='left', va='bottom',
            alpha=alpha, transform=ax.transAxes).set_path_effects(glow)
    ax.text(0.01, 0.02, author, fontsize=13, color='lightgrey', ha='left', va='bottom',
            alpha=alpha, transform=ax.transAxes).set_path_effects(glow)

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    canvas = FigureCanvas(fig)
    canvas.draw()

    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(canvas.get_width_height()[::-1] + (4,))
    img_rgb = Image.fromarray(img[:, :, :3])  # drop alpha
    img_rgb.save(f"frames/frame_{i:05d}.png")

    plt.close('all')
    plt.close(fig)
    plt.clf()  # Clear the figure (reset the state)
    del fig, ax  # Explicitly delete figure and axes
    gc.collect()  # Trigger garbage collection to release memory

def generate_circle_frame(i, samples, spf, title, author, total_frames):
    start = i * spf
    end = start + spf
    frame = samples[start:end]
    norm_frame = frame / (2**15)

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
    ax.text(0.0, -0.85, title, fontsize=18, color='white', ha='center', va='top',
            alpha=alpha).set_path_effects(glow)
    ax.text(0.0, -0.9, author, fontsize=13, color='lightgrey', ha='center', va='top',
            alpha=alpha).set_path_effects(glow)

    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    canvas = FigureCanvas(fig)
    canvas.draw()

    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(canvas.get_width_height()[::-1] + (4,))
    img_rgb = Image.fromarray(img[:, :, :3])
    img_rgb.save(f"frames/frame_{i:05d}.png")

    plt.close('all')
    plt.close(fig)
    plt.clf()  # Clear the figure (reset the state)

    del fig, ax  # Explicitly delete figure and axes
    gc.collect()  # Trigger garbage collection to release memory


# === MULTIPROCESS WRAPPER ===

def generate_and_save_frame(args):
    generate_func, i, samples, spf, title, author, total_frames = args
    generate_func(i, samples, spf, title, author, total_frames)


    # Explicit cleanup
    gc.collect()  # Manually trigger garbage collection


# === MAIN ===

def main():
    if not os.path.exists("frames"):
        os.makedirs("frames")

    AUDIO_FILE = input("Enter path to WAV audio: ")
    AUTHOR_NAME = input("Enter your name: ")
    TITLE_NAME = input("Enter the title: ")
    VISUAL_OPTION = int(input("Enter visual mode (0 = wave, 1 = circle): "))
    OUTPUT_VIDEO = "visualization.mp4"

    with wave.open(AUDIO_FILE, 'rb') as wav:
        n_channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        audio_data = wav.readframes(n_frames)

    samples = np.frombuffer(audio_data, dtype=np.int16)
    if n_channels == 2:
        samples = samples[::2]  # stereo to mono

    spf = int(sample_rate / FRAME_RATE)
    total_frames = len(samples) // spf

    generate_func = generate_wave_frame if VISUAL_OPTION == 0 else generate_circle_frame

    # print("Generating frames using max 4 processes...")  # processes=4
    with multiprocessing.Pool(cpu_count()-1) as pool:
        args_list = [(generate_func, i, samples, spf, TITLE_NAME, AUTHOR_NAME, total_frames) for i in range(total_frames)]
        list(tqdm(pool.imap_unordered(generate_and_save_frame, args_list), total=total_frames, unit="frame"))




    print("Composing video with MoviePy...")
    frame_files = [f"frames/frame_{i:05d}.png" for i in range(total_frames)]
    clip = ImageSequenceClip(frame_files, fps=FRAME_RATE)
    clip = clip.set_audio(AudioFileClip(AUDIO_FILE))
    clip.write_videofile(OUTPUT_VIDEO, codec="libx264", audio_codec="aac")

    print(f"Done. Video saved to: {OUTPUT_VIDEO}")


    #clean
    print("Cleaning up...")
    import shutil
    shutil.rmtree("frames")

    print("Press enter to exit...")
    input()



if __name__ == "__main__":
    main()
