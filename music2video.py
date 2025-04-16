import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patheffects as path_effects
from pydub import AudioSegment
from moviepy.editor import ImageSequenceClip, AudioFileClip
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

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

# These variables will be shared via global scope into the pool
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

def generate_wave_frame(i):
    start = i * samples_per_frame
    end = start + samples_per_frame
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

    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def generate_circle_frame(i):
    start = i * samples_per_frame
    end = start + samples_per_frame
    frame = samples[start:end]
    norm_frame = frame / (2**15)

    # Create a 1920x1080 figure
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    draw_starfield(ax)  # Your starfield function, assumed defined elsewhere

    # Create circular waveform
    N = len(norm_frame)
    theta = np.linspace(0, 2 * np.pi, N)
    radius = 0.3 + 0.2 * norm_frame  # Base radius + waveform

    # Convert polar to cartesian
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Plot each segment in rainbow color
    for j in range(N - 1):
        c = cm.hsv(j / N)
        ax.plot(x[j:j+2], y[j:j+2], color=c, linewidth=2.5)

    # Fading alpha at beginning and end
    fade_duration = FRAME_RATE
    alpha = 1.0
    if i < fade_duration:
        alpha = i / fade_duration
    elif i > total_frames - fade_duration:
        alpha = (total_frames - i) / fade_duration

    # Add soft glow text
    glow = [path_effects.Stroke(linewidth=4, foreground='black'), path_effects.Normal()]
    txt1 = ax.text(0.0, -1.1, f"{TITLE}", fontsize=18, color='white',
                   ha='center', va='top', alpha=alpha, path_effects=glow)
    txt2 = ax.text(0.0, -1.25, f"{AUTHOR}", fontsize=13, color='lightgrey',
                   ha='center', va='top', alpha=alpha, path_effects=glow)

    # Adjust view and layout
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Render the figure to a NumPy array (image)
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def main():
    global samples, samples_per_frame, total_frames

    # === GET USER INPUT ===
    AUDIO_FILE = input("Enter the path to the audio file (wav): ")
    AUTHOR_NAME = input("Enter your name: ")
    TITLE_NAME = input("Enter the title: ")
    VISUAL_OPTION = int(input("Enter the visual option (0 for wave, 1 for circle): "))
    OUTPUT_VIDEO = "visualization.mp4"

    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(f"File not found: {AUDIO_FILE}")

    # === LOAD AUDIO ===
    audio = AudioSegment.from_file(AUDIO_FILE).set_channels(1)
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    samples_per_frame = int(sample_rate / FRAME_RATE)
    total_frames = int(len(samples) / samples_per_frame)

    # === PARALLEL FRAME GENERATION ===
    print("Generating galaxy frames in parallel...")


    if VISUAL_OPTION == 0:
        generate_frame = generate_wave_frame
    else:
        generate_frame = generate_circle_frame
    with Pool(cpu_count(), initializer=init_worker,
              initargs=(samples, samples_per_frame, AUTHOR_NAME, TITLE_NAME, total_frames)) as pool:
        frames = list(tqdm(pool.imap(generate_frame, range(total_frames)),
                           total=total_frames, desc="Generating Frames", unit="frame"))

    # === CREATE VIDEO ===
    print("Composing video with audio...")
    clip = ImageSequenceClip(frames, fps=FRAME_RATE)
    clip = clip.set_audio(AudioFileClip(AUDIO_FILE))
    clip.write_videofile(OUTPUT_VIDEO, codec="libx264", audio_codec="aac", ffmpeg_params=["-loglevel", "quiet"])

    print(f"Done! Saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()