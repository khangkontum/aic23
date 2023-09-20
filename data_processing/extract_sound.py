import os
from moviepy.editor import *
from tqdm import tqdm
from multiprocessing import Pool
from subprocess import call

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/6.0_1/bin/ffmpeg"


# Input folder containing videos
input_folder = "./raw/vids/video"

# Output folder for extracted frames
output_folder = "./raw/sounds/"


# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def extract_sounds(video_path):
    filename = video_path.replace(".mp4", ".mp3").split("/")[-1]
    filename = os.path.join(output_folder, filename)

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(filename)


if __name__ == "__main__":
    # List all files in the input folder
    videos = []
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".mp4") or filename.endswith(
            ".avi"
        ):
            video_path = os.path.join(
                input_folder, filename
            )
            videos.append(video_path )
            # extract_sounds(video_path)


    pool = Pool(processes=4)
    pool.map(extract_sounds, videos , 1) # Ensure the chunk size is 1
    pool.close()
    pool.join()