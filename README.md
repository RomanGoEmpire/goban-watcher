# Goban Watcher

This project aims to automatically record Go games with the help of cameras and computer vision.

### Prerequisites
>
> [!NOTE]
> Using KataGo is optional and can be skipped!
> Continue with "Starting the application"

It can happen that multiple moves are picked up at the same time and it is unclear in which order they where played.
To combat this issue [KataGo](https://github.com/lightvector/KataGo) is be used to select the "best" possible variation.
To be able to use KataGo the following steps **must** be taken:

1. Install [KataGo](https://github.com/lightvector/KataGo). Validate if `katago` command works.
2. Install a KataGo model from [here](https://katagotraining.org/networks/). During development a model with the architecture of `b28c512nbt` was used.
    1. Rename the model file to `b28c512nbt.bin.gz`
    2. Move it into `katago/models/`.
    3. Validate from root folder that this command works:

      ```shell
      katago analysis -config "katago/configs/analysis_example.cfg" -model "katago/models/b28c512nbt.bin.gz"
      ```

### Starting the application

The easiest way to get up and running is to use [uv](https://docs.astral.sh/uv/).
Start the application with:

```sh
uv run main.py
```

Use `uv run main.py --help` to see all available flags.

### Usage

Goban Watcher supports two modes:

1. **Live Camera Mode**: Real-time recording using a camera or webcam
2. **Video Mode**: Process pre-recorded videos from the `videos/` folder

The camera/video should be positioned with a top-down view or slightly from the side.

> [!WARNING]
> The application is still very bare bones and not very user friendly.
> So for now I try to document it as much as possible here.

## Mode 1: Live Camera Recording (Default)

Run the application with:

```sh
uv run main.py
```

#### Setup Corners

This is the first and most manual part of the setup. The application does not have an automatic board detection yet.
So setting up the corners requires manual work.

> [!Warning]
> Since the setup of the corners is manual and happens at the start of the recording any movement of the board during the game will break the recording!

![start](/docs/images/001-start.jpeg)

Two windows shoud appear. One of which is called "Default". This window is used to setup the corners.
Press 1-4 on the keyboard to select one of the corners. It will follow the mouse cursor and it can release it by pressing
the same number or another number.

The "Transformed" window can be used to check the aligment. The sgf will have the same orientation as the window displays. The recommendation is to start with the top left corner (from blacks perspective) as corner 1 and go clockwise for each corner. The top right corner should be 2 from the view of the black player.

![setup-corners](/docs/images/002-setup-corners.jpeg)
Example of the finished aligment on "Default" window.

![setup-corners](/docs/images/003-setup-corners-result.jpeg)
Example of the finished aligment on "Transformed" window. The stones do not have to be perfect on the intersection. The actual images that are used to detect the stones are larger than the green squares visible for aligment.

## Mode 2: Video Analysis

To process pre-recorded videos instead of live camera feed:

### Setup

1. Create a `videos/` folder in the project root (if it doesn't exist)
2. Place your video files in the `videos/` folder
3. Supported formats: `.mp4`, `.avi`, `.mov`
4. Videos will be processed in alphabetical/numerical order

### Running Video Mode

```sh
uv run main.py --video-mode
```

Optional flags:

```sh
# Specify custom video folder path (default: videos/)
uv run main.py --video-mode --video-path /path/to/your/videos

# Adjust frame skip for speed (default: 10 = process every 10th frame)
# Lower = slower but more accurate, Higher = faster but may miss moves
uv run main.py --video-mode --frame-skip 5   # More accurate
uv run main.py --video-mode --frame-skip 15  # Faster

# Adjust frame stability threshold (default: 3 for video mode, 15 for camera)
uv run main.py --video-mode --identical-frames 5

# Enable KataGo for move reconstruction
uv run main.py --video-mode --enable-katago

# Use saved corners (skip manual setup for each video)
uv run main.py --video-mode --use-saved-corners

# Combine multiple options for optimal speed
uv run main.py --video-mode --video-path ~/my_go_videos --frame-skip 15 --identical-frames 3 --use-saved-corners
```

### Performance Optimization

**Frame Skip** (`--frame-skip`): Process every Nth frame instead of every frame

- Default: 10 (10× faster, processes ~3 frames/second for 30fps video)
- Lower values (5-7): More accurate move detection
- Higher values (15-20): Faster processing, may miss quick moves

**Identical Frames** (`--identical-frames`): Frames that must match before detecting a move

- Default for video mode: 3 (with frame-skip=10, this is ~1 second stability)
- Default for camera mode: 15 (0.5 seconds at 30fps)

**Speed Estimate for 1-hour video:**

- Default settings (skip=10, identical=3): **~12-18 minutes** (10-15× faster than real-time)
- Conservative (skip=5, identical=5): **~25-35 minutes** (5-7× faster)
- Aggressive (skip=15, identical=2): **~8-12 minutes** (15-20× faster)

```

### Video Processing Features

#### Automatic Board Position Change Detection

The system automatically detects when the camera position changes during a video by:

- Monitoring corner positions
- Comparing frame histograms
- Checking every 30 frames for significant changes

When a position change is detected, the video **automatically pauses** and displays a warning.

#### Manual Controls During Video Processing

- **`p`**: Pause/Resume video playback
- **`c`**: Recalibrate corners (when paused or after automatic pause)
- **`r`**: Resume processing after recalibration
- **`q` or `ESC`**: Stop processing current video and move to next

#### Corner Setup for Videos

For each video, you'll need to set up corners:

1. The first frame will be displayed
2. Press keys `1-4` to position the four corners of the board
3. Press `s` to save corners (optional, for reuse with `--use-saved-corners`)
4. Press `Enter` to start processing

If using `--use-saved-corners`, the system will attempt to load previously saved corners from `backup/corners.json`.

#### Output

All videos are processed sequentially and combined into a **single SGF file** in the `recording/` folder with the format:

```

combined_videos_<timestamp>.sgf

```

For example:

- Videos: `videos/1.mp4`, `videos/2.mp4`, `videos/3.mp4`
- Output: `recording/combined_videos_22042026163045.sgf`

The SGF file contains all moves from all videos in the order they were processed (alphabetically/numerically by filename).

### Video Processing Workflow

1. System scans `videos/` folder for supported video files
2. Creates a single shared game state and SGF file
3. For each video (in alphabetical/numerical order):
   - Load or set up board corners
   - Process frames sequentially
   - Detect stone placements
   - Monitor for camera position changes
   - Add moves to the shared SGF file
4. Move to next video automatically
5. Save the combined SGF file after all videos are processed

### Tips for Video Recording

- **Stable camera**: Mount camera to avoid movement during recording
- **Good lighting**: Ensure consistent lighting throughout the video
- **Clear view**: Top-down or slightly angled view works best
- **Minimize hand interference**: Keep hands out of frame when not placing stones
- **Pause between moves**: Allow 0.5-1 second between moves for detection

#### Saving the corners

Press `s` to save the corner placement. There should be a log message about a successful save in the terminal. The saved corners can be reused by passing the `--use-saved-corners` param after the start commmand. The program will try to read the `backup/corners.json`. A single stone should fit into one of the small squares.

#### Resetting the corners

Press `r` to reset the position of all corners to the default position.

#### Starting Recording

Press `enter` or `return` to start recording. When the real board and digital board are side by side it is recording.
Each move now should be recorded and also the sgf, located in `/recodings`, should be update after each change.
If more than then 3 stones are recognized at the same time it will "edit" the sgf and add all at the same time (if KataGo is disabled).

![recording](/docs/images/004-recording.jpeg)

To stop the recording press `q` or `esc` which will save the game one more time and exit the application.

## Research

- Initial development was done by Roman Gerloff for his bachelor thesis: [Go Game Capture and Reconstruction of Missing Moves Using Deep Learning Techniques](https://ieeexplore.ieee.org/abstract/document/11114356)
