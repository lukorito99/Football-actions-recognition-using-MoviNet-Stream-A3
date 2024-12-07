import av
import cv2
from matplotlib import patches
import numpy as np

from typing import Tuple, List, Dict
import json
from tqdm import tqdm
import tensorflow as tf
from sortedcontainers import SortedDict
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from functools import partial

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from matplotlib.gridspec import GridSpec
import os
from scenedetect import VideoManager, SceneManager, ContentDetector
class SortedDictManager(BaseManager):
    pass

SortedDictManager.register('SortedDict', SortedDict)

def create_action_collage(start_frame_path: str, middle_frame_path: str, end_frame_path: str, collages_dir: str, action_index: int):
    """
    Creates and saves a horizontal collage of three frames showing the progression of an action.
    
    Args:
        start_frame_path: Path to the starting frame.
        middle_frame_path: Path to the middle frame.
        end_frame_path: Path to the end frame.
        collages_dir: Directory where collages will be saved.
        action_index: Index of the action for clear identification.
    """
    os.makedirs(collages_dir, exist_ok=True)

    try:
        # Open images
        start_img = Image.open(start_frame_path)
        middle_img = Image.open(middle_frame_path)
        end_img = Image.open(end_frame_path)
        
        # Get dimensions
        widths, heights = zip(*(i.size for i in [start_img, middle_img, end_img]))
        total_width = sum(widths)
        max_height = max(heights)
        
        # Create new image with white background
        collage = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        
        # Paste images side-by-side
        x_offset = 0
        for img in [start_img, middle_img, end_img]:
            # Maintain aspect ratio while ensuring consistent height
            aspect_ratio = img.size[0] / img.size[1]
            new_height = max_height
            new_width = int(aspect_ratio * new_height)
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            collage.paste(img_resized, (x_offset, 0))
            x_offset += new_width
        
        # Add labels to indicate the progression
        draw = ImageDraw.Draw(collage)
        font = ImageFont.load_default()
        labels = ["Start", "Middle", "End"]
        positions = [0, total_width // 3, 2 * total_width // 3]
        
        for label, x_pos in zip(labels, positions):
            # Calculate text size
            text_bbox = font.getbbox(label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position text
            x = x_pos + (total_width // 3 - text_width) // 2
            y = 10  # Position near the top
            
            # Add a semi-transparent background for the label
            draw.rectangle([x - 5, y - 5, x + text_width + 5, y + text_height + 5], fill=(0, 0, 0, 128))
            draw.text((x, y), label, font=font, fill=(255, 255, 255))
        
        # Save the collage
        output_path = os.path.join(collages_dir, f"action_{action_index:02d}_progression_collage.jpg")
        collage.save(output_path, 'JPEG', quality=95, optimize=True, subsampling=0)
        print(f"Saved collage: {output_path}")
    
    except Exception as e:
        print(f"Error creating collage for action {action_index}: {e}")


def save_frame(frame: np.ndarray, output_path: str, size: Tuple[int, int] = (1920, 1080)):
    """
    Save a frame in high resolution, optimized for later computer vision tasks.
    Maintains aspect ratio while ensuring minimum dimensions for accurate object detection.
    
    Args:
        frame: Input frame in RGB/BGR format
        output_path: Where to save the frame
        size: Target size (width, height) - defaults to 1080p
    """
    try:
        # Ensure minimum dimensions for good object detection
        MIN_WIDTH = 1280  # minimum width for reliable small object detection
        MIN_HEIGHT = 720  # minimum height for reliable small object detection
        
        h, w = frame.shape[:2]
        aspect = w/h
        
        # Calculate new dimensions maintaining aspect ratio
        if aspect > size[0]/size[1]:  # Width limited
            new_w = max(MIN_WIDTH, min(size[0], w))  # Don't upscale beyond original
            new_h = int(new_w/aspect)
            # Ensure minimum height
            if new_h < MIN_HEIGHT:
                new_h = MIN_HEIGHT
                new_w = int(new_h * aspect)
        else:  # Height limited
            new_h = max(MIN_HEIGHT, min(size[1], h))  # Don't upscale beyond original
            new_w = int(new_h * aspect)
            # Ensure minimum width
            if new_w < MIN_WIDTH:
                new_w = MIN_WIDTH
                new_h = int(new_w/aspect)
        
        # Use high-quality interpolation for resizing
        if len(frame.shape) == 2:  # Grayscale
            frame = np.stack([frame] * 3, axis=-1)
        
        # Use PIL for high-quality resizing
        
        pil_img = Image.fromarray(frame)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Save with high quality
        pil_img.save(
            output_path,
            'JPEG',
            quality=95,  # High quality for detail preservation
            optimize=True,  # Optimize encoding
            subsampling=0  # No chroma subsampling for better quality
        )
        
    except Exception as e:
        print(f"Error in high-quality save, attempting fallback: {e}")
        try:
            # Fallback to OpenCV if PIL fails
           
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                output_path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            )
        except Exception as e2:
            print(f"Both save methods failed: {e2}")



from multiprocessing import Pool, cpu_count



def process_scene_worker(
    model_path: str,
    video_path: str,
    scene_info: Tuple[int, int, str],
    resolution: Tuple[int, int],
    frames_per_window: int,
    confidence_threshold: float,
    classes: List[str]
) -> Dict:
    """
    Enhanced scene worker with original confidence recording.
    
    Args:
        model_path: Path to the TFLite model.
        video_path: Path to the video file.
        scene_info: Tuple (start_frame, end_frame, tag) with scene details.
        resolution: Tuple (width, height) for resizing input frames.
        frames_per_window: Number of frames per inference window.
        confidence_threshold: Confidence threshold for action detection.
        classes: List of action class names.

    Returns:
        Dictionary of detected actions with original confidence recorded.
    """
    try:
        # Initialize TensorFlow Lite interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=None)
        interpreter.allocate_tensors()
        runner = interpreter.get_signature_runner()

        init_states = {
            name: tf.zeros(x['shape'], dtype=x['dtype'])
            for name, x in runner.get_input_details().items()
            if name != 'image'
        }
        states = init_states

        # Scene parameters
        scene_start, scene_end = scene_info 
        scene_length = scene_end - scene_start

        # Compute dynamic stride
        stride = max(scene_length // 50, 5)
        print(f'Current stride: {stride}, scene length: {scene_length}')
        scene_actions = {}
        frame_window = []
        prob_history = []
        current_action = None
        action_start_time = None
       
        
        confidence_history = []  # To track per-frame confidences for the current action
        original_confidence = None  # Store the original confidence when action starts
        decay_rate = 0.9  # Confidence decay rate
        smoothed_prob_history_size = 3  # Temporal smoothing window intially it was 5

        # Hysteresis thresholds
        start_threshold = confidence_threshold
        continue_threshold = confidence_threshold * 0.6 #intially it was 0.8,

        # Extend scene bounds slightly for buffer
        padding_frames = int(frames_per_window / 2)
        adjusted_scene_end = min(scene_end + padding_frames, scene_end)

        # Open video container
        container = av.open(video_path)
        stream = container.streams.video[0]
        frame_counter = scene_start

        for frame in container.decode(video=0):
            if frame_counter >= adjusted_scene_end:
                break

            # Prepare frame for processing
            frame_np = frame.to_ndarray(format='rgb24')
            frame_resized = cv2.resize(frame_np, resolution)
            frame_window.append(frame_resized)

            if len(frame_window) == frames_per_window:
                # Model inference
                processed_window = np.stack([f.astype(np.float32) / 255.0 for f in frame_window])
                outputs = runner(**states, image=processed_window)
                logits = outputs.pop('logits')[0]
                probabilities = tf.nn.softmax(logits).numpy()

                # Smooth probabilities
                prob_history.append(probabilities)
                if len(prob_history) > smoothed_prob_history_size:
                    prob_history.pop(0)
                smoothed_probs = np.mean(prob_history, axis=0)

                # Get max probability and detected action
                max_prob = np.max(smoothed_probs)
                action_id = np.argmax(smoothed_probs)
                detected_action = classes[action_id]
                current_time = float(frame.pts * stream.time_base)

                # Track confidence for the current frame
                if current_action is not None:
                    confidence_history.append(max_prob)

                # Hysteresis logic for action detection
                if current_action is None and max_prob >= start_threshold:
                    # Start a new action
                    current_action = detected_action
                    action_start_time = current_time
                    original_confidence = max_prob  # Record the original confidence
                    confidence_history = [max_prob]  # Initialize confidence history
                  

                elif current_action is not None:
                    if detected_action != current_action and max_prob >= start_threshold:
                        # End current action and start a new one
                        scene_actions[action_start_time] = {
                            'action': current_action,
                            'start_time': action_start_time,
                            'middle_time': (action_start_time + current_time) / 2,
                            'end_time': current_time + 1.0,  # Add buffer for completeness
                            'confidence': max_prob,
                            'original_confidence': original_confidence,  # Include the original confidence
                            'confidence_history': confidence_history,
                            
                        }
                       
                        

                        # Start new action
                        current_action = detected_action
                        action_start_time = current_time
                        original_confidence = max_prob  # Record the new original confidence
                        confidence_history = [max_prob]
                       

                    elif max_prob >= continue_threshold:
                        # Continue current action
                        confidence_history.append(max_prob)

                    else:
                        # Decay confidence and end action if below continue threshold
                        max_prob *= decay_rate
                        if max_prob < continue_threshold:
                            scene_actions[action_start_time] = {
                                'action': current_action,
                                'start_time': action_start_time,
                                'middle_time': (action_start_time + current_time) / 2,
                                'end_time': current_time + 1.0,  # Add buffer
                                'confidence': max_prob,
                                'original_confidence': original_confidence,  # Include the original confidence
                                'confidence_history': confidence_history,
                               
                            }
                            current_action = None
                            confidence_history = []  # Reset confidence history

                # Slide the window
                frame_window = frame_window[stride:]

            frame_counter += 1

        # Handle last ongoing action
        if current_action is not None:
            scene_actions[action_start_time] = {
                'action': current_action,
                'start_time': action_start_time,
                'middle_time': (action_start_time + current_time) / 2,
                'end_time': current_time + 1.0,
                'confidence': max_prob,
                'original_confidence': original_confidence,  # Include the original confidence
                'confidence_history': confidence_history,
                
            }

        container.close()
        return scene_actions

    except Exception as e:
        print(f"Error processing scene {scene_info}: {e}")
        if 'container' in locals():
            container.close()
        return {}
    
def detect_scenes(video_path: str) -> List[Tuple[int, int]]:
    """
    Detect scenes in a video using PySceneDetect with default content detection.
    
    Args:
        video_path (str): Path to the input video file.

    Returns:
        List of scene frame ranges.
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    try:
        video_manager.start()
        scene_manager.detect_scenes(video_manager)
        scene_list = scene_manager.get_scene_list()

        return [
            (scene[0].get_frames(), scene[1].get_frames())  
            for scene in scene_list
        ]

    finally:
        video_manager.release()

class ParallelActionRecognizer:
    def __init__(self,
                 model_path: str,
                 resolution: Tuple[int, int] = (290, 290),
                 frames_per_window: int = 8,
                 confidence_threshold: float = 0.75,
                 num_processes: int = None):
        
        self.model_path = model_path
        self.resolution = resolution
        self.frames_per_window = frames_per_window
        self.confidence_threshold = confidence_threshold
        self.num_processes = num_processes or max(1, mp.cpu_count() - 1)

        self.classes = ['shortpass', 'longpass', 'throw', 'goalkick',
                       'penalty', 'corner', 'freekick', 'ontarget']

        self.dict_manager = SortedDictManager()
        self.dict_manager.start()
        self.actions = self.dict_manager.SortedDict()

    

    def process_video(self, video_path: str):
        """Process the video with parallel scene processing."""
        scenes = detect_scenes(video_path)
        print(f"Detected {len(scenes)} scenes")

        # Initialize actions dictionary by type
        self.actions = {action_type: [] for action_type in self.classes}

        with mp.Pool(processes=self.num_processes) as pool:
            process_scene_partial = partial(
                process_scene_worker,
                self.model_path,
                video_path,
                resolution=self.resolution,
                frames_per_window=self.frames_per_window,
                confidence_threshold=self.confidence_threshold,
                classes=self.classes
            )

            results = []
            for scene_actions in tqdm(
                pool.imap_unordered(process_scene_partial, scenes),
                total=len(scenes),
                desc="Processing scenes"
            ):
                # Group actions by type
                for _, action_data in scene_actions.items():
                    action_type = action_data['action']
                    if action_type in self.actions:
                        self.actions[action_type].append(action_data)

    def save_actions(self, output_path: str):
        """Save detected actions to JSON."""
        # Convert all float32 values in the actions dictionary to native Python float
        def convert_to_float32(o):
            if isinstance(o, np.float32):
                return float(o)  # Convert np.float32 to Python float
            elif isinstance(o, dict):
                return {k: convert_to_float32(v) for k, v in o.items()}
            elif isinstance(o, list):
                return [convert_to_float32(i) for i in o]
            else:
                return o

        # Sort actions by start time within each action type
        sorted_actions = {}
        for action_type, actions in self.actions.items():
            sorted_actions[action_type] = sorted(actions, key=lambda x: x['start_time'])

        # Convert to JSON serializable format
        sorted_actions = convert_to_float32(sorted_actions)

        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(sorted_actions, f, indent=4)
    

def save_ontarget_frames_and_collages(actions_json_path: str, video_path: str, frames_dir: str, collages_dir: str):
    """
    Saves high-resolution frames and creates collages for all "ontarget" actions based on a JSON file.

    Args:
        actions_json_path: Path to the JSON file with detected actions.
        video_path: Path to the original video file.
        frames_dir: Directory where high-resolution frames will be saved.
        collages_dir: Directory where collages will be saved.
    """

    # Create output directories
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(collages_dir, exist_ok=True)

    # Load actions from JSON
    with open(actions_json_path, 'r') as f:
        actions = json.load(f)

    # Process only "ontarget" actions
    if 'ontarget' not in actions or not actions['ontarget']:
        print("No 'ontarget' actions found in the JSON.")
        return

    # Open the video file
    video = VideoFileClip(video_path)

    for i, action in enumerate(actions['ontarget']):
        try:
            # Extract timestamps
            start_time = action['start_time']
            middle_time = action['middle_time']
            end_time = action['end_time']

            # Create a unique folder for this action's frames
            action_frames_dir = os.path.join(frames_dir, f"ontarget_action_{i + 1}")
            os.makedirs(action_frames_dir, exist_ok=True)

            # Save frames at start, middle, and end timestamps
            timestamps = {"start": start_time, "middle": middle_time, "end": end_time}
            frame_paths = {}

            for key, timestamp in timestamps.items():
                frame_path = os.path.join(action_frames_dir, f"{key}.jpg")
                frame = video.get_frame(timestamp)  # Get the frame as a numpy array
                save_frame(frame, frame_path)
                frame_paths[key] = frame_path

            # Create a collage for the action
            if all(frame_paths.values()):
                create_action_collage(
                    frame_paths["start"],
                    frame_paths["middle"],
                    frame_paths["end"],
                    collages_dir,
                    i + 1,
                )
        except Exception as e:
            print(f"Error processing 'ontarget' action {i + 1}: {e}")

    video.close()

def create_action_timeline(actions_json_path, output_path="Simple_A3_actions_timeline.png"):
    """
    Create a detailed timeline visualization of soccer events with better layout 
    and colorblind-friendly palette.
    
    Args:
        actions_json_path: Path to the JSON file containing detected actions in format:
            { "action_type": [ { "start_time": float, "middle_time": float, 
                                "end_time": float, "confidence": float } ] }
        output_path: Where to save the PNG visualization.
    """
    # Load actions
    with open(actions_json_path) as f:
        actions_by_type = json.load(f)
    
    # Convert to flat action list for density calculation
    action_list = []
    for action_type, instances in actions_by_type.items():
        for instance in instances:
            action_list.append({
                'start_time': float(instance['start_time']),
                'middle_time': float(instance['middle_time']),
                'end_time': float(instance['end_time']),
                'action': action_type,
                'confidence': float(instance['confidence'])
            })
    
    # Get video duration
    video_duration = max(action['end_time'] for action in action_list)
    
    # Define action types with descriptive names and colorblind-friendly colors
    action_info = {
        'shortpass': {'color': '#0072B2', 'name': 'Short Pass'},  # Blue
        'longpass': {'color': '#009E73', 'name': 'Long Pass'},    # Green
        'throw': {'color': '#F0E442', 'name': 'Throw-in'},        # Yellow
        'goalkick': {'color': '#CC79A7', 'name': 'Goal Kick'},    # Magenta
        'penalty': {'color': '#D55E00', 'name': 'Penalty'},       # Red-Orange
        'corner': {'color': '#E69F00', 'name': 'Corner Kick'},    # Orange
        'freekick': {'color': '#56B4E9', 'name': 'Free Kick'},    # Sky Blue
        'ontarget': {'color': '#E31A1C', 'name': 'Shot on Target'}  # Crimson
    }
    
    # Create figure with improved layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1], figure=fig)
    
    # Top subplot: Event blocks
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot event blocks
    for i, (action_type, info) in enumerate(action_info.items()):
        y_position = i
        
        # Get all instances of this action type
        if action_type in actions_by_type:
            for instance in actions_by_type[action_type]:
                # Create event block
                rect = patches.Rectangle(
                    (float(instance['start_time']), y_position - 0.3),
                    float(instance['end_time']) - float(instance['start_time']),
                    0.6,
                    facecolor=info['color'],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=2
                )
                ax1.add_patch(rect)
                
                # Add middle time marker
                middle_x = float(instance['middle_time'])
                ax1.plot([middle_x, middle_x], 
                        [y_position - 0.3, y_position + 0.3],
                        color='white',
                        linewidth=1.5,
                        alpha=0.8,
                        zorder=3)
    
    # Customize top subplot
    ax1.set_ylim(-0.5, len(action_info) - 0.5)
    ax1.set_xlim(-2, video_duration + 2)
    ax1.set_yticks(range(len(action_info)))
    ax1.set_yticklabels([info['name'] for info in action_info.values()], fontsize=12)
    
    # Improved x-axis formatting
    def format_time(x, p):
        """Convert seconds to MM:SS format"""
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    ax1.xaxis.set_major_locator(plt.MultipleLocator(30))  # Major ticks every 30 seconds
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(10))  # Minor ticks every 10 seconds
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Add grid with custom styling
    ax1.grid(True, which='major', axis='x', linestyle='-', alpha=0.3, zorder=1)
    ax1.grid(True, which='minor', axis='x', linestyle=':', alpha=0.2, zorder=1)
    ax1.set_axisbelow(True)
    
    # Add title and labels
    ax1.set_title('Match Event Timeline', fontsize=16, pad=20, fontweight='bold')
    ax1.set_ylabel('Event Type', fontsize=12, fontweight='bold')
    
    # Bottom subplot: Event density
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Create event density plot
    time_points = np.linspace(0, video_duration, 1000)
    density = np.zeros_like(time_points)
    
    for t_idx, t in enumerate(time_points):
        count = sum(1 for action in action_list 
                   if action['start_time'] <= t <= action['end_time'])
        density[t_idx] = count
    
    # Plot density
    ax2.fill_between(time_points, density, alpha=0.3, color='gray')
    ax2.plot(time_points, density, color='gray', linewidth=1.5)
    
    # Apply same x-axis formatting
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    ax2.xaxis.set_major_locator(plt.MultipleLocator(30))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(10))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Customize bottom subplot
    ax2.set_xlim(-2, video_duration + 2)
    ax2.set_ylabel('Concurrent\nEvents', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Time (MM:SS)', fontsize=12, fontweight='bold')
    ax2.grid(True, which='major', linestyle='-', alpha=0.3)
    ax2.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Add custom legend to the side
    legend_elements = [patches.Patch(facecolor=info['color'], edgecolor='black', label=info['name']) 
                       for info in action_info.values()]
    
    ax_legend = fig.add_subplot(gs[:, 1])
    ax_legend.axis('off')
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=10, title='Event Types')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


from moviepy.editor import VideoFileClip

def create_ontarget_clips(actions_json_path: str, video_path: str, output_dir: str, variant: str):
    """
    Creates video clips for each "ontarget" action from the actions JSON.

    Args:
        actions_json_path: Path to the JSON file with detected actions.
        video_path: Path to the original video file.
        output_dir: Directory where clips will be saved.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Load actions from JSON
    with open(actions_json_path, 'r') as f:
        actions = json.load(f)

    # Process only "ontarget" actions
    if 'ontarget' not in actions or not actions['ontarget']:
        print("No 'ontarget' actions found in the JSON.")
        return

    # Open the video file using MoviePy
    video = VideoFileClip(video_path)
    
    # Loop through all "ontarget" actions
    for i, action in enumerate(actions['ontarget']):
        start_time = action['start_time']
        end_time = action['end_time']

        try:
            # Extract the clip for the action
            clip = video.subclip(start_time, end_time)

            # Define output path
            output_clip_path = os.path.join(output_dir, f"from MoviNet {variant} ontarget_clip_{i+1}.mp4")

            # Write the clip to file
            clip.write_videofile(output_clip_path, codec="libx264", audio=False)
            print(f"Saved clip for 'ontarget' action {i+1}: {output_clip_path}")

        except Exception as e:
            print(f"Error creating clip for action {i+1}: {e}")

    # Close the video file
    video.close()


def main():

    video_path = "A point apiece ⚖️ ｜ HIGHLIGHTS ｜ Arsenal v Brighton & Hove Albion (1-1) ｜ Havertz ｜ Premier league [0dV0RrYuE14].mp4"
    output_json = "Alternative_A3_detected_actions_parallel.json"
    output_dir = "Alternative Ontarget Actions Frames"
    collages_dir = "Alternative A3 Ontarget Actions Collages"

    recognizer = ParallelActionRecognizer(
        model_path="model_a3_operations_using_fp16_with_8_frames_at_single_batch_from_98.40%_model_at_training_split_75.0_25.0.tflite",
        resolution=(256, 256),
        frames_per_window=8,
        confidence_threshold=0.85,
        num_processes=None
    )

    recognizer.process_video(video_path)
    recognizer.save_actions(output_json)
    save_ontarget_frames_and_collages(output_json, video_path, output_dir, collages_dir)
    create_action_timeline(output_json)
    create_ontarget_clips(output_json,video_path, 'Test Clips From Alternative A3', 'A3')


if __name__ == "__main__":
    main()
   
