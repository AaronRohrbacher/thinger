from moviepy.editor import VideoFileClip, concatenate_videoclips
import speech_recognition as sr
import os
from pydub import AudioSegment
import numpy as np
from pydub.silence import detect_nonsilent
import multiprocessing as mp
from functools import partial
import psutil
import time
from datetime import timedelta
import uuid
import tempfile
import shutil
from pathlib import Path

def process_chunk(chunk_params):
    """
    Static function to process a single chunk in parallel.
    """
    (chunk_idx, start_time, end_time, temp_dir, audio_path, 
     target_word, min_confidence, duration, buffer_before, buffer_after) = chunk_params
    
    # Create unique identifiers for this process's temporary files
    process_id = uuid.uuid4().hex[:8]
    temp_chunk_path = os.path.join(temp_dir, f"chunk_{process_id}_{chunk_idx}.wav")
    instances = []
    
    try:
        # Initialize recognizer for this process
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        
        # Load and process audio chunk
        audio = AudioSegment.from_wav(audio_path)
        chunk = audio[int(start_time * 1000):int(end_time * 1000)]
        chunk.export(temp_chunk_path, format="wav")
        
        with sr.AudioFile(temp_chunk_path) as source:
            audio_data = recognizer.record(source)
            try:
                response = recognizer.recognize_google(audio_data, show_all=True)
                
                if response and 'alternative' in response:
                    for alt in response['alternative']:
                        if 'confidence' in alt and alt['confidence'] >= min_confidence:
                            text = alt['transcript'].lower()
                            words = text.split()
                            
                            for word_idx, word in enumerate(words):
                                if word == target_word.lower():
                                    # Calculate approximate word position
                                    word_start = start_time + (end_time - start_time) * (word_idx / len(words))
                                    window_start = max(0, word_start - 0.5)
                                    window_end = min(duration, word_start + 1.0)
                                    
                                    # Extract and analyze the word audio
                                    word_audio = audio[int(window_start * 1000):int(window_end * 1000)]
                                    nonsilent = detect_nonsilent(
                                        word_audio,
                                        min_silence_len=50,
                                        silence_thresh=-40
                                    )
                                    
                                    if nonsilent:
                                        actual_start = window_start + (nonsilent[0][0] / 1000.0)
                                        actual_end = window_start + (nonsilent[0][1] / 1000.0)
                                        
                                        clip_start = max(0, actual_start - buffer_before)
                                        clip_end = min(duration, actual_end + buffer_after)
                                        instances.append((clip_start, clip_end))
                                        
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"\nWarning: Error processing chunk {chunk_idx}: {str(e)}")
                
    except Exception as e:
        print(f"\nError processing chunk {chunk_idx}: {str(e)}")
    finally:
        # Clean up temporary chunk file
        try:
            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path)
        except Exception as e:
            print(f"\nWarning: Could not remove temporary file {temp_chunk_path}: {str(e)}")
    
    return instances

class VideoWordFinder:
    def __init__(self, video_path, buffer_before=0.00001, buffer_after=0.00001):
        """
        Initialize the VideoWordFinder with a video file path and timing parameters.
        
        Args:
            video_path (str): Path to the input video file
            buffer_before (float): Seconds of buffer before each word
            buffer_after (float): Seconds of buffer after each word
        """
        self.video_path = video_path
        self.video = VideoFileClip(video_path)
        self.audio = self.video.audio
        self.duration = self.video.duration
        self.buffer_before = buffer_before
        self.buffer_after = buffer_after
        self.temp_dir = None
        
    def _create_temp_dir(self):
        """Create a temporary directory for processing files."""
        self.temp_dir = tempfile.mkdtemp(prefix="word_finder_")
        return self.temp_dir
        
    def _cleanup_temp_dir(self):
        """Clean up temporary directory and its contents."""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Error cleaning up temporary directory: {str(e)}")
                
    def extract_audio(self):
        """Extract audio from video and save as WAV."""
        if not self.temp_dir:
            self._create_temp_dir()
        temp_audio = os.path.join(self.temp_dir, "full_audio.wav")
        self.audio.write_audiofile(temp_audio)
        return temp_audio
        
    def segment_audio(self, chunk_duration=5):
        """
        Split audio into smaller chunks for more accurate processing.
        
        Args:
            chunk_duration (int): Duration of each chunk in seconds
            
        Returns:
            list: List of time ranges for each chunk
        """
        chunks = []
        start_time = 0
        while start_time < self.duration:
            end_time = min(start_time + chunk_duration, self.duration)
            chunks.append((start_time, end_time))
            start_time = end_time
        return chunks
        
    def find_word_instances(self, target_word, min_confidence=0.01):
        """
        Find all instances of a target word using parallel processing.
        
        Args:
            target_word (str): Word to search for
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            list: List of time ranges where the word appears
        """
        # Create temporary directory for processing
        self._create_temp_dir()
        
        try:
            # Extract audio and prepare chunks
            audio_path = self.extract_audio()
            chunks = self.segment_audio()
            
            # Determine optimal number of processes
            cpu_count = psutil.cpu_count(logical=True)  # Physical cores only
            num_processes = max(1, min(cpu_count - 1, len(chunks)))  # Leave one core free
            
            # Prepare parameters for parallel processing
            chunk_params = [
                (i, start, end, self.temp_dir, audio_path, target_word,
                 min_confidence, self.duration, self.buffer_before, self.buffer_after)
                for i, (start, end) in enumerate(chunks)
            ]
            
            # Initialize progress tracking
            total_chunks = len(chunks)
            print(f"Processing {total_chunks} chunks using {num_processes} processes...")
            start_time = time.time()
            processed_chunks = 0
            word_instances = []
            
            # Process chunks in parallel
            with mp.Pool(num_processes) as pool:
                for chunk_instances in pool.imap_unordered(process_chunk, chunk_params):
                    processed_chunks += 1
                    
                    # Update progress
                    elapsed_time = time.time() - start_time
                    chunks_per_second = processed_chunks / elapsed_time
                    remaining_chunks = total_chunks - processed_chunks
                    est_time_remaining = remaining_chunks / chunks_per_second if chunks_per_second > 0 else 0
                    
                    print(f"\rProcessing chunk {processed_chunks}/{total_chunks} "
                          f"({(processed_chunks/total_chunks*100):.1f}%) - "
                          f"Est. time remaining: {str(timedelta(seconds=int(est_time_remaining)))} "
                          f"({chunks_per_second:.2f} chunks/sec)", end="")
                    
                    word_instances.extend(chunk_instances)
            
            print("\nMerging overlapping instances...")
            word_instances = self._merge_overlapping_instances(word_instances)
            return word_instances
            
        finally:
            # Clean up temporary files
            self._cleanup_temp_dir()
            
    def _merge_overlapping_instances(self, instances):
        """
        Merge overlapping word instances to avoid duplicates.
        
        Args:
            instances (list): List of (start, end) times
            
        Returns:
            list: Merged list of time ranges
        """
        if not instances:
            return []
        
        # Sort by start time
        instances = sorted(instances, key=lambda x: x[0])
        merged = [instances[0]]
        
        for current in instances[1:]:
            previous = merged[-1]
            
            # Check for overlap
            if current[0] <= previous[1]:
                # Merge overlapping instances
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)
        
        return merged
        
    def compile_clips(self, time_ranges, output_path):
        """
        Compile video clips of word instances into a single video.
        
        Args:
            time_ranges (list): List of (start, end) times for each clip
            output_path (str): Path to save the compiled video
            
        Returns:
            str: Path to the compiled video file
        """
        clips = []
        for start, end in time_ranges:
            try:
                clip = self.video.subclip(start, end)
                clips.append(clip)
            except Exception as e:
                print(f"Error creating subclip {start}-{end}: {e}")
                continue
        
        if clips:
            try:
                final_clip = concatenate_videoclips(clips)
                final_clip.write_videofile(output_path, codec="libx264")
                return output_path
            except Exception as e:
                print(f"Error concatenating clips: {e}")
                return None
        return None
        
    def process_video(self, target_word, output_path, min_confidence=0.8):
        """
        Main function to process video and create compilation.
        
        Args:
            target_word (str): Word to search for
            output_path (str): Path to save the compiled video
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            tuple: (output_path, number of instances found)
        """
        print(f"Searching for instances of '{target_word}'...")
        instances = self.find_word_instances(target_word, min_confidence)
        
        if instances:
            print(f"\nFound {len(instances)} instances of '{target_word}'")
            output_file = self.compile_clips(instances, output_path)
            return output_file, len(instances)
        else:
            print(f"\nNo instances of '{target_word}' found in the video")
            return None, 0
            
    def __del__(self):
        """Clean up video and audio objects."""
        if hasattr(self, 'video'):
            self.video.close()
        if hasattr(self, 'audio'):
            self.audio.close()
        self._cleanup_temp_dir()

# Example usage
if __name__ == "__main__":
    video_path = "../hey_full_vid.mp4"  # Update this to your actual video path
    target_word = "hey"
    output_path = "hey_compiled.mp4"
    
    finder = VideoWordFinder(video_path)
    result_path, count = finder.process_video(target_word, output_path, min_confidence=0.65)