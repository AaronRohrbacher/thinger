*Thinger isolates usage of a word in a given video using SpeechRecognition and extracts all uses of that word into a single video. Supposedly. Uses Google Speech Recognition as of the time of this'y here writing. Many engines are available via the Python dependency.*

*Python Dependencies:*
Easiest to install dependencies individually, as some have dependencies that annoy one another when installing all at once.
pip install setuptools
pip install decorator==4.4.2
pip install imageio==2.4.1
pip install imageio-ffmpeg==0.4.5
pip install moviepy==1.0.3
pip install SpeechRecognition
pip install pydub

*This script features three adjustable knobs (no warranty, all sales final):*

`308:  result_path, count = finder.process_video(target_word, output_path, min_confidence=0.65)`
  --Adjust the min_confidence argument to increase or decrease speech recognition sensitivity to the given word you're looking for in the video's audio.
  --Note: It is in this main block where you can also configure the path to the video you want to analyze, and the word you want to isolate.

`92:     def __init__(self, video_path, buffer_before=0.00001, buffer_after=0.00001):`
  --Adjust the amount of time before and after the word isolation to include in the compiled video.