# Usage
1. Install SAM2
2. use ffmpeg to extract images from video for sam to process 
```bash
mkdir out
ffmpeg -i <video>.mp4 -q:v 2 -start_number 0 <video-dir>/'%05d.jpg'
```
3. set appropriate names in momag.py 
  - `H, W, INPUT_DIR, INPUT_FILE, obj_id_map, video_dir` should all be set appropriately
  
4. run using `python momag.py`
