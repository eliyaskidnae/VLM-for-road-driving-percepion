
## Driving Video Understanding with Vision-Language Models

### 1. Project Overview 

This project explores how lightweight Vision-Language Models (VLMs) can understand and explain real-world driving videos. The system takes a driving video together with a natural-language prompt and generates human-readable descriptions of the current road environment.
The video is analyzed periodically using overlapping time windows. For example, with a 10-second context window and a 5-second interval, the system analyzes 0–10 s, 5–15 s, 10–20 s, and so on. The same prompt can be used throughout the video, or the user can change the question while the video is running.
The main goal is to compare different lightweight VLMs in terms of their ability to recognize road layout, vehicles, pedestrians, cyclists, traffic signs, traffic lights, obstacles, visibility conditions, and potentially important driving situations. The project also considers practical factors such as inference latency and GPU memory usage. Another part of the experiment investigates whether giving the model its previous textual observation improves temporal understanding or causes it to repeat earlier predictions.

### 2. User Interface

The system provides a simple web interface built with Gradio. The driving video is displayed together with a text box where the user can enter a question or instruction. 

The interface periodically sends the current video segment and prompt to the VLM and displays the generated explanation. The user can control the analysis interval, video context length, frame-sampling rate, and output length. Previous model responses are also displayed as a reasoning history.
<!-- add media video  -->
[▶ Watch the driving demo](media/driving_video.mp4)

### 3. Experimental Hardware
The experiments were performed using:

        ```
        GPU: NVIDIA GeForce RTX 4050
        GPU memory: 6 GB
        Available CUDA memory: approximately 5.64 GiB
        ```


### 4. Models Tested

#### A) Qwen2.5-VL-3B-Instruct
Qwen/Qwen2.5-VL-3B-Instruct is the main baseline model tested so far.
Because of the 6 GB GPU limitation, the model is loaded using 4-bit quantization. It can successfully run on the available hardware and provides reasonable driving-scene descriptions.
The model performs well at understanding the general environment. It consistently recognizes that the video contains a multi-lane highway, surrounding traffic, clear weather, lane markings, vegetation, and other large visual elements. It also successfully detected vehicles and, later in the video, a motorcycle.
A more difficult task is recognizing small traffic signs and text. When signs are distant or occupy only a small portion of the frame, the model sometimes fails to read them or confidently reports information that is not clearly visible. When a large overhead freeway sign becomes close to the vehicle, recognition improves considerably.
This suggests that Qwen2.5-VL-3B is stronger at coarse scene understanding than at fine visual details such as small road signs and text.


### B)Qwen3-VL-2B-Instruct
Qwen/Qwen3-VL-2B-Instruct is a smaller but newer-generation Vision-Language Model. It is particularly interesting because it has fewer parameters than Qwen2.5-VL-3B while using a newer multimodal architecture.
The model is expected to provide good general scene understanding while requiring less GPU memory and offering faster inference. This makes it a strong candidate for periodic driving-video analysis on limited hardware. It should be suitable for recognizing the overall road layout, vehicles, traffic conditions, and larger visual elements. The average latency is around 3–8 seconds per 10-second video window.

### C) Qwen3-VL-4B-Instruct 
Qwen/Qwen3-VL-4B-Instruct is a larger Qwen3-VL model intended to provide stronger visual reasoning than the 2B version. It is expected to perform better on more detailed tasks such as traffic-sign recognition, text reading, object relationships, and more complex driving-scene explanations.
For this project, the model is particularly interesting because it may provide a better balance between accuracy and efficiency than both Qwen3-VL-2B and Qwen2.5-VL-3B. However, because of the 6 GB GPU limitation, it should be tested using 4-bit quantization and controlled video-resolution and frame-sampling settings.
Compared with the 2B model, it is expected to use more GPU memory and require longer inference time, but potentially produce more detailed and accurate descriptions.
On the RTX 4050 6 GB GPU used in this project, the expected latency is approximately 6–15 seconds per 10-second video window under 4-bit quantization and reduced-resolution video settings. The actual latency will depend strongly on the number of sampled frames and the visual resolution used during inference.
The average latency is around 6–15 seconds per 10-second video window.



#### D) Qwen2.5-VL-7B-Instruct
Qwen/Qwen2.5-VL-7B-Instruct was also considered.

However, it could not run successfully on the available RTX 4050 because of the 6 GB GPU-memory limitation.
The larger model requires substantially more memory for model parameters, the vision encoder, visual tokens, activations, and generation.
Therefore, Qwen2.5-VL-7B is currently excluded from local experiments because of hardware limitations.

#### Effect of Previous Observations
Two configurations of Qwen2.5-VL-3B were tested.
In the first configuration, the previous model response was included in the next model input. The intention was to provide short-term temporal memory.
However, this caused a noticeable problem. The model began repeating similar descriptions across consecutive video segments and sometimes carried previously mentioned traffic-sign information into later predictions.

When the previous observation was removed, the model became more responsive to the actual content of each current video segment. It detected changes such as the approaching overpass, large road signs, changing traffic, and the motorcycle more clearly.
Therefore, the initial results suggest that independent video-window reasoning performs better than simply inserting the previous free-text response as memory.
This does not mean temporal memory is unnecessary. Instead, a more structured memory representation may be preferable in future work.