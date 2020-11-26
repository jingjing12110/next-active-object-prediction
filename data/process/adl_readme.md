This is ADLdataset introduced in our CVPR 2012 paper:

Hamed Pirsiavash, Deva Ramanan, "Detecting Activities of Daily Living in First-Person Camera Views," CVPR 2012.

It includes the following folders:
"ADL_videos/":
These are large MP4 files so please download them one by one from http://deepthought.ics.uci.edu/ADLdataset/ADL_videos/ and copy them to a folder named "ADL_videos/"

"ADL_annotations/object_annotation/":
- "object_annot_P_01.txt": each row corresponds to an object instance in the video. 
order of columns: "object_track_id", "x1", "y1", "x2", "y2", "frame number", "1 if the object is active and 0 otherwise", "object label"
[x1, y1] and [x2 y2] correspond to the top-left and bottom-right corner of the bounding box respectively. Note that a long track of an object may be divided to multiple tracks so may have multiple "object_track_id"s. We do not use "object_track_id" in our paper. Moreover, annotations are not super clean and there might be some mistakes including not annotated objects and multiple annotations for one object.
- "object_annot_P_01_annotated_frames.txt": list of annotated frames.


"ADL_annotations/action_annotation/":
- "action_list.txt": list of our activities of daily livings.
- "P_01.txt": each row corresponds to one action. start frame, end frame, and action label. It may include a note. 
Note that actions can be overlapping. Some actions have multiple stages. Those are interrupted by other actions for a long time.


"ADL_detected_objects/":
results of part-based model for object detection. The models are trained on the first 6 videos and tested on the rest. For some objects, separate models are learned for active and passive object instances.


"ADL_code/"
- "main.m" runs our action classifier. It uses libsvm "http://www.csie.ntu.edu.tw/~cjlin/libsvm/" included in the folder named "third_party/". 
It uses object models trained on the first 6 videos to classify actions on the other videos. It performs leave-one-out cross validation and results in 37% accuracy on 18 action categories. Due to some modifications on the dataset, this is a little less than the reported number on the original paper (40%).

- "dump_frames.m" dumps videos into frames and then shows object annotations on the annotated frames.

