# tfrecord-creation-example


# Create a single record

```python

import tensorflow.compat.v1 as tf
from object_detection.utils import dataset_util
import os



train_dir="."

# Path for train tfrecord
train_tfrec_path = os.path.join(train_dir, 'train.tfrecord')

# Writer for the train dataset
train_writer = tf.python_io.TFRecordWriter(train_tfrec_path)


tf_example = tf.train.Example(features=tf.train.Features(feature={
'train/facebbx_h': dataset_util.int64_feature(123),
'train/facebbx_w': dataset_util.int64_feature(123),
'train/facebbx_x': dataset_util.int64_feature(123),
'train/facebbx_y': dataset_util.int64_feature(123),
'train/image_frame_height': dataset_util.int64_feature(123),
'train/image_frame_name': dataset_util.bytes_list_feature(["test".encode()]),
'train/image_frame_width': dataset_util.int64_feature(123),
'train/landmarks': dataset_util.float_list_feature([1,2,3]),
'train/landmarks_occ': dataset_util.int64_list_feature([1,2,3]),
}))

train_writer.write(tf_example.SerializeToString())


# Finished with the train dataset
train_writer.close()

```

# Read a single record

```python
import tensorflow as tf

for example in tf.compat.v1.python_io.tf_record_iterator("./train.tfrecord"):
    val = tf.train.Example.FromString(example)

    with open('out.txt', 'w') as f:
        f.write(str(val))

    # Just want one in out.txt. In case there is more exit here
    exit
```

Example output:

```pbtxt
features {
  feature {
    key: "train/facebbx_h"
    value {
      int64_list {
        value: 123
      }
    }
  }
  feature {
    key: "train/facebbx_w"
    value {
      int64_list {
        value: 123
      }
    }
  }
  feature {
    key: "train/facebbx_x"
    value {
      int64_list {
        value: 123
      }
    }
  }
  feature {
    key: "train/facebbx_y"
    value {
      int64_list {
        value: 123
      }
    }
  }
  feature {
    key: "train/image_frame_height"
    value {
      int64_list {
        value: 123
      }
    }
  }
  feature {
    key: "train/image_frame_name"
    value {
      bytes_list {
        value: "test"
      }
    }
  }
  feature {
    key: "train/image_frame_width"
    value {
      int64_list {
        value: 123
      }
    }
  }
  feature {
    key: "train/landmarks"
    value {
      float_list {
        value: 1.0
        value: 2.0
        value: 3.0
      }
    }
  }
  feature {
    key: "train/landmarks_occ"
    value {
      int64_list {
        value: 1
        value: 2
        value: 3
      }
    }
  }
}

```

# Read multiple records and decode

```python
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

TF_RECORD_PATH="./train.tfrecord"


for example in tf.compat.v1.python_io.tf_record_iterator(TF_RECORD_PATH):

    val = tf.train.Example.FromString(example)

    bbx_h = val.features.feature["train/facebbx_h"].int64_list.value[0]
    bbx_w = val.features.feature["train/facebbx_w"].int64_list.value[0]

    bbx_tl_x = val.features.feature["train/facebbx_x"].int64_list.value[0]
    bbx_tl_y = val.features.feature["train/facebbx_y"].int64_list.value[0]

    bbx_br_x = bbx_tl_x + bbx_w
    bbx_br_y = bbx_tl_y + bbx_h

    img_path = val.features.feature["train/image_frame_name"].bytes_list.value[0].decode()

    img_height = val.features.feature["train/image_frame_height"].int64_list.value[0]
    img_width = val.features.feature["train/image_frame_width"].int64_list.value[0]


    keypoints = val.features.feature["train/landmarks"].float_list.value
```
